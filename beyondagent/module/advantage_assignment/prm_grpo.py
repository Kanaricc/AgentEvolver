# -*- coding: utf-8 -*-
# PRM step → (optional) group-level standardization on steps → per-trajectory projection (optional) → suffix-sum on steps → broadcast to tokens
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch
import math

# =========================
# Hyper & small utilities
# =========================

@dataclass
class PRMHyper:
    # 权重：一致性步的权重大，不一致性步的权重小（用于 allocation / allocation_c）
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2   # 成功轨迹里的 BAD 步权重
    neg_unconsistent_scale: float = 0.2   # 失败轨迹里的 GOOD 步权重
    eps: float = 1e-8
    do_batch_norm: bool = True          # 是否做组内 z-score（按 step 级，allocation_c/decouple 会用到）
    equal_trajectory_weight: bool = True  # True=每条轨迹等权（GRPO）；False=把所有 step 拉平成一个大样本（GSPO）
    fix_base: float = 0.2                 # fix 方案的基础幅度（good=+base, bad=-base）
    alpha: float = 1.0                   # PRM权重平衡系数
    orm_distribution: str = "last_step"   # ORM分配方式："last_step" 或 "all_steps"

def _ensure_tensor(x, device, dtype=None):
    """确保输入转换为指定设备和类型的张量"""
    if torch.is_tensor(x):
        t = x.to(device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)

def _num_steps_from_step_ids(step_ids_row: torch.Tensor) -> int:
    """根据step_ids计算轨迹中的步数"""
    if step_ids_row.numel() == 0:
        return 0
    m = torch.amax(step_ids_row)
    return int(m.item() + 1) if m.item() >= 0 else 0

def _align_flags(flags: List[bool], K: int, is_success: bool) -> List[bool]:
    """对齐flags长度与步数K，不足时用默认值填充"""
    if len(flags) == K:
        return list(flags)
    default_flag = True if is_success else False
    if len(flags) < K:
        return list(flags) + [default_flag] * (K - len(flags))
    else:
        return list(flags[:K])

# =========================
# Group normalization helpers (group-wise, step-level)
# =========================

def _group_zscore_on_steps(
    step_rewards_raw: List[List[float]],
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """对 step 奖励做“组内”减均值/除方差标准化。
    - equal_trajectory_weight=True: 每条轨迹等权；组均值 = 轨迹均值的均值；
      组方差 = 轨迹内相对组均值的均方差的均值（second-moment around group mean）
    - equal_trajectory_weight=False: 拉平本组所有 step 一起算
    """
    if not hyper.do_batch_norm:
        return [list(r) for r in step_rewards_raw]

    B = len(step_rewards_raw)
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)

    step_rewards_std: List[List[float]] = [[] for _ in range(B)]
    for _, idxs in g2idx.items():
        if hyper.equal_trajectory_weight:
            # 1) 组均值：轨迹均值的等权平均
            traj_means = []
            for i in idxs:
                ri = step_rewards_raw[i]
                if ri:
                    traj_means.append(sum(ri) / len(ri))
            if len(traj_means) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = float(sum(traj_means) / len(traj_means))
                # 2) 组方差：先对每条轨迹围绕 mu_g 求均方差，再对轨迹等权平均
                second_moments = []
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    second_moments.append(sum((x - mu_g) * (x - mu_g) for x in ri) / len(ri))
                var_g = float(sum(second_moments) / len(second_moments)) if second_moments else 0.0
                sd_g = float(math.sqrt(var_g + hyper.eps))
        else:
            # 拉平：把本组所有 step 拼在一起
            flat = []
            for i in idxs:
                flat.extend(step_rewards_raw[i])
            if len(flat) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                t = torch.tensor(flat, dtype=torch.float32)
                mu_g = float(t.mean().item())
                sd_g = float(max(t.std(unbiased=False).item(), hyper.eps))

        inv = 1.0 / (sd_g + 1e-12)
        for i in idxs:
            ri = step_rewards_raw[i]
            if not ri:
                step_rewards_std[i] = []
            else:
                step_rewards_std[i] = [float((x - mu_g) * inv) for x in ri]
    return step_rewards_std

def _per_traj_scale_to_target_sum(
    r_std: List[float],
    target_sum: float,
    eps: float,
) -> List[float]:
    """将轨迹的step奖励按比例缩放，使总和等于目标值
    
    当当前总和接近0时，将目标值均匀分配给所有step
    
    Args:
        r_std: 标准化后的step奖励列表
        target_sum: 目标总和值
        eps: 数值稳定性常数
        
    Returns:
        缩放后的step奖励列表
    """
    if len(r_std) == 0:
        return []
    cur = sum(r_std)
    if abs(cur) <= eps:
        return [target_sum / len(r_std) for _ in r_std]
    scale = target_sum / cur
    return [float(x * scale) for x in r_std]



# =========================
# Builders for 4 schemes
# =========================
def _build_fix(
    orm_scores: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案1：fix —— 固定基数奖励构造 + 轨迹最后step的ORM符号调整
    
    算法原理：
      1. 基础奖励构造：根据step flags构造固定幅度的step-level奖励
         - GOOD步骤：+fix_base
         - BAD步骤：-fix_base
      2. 轨迹最后step的ORM符号调整：根据ORM分数符号，在轨迹最后一步添加方向控制项
         - 成功轨迹(ORM>0)：最后一步奖励 += +1
         - 失败轨迹(ORM≤0)：最后一步奖励 += -1
    
    优势函数特性：
      - 奖励幅度固定，不随轨迹长度变化
      - 通过ORM符号调整确保奖励方向与ORM一致
      - 适用于简单的二元奖励场景
    
    Args:
        orm_scores (torch.Tensor): 完整ORM分数，shape (B,)，用于确定奖励方向
        step_flags (List[List[bool]]): 每条轨迹的step级别GOOD/BAD标志
        step_ids (torch.Tensor): step标识符，shape (B, L_resp)，-1表示非response token
        group_ids (torch.Tensor): 组标识符，用于组内归一化，shape (B,)
        hyper (PRMHyper): PRM超参数配置，主要使用fix_base参数
        
    Returns:
        List[List[float]]: 每条轨迹的step-level奖励列表，长度与step数一致
        
    Example:
        orm_scores = [2.5, -1.5]  # 第一条轨迹成功，第二条轨迹失败
        step_flags = [[True, False, True], [False, True]]  # 两条轨迹的step标志
        hyper.fix_base = 0.2
        # 输出示例：
        # [[0.2, -0.2, 0.2],  # 第一条轨迹：+0.2-0.2+0.2+1.0 = 1.2
        #  [-0.2, 0.2]]       # 第二条轨迹：-0.2+0.2-1.0 = -1.0
    """
    B = step_ids.size(0)
    prm_rewards_raw: List[List[float]] = []
    base = float(hyper.fix_base)
    
    # ---- 1. 构造原始 PRM 奖励 ----
    for i in range(B):
        # 获取当前轨迹的step数量
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            prm_rewards_raw.append([]); continue
            
        # 对齐step flags长度，确保与step数量一致
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)
        
        # 构造基础PRM奖励：GOOD步骤为+base，BAD步骤为-base
        r = [(+base if f else -base) for f in flags]
        
        # 基于ORM分数符号调整最后一步奖励，确保整体奖励方向与ORM一致
        orm_sign = 1.0 if float(orm_scores[i].item()) > 0 else -1.0
        if len(r) > 0:
            r[-1] += orm_sign

        prm_rewards_raw.append(r)

    # ---- 2. 组内 z-score (标准化) ----
    # 使用 _group_zscore_on_steps 来做组内标准化
    prm_rewards_norm = _group_zscore_on_steps(prm_rewards_raw, group_ids, hyper)
    return prm_rewards_norm

def _build_allocation(
    orm_scores: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    方案2：allocation —— 一致性权重瓜分 + 组内减均值中心化
    
    算法原理：
      1. 一致性权重瓜分：根据ORM符号和step flags为每个step分配权重，确保轨迹奖励和等于ORM符号
         - 成功轨迹：一致性步骤权重高，不一致性步骤权重低
         - 失败轨迹：一致性步骤权重低，不一致性步骤权重高
      2. 组内减均值中心化：对整个batch的step奖励进行组内中心化处理，获得真正的优势函数
      
    优势函数特性：
      - 保持奖励符号与ORM一致
      - 通过权重分配体现步骤重要性差异
      - 组内减均值得到相对优势值
      
    Args:
        orm_scores (torch.Tensor): 完整ORM分数，shape (B,)，用于确定奖励方向和权重分配策略
        step_flags (List[List[bool]]): 每条轨迹的step级别GOOD/BAD标志
        step_ids (torch.Tensor): step标识符，shape (B, L_resp)
        group_ids (torch.Tensor): 组标识符，用于组内归一化，shape (B,)
        hyper (PRMHyper): PRM超参数配置
        
    Returns:
        List[List[float]]: 每条轨迹的step-level优势奖励，已进行组内减均值处理
    """
    B = step_ids.size(0)

    # ---- 第一阶段：生成原始PRM奖励（一致性权重瓜分，逐轨迹奖励和 = ORM符号）----
    step_rewards_raw: List[List[float]] = []
    for i in range(B):
        # 获取当前轨迹的step数量
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            step_rewards_raw.append([]); continue
            
        # 根据ORM分数符号确定轨迹类型和权重分配策略
        is_success = bool(orm_scores[i].item() > 0)
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        
        # 统计GOOD和BAD步骤数量
        n_g = sum(1 for f in flags if f); n_b = K - n_g
        
        # 根据轨迹类型设置权重参数
        if is_success:
            # 成功轨迹：一致性步骤(GOOD)权重高，不一致性步骤(BAD)权重低
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            # 失败轨迹：一致性步骤(BAD)权重低，不一致性步骤(GOOD)权重高
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0
            
        # 权重归一化：确保轨迹总奖励等于ORM符号
        total_w = n_g * w_g + n_b * w_b
        unit = 0.0 if total_w <= hyper.eps else (1.0 / total_w)
        r_raw = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags]
        step_rewards_raw.append([float(x) for x in r_raw])
    
    # ---- 第二阶段：组内 z-score 标准化（获得真正的优势函数）----
    # 使用 _group_zscore_on_steps 函数进行标准化
    r_norm = _group_zscore_on_steps(step_rewards_raw, group_ids, hyper)
    
    return r_norm

def _build_allocation_c(
    orm_scores: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    方案3：allocation_c —— 一致性瓜分（同号） → 组内归一化 → 按比例缩放投影（∑=±1）
    
    算法原理：
      1. 一致性权重瓜分：根据ORM符号和step flags为每个step分配权重
         - 成功轨迹：一致性步骤权重高，不一致性步骤权重低
         - 失败轨迹：一致性步骤权重低，不一致性步骤权重高
         - 确保轨迹奖励和等于ORM符号（∑=±1）
      2. 组内归一化：对整个batch的step奖励进行组内标准化处理
         - 使用归一化得到标准化优势值
         - 消除不同组间的绝对奖励差异
      3. 按比例缩放投影：将标准化后的奖励按比例缩放，确保轨迹奖励和等于ORM值
         - 保持奖励分布形状，仅调整幅度
    
    优势函数特性：
      - 通过权重分配体现步骤重要性差异
      - 组内标准化消除绝对尺度影响
      - 保持奖励符号与ORM一致
      - 通过比例缩放确保奖励幅度与ORM匹配
    
    Args:
        orm_scores (torch.Tensor): 完整ORM分数，shape (B,)，用于确定奖励方向和最终幅度
        step_flags (List[List[bool]]): 每条轨迹的step级别GOOD/BAD标志
        step_ids (torch.Tensor): step标识符，shape (B, L_resp)
        group_ids (torch.Tensor): 组标识符，用于组内标准化，shape (B,)
        hyper (PRMHyper): PRM超参数配置
            - consistent_scale: 一致性步骤的权重
            - pos_unconsistent_scale: 成功轨迹中不一致性步骤的权重
            - neg_unconsistent_scale: 失败轨迹中不一致性步骤的权重
            - eps: 数值稳定性常数
            
    Returns:
        List[List[float]]: 每条轨迹的step-level优势奖励
            - 已进行组内归一化
            - 已按比例缩放使轨迹总和等于ORM值
    """
    B = step_ids.size(0)
    
    # ---- 第一阶段：生成原始PRM奖励（一致性权重瓜分，逐轨迹奖励和 = ORM符号）----
    step_rewards_raw: List[List[float]] = []
    for i in range(B):
        # 获取当前轨迹的step数量
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            step_rewards_raw.append([]); continue
            
        # 根据ORM分数符号确定轨迹类型和权重分配策略
        is_success = bool(orm_scores[i].item() > 0)
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        
        # 统计GOOD和BAD步骤数量
        n_g = sum(1 for f in flags if f); n_b = K - n_g
        
        # 根据轨迹类型设置权重参数
        if is_success:
            # 成功轨迹：一致性步骤(GOOD)权重高，不一致性步骤(BAD)权重低
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            # 失败轨迹：一致性步骤(BAD)权重低，不一致性步骤(GOOD)权重高
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0
            
        # 权重归一化：确保轨迹总奖励等于ORM符号
        total_w = n_g * w_g + n_b * w_b
        unit = 0.0 if total_w <= hyper.eps else (1.0 / total_w)
        r_raw = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags]
        step_rewards_raw.append([float(x) for x in r_raw])
    
    # ---- 第二阶段：组内 z-score 标准化 ----
    # 使用 _group_zscore_on_steps 函数进行标准化
    r_norm = _group_zscore_on_steps(step_rewards_raw, group_ids, hyper)
    
    # ---- 第三阶段：按比例缩放投影（逐轨迹 ∑=ORM值）----
    out: List[List[float]] = []
    for i in range(B):
        # 按比例缩放，使轨迹总和等于ORM值
        out.append(_per_traj_scale_to_target_sum(r_norm[i], float(orm_scores[i].item()), eps=hyper.eps))
        
    return out

def _build_decouple(
    orm_full_scores: torch.Tensor,  # 完整的ORM分数
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper
) -> List[List[float]]:
    """方案4：decouple —— PRM 和 ORM 分别标准化后组合；不强制 ∑=±1。
    - PRM：基于 flags 构造基础奖励，做组内 z-score 标准化
    - ORM：使用完整的 ORM 分数，做组内 z-score 标准化
    - 组合：alpha * normalized_prm + normalized_orm（按 orm_distribution 方式分配）
    """
    B = step_ids.size(0)
    device = step_ids.device
    alpha = hyper.alpha
    orm_distribution = hyper.orm_distribution
    
    # ---- 1. 构造基础 PRM 奖励（与 ORM 无关）----
    prm_rewards_raw: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            prm_rewards_raw.append([])
            continue
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)
        # 使用固定的 good/bad 值，不依赖 ORM
        prm_rewards = [hyper.fix_base if f else -hyper.fix_base for f in flags]
        prm_rewards_raw.append(prm_rewards)
    
    # ---- 2. 对 PRM 奖励做组内 z-score 标准化 ----
    prm_rewards_std = _group_zscore_on_steps(prm_rewards_raw, group_ids, hyper)
    
    # ---- 3. 对完整的 ORM 分数做组内标准化 ----
    orm_scores = orm_full_scores.cpu().tolist()
    
    # 对 ORM 分数做组内标准化
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)
    
    orm_scores_std = [0.0] * B
    for _, idxs in g2idx.items():
        group_orms = [orm_scores[i] for i in idxs]
        if len(group_orms) == 0:
            continue
        orm_tensor = torch.tensor(group_orms, dtype=torch.float32)
        orm_mean = orm_tensor.mean()
        orm_std = orm_tensor.std(unbiased=False)
        
        if orm_std <= hyper.eps:
            # 标准差太小，只减均值
            for i in idxs:
                orm_scores_std[i] = float(orm_scores[i] - orm_mean.item())
        else:
            # 标准 z-score
            for i in idxs:
                orm_scores_std[i] = float((orm_scores[i] - orm_mean.item()) / (orm_std.item() + 1e-12))
    
    # ---- 4. 组合标准化的 PRM 和 ORM ----
    combined_rewards: List[List[float]] = []
    for i in range(B):
        if not prm_rewards_std[i]:
            combined_rewards.append([])
            continue
        
        prm_std = prm_rewards_std[i]
        orm_std = orm_scores_std[i]
        
        combined = []
        for j, prm_reward in enumerate(prm_std):
            if orm_distribution == "last_step":
                # ORM 只加在最后一步
                if j == len(prm_std) - 1:
                    combined_reward = alpha * prm_reward + orm_std
                else:
                    combined_reward = alpha * prm_reward
            elif orm_distribution == "all_steps":
                # ORM 每步都加
                combined_reward = alpha * prm_reward + orm_std
            else:
                raise ValueError(f"Unknown orm_distribution: {orm_distribution}")
            
            combined.append(float(combined_reward))
        
        combined_rewards.append(combined)
    
    return combined_rewards

# =========================
# Step → Token broadcast + suffix-sum
# =========================

def suffix_sum_on_steps(step_rewards: List[List[float]]) -> List[List[float]]:
    """计算每个轨迹step奖励的后缀和（从后往前累加）
    
    例如: [1, 2, 3] => [6, 5, 3]
    
    Args:
        step_rewards: 每条轨迹的step奖励列表
        
    Returns:
        每条轨迹的step优势值列表（后缀和形式）
    """
    adv: List[List[float]] = []
    for r in step_rewards:
        if not r:
            adv.append([]); continue
        t = torch.tensor(r, dtype=torch.float32)
        s = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])
        adv.append([float(x) for x in s])
    return adv

def broadcast_step_adv_to_tokens(
    step_adv: List[List[float]],
    step_ids: torch.Tensor,
) -> torch.Tensor:
    """将step级别的优势值广播到token级别
    
    根据step_ids将每个step的优势值赋给对应的token位置
    step_ids为-1的位置（非响应token）保持为0
    
    Args:
        step_adv: 每条轨迹的step优势值列表
        step_ids: step标识符张量，shape (B, L_resp)，-1表示非响应token
        
    Returns:
        广播到token级别的优势值张量，shape (B, L_resp)
    """
    device = step_ids.device
    B, L = step_ids.shape
    out = torch.zeros((B, L), device=device, dtype=torch.float32)
    for i in range(B):
        if not step_adv[i]:
            continue
        adv_i = torch.tensor(step_adv[i], device=device, dtype=torch.float32)
        sid_row = step_ids[i]
        valid = sid_row >= 0
        if torch.any(valid):
            sids = sid_row[valid]
            out[i, valid] = adv_i[sids]
    return out

# =========================
# Entry
# =========================

def compute_prm_grpo_advantages(
    batch,                          # DataProto 或兼容结构：batch.batch[...] 可索引
    step_flags: List[List[bool]],   # 每条轨迹的 GOOD/BAD 标志
    hyper: Optional[PRMHyper] = None,
    scheme: str = "allocation_c",   # "fix" | "allocation" | "allocation_c" | "decouple"
) -> dict:
    """
    PRM-GRPO优势函数计算统一入口
    
    算法流程:
      1. 数据准备阶段:
         - 提取必要字段：step_ids, group_ids, token_level_rewards
         - 计算ORM分数：对token-level奖励求和得到轨迹级ORM分数
      2. 方案选择阶段:
         - 根据scheme参数选择具体的奖励构造方案
         - 调用对应方案的builder函数构造step-level奖励
      3. 优势值计算阶段:
         - 对step-level奖励进行后缀和计算得到step-level优势值
         - 将step-level优势值广播到token-level
      4. 结果返回阶段:
         - 返回token-level优势值和原始ORM分数
    
    优势函数特性:
      - 支持多种奖励构造方案，适应不同场景需求
      - 统一的处理流程，便于维护和扩展
      - 完整的错误处理机制，确保数据完整性
      - 灵活的参数配置，支持自定义超参数
    
    Args:
        batch: 数据批次，包含responses, step_ids, group_ids等字段
            - responses: 响应张量
            - step_ids: step标识符，shape (B, L_resp)，-1表示非response token
            - group_ids: 组标识符，用于分组处理，shape (B,)
            - token_level_rewards: token级奖励，用于计算ORM分数
        step_flags: 每条轨迹的step级别GOOD/BAD标志
        hyper: PRM超参数配置，若为None则使用默认配置
        scheme: 奖励构造方案
            - "fix": 固定基数奖励构造
            - "allocation": 一致性权重瓜分 + 组内减均值中心化
            - "allocation_c": 一致性瓜分 → 组内归一化 → 按比例缩放投影
            - "decouple": PRM和ORM分别标准化后组合
    
    Returns:
        dict: 包含以下字段的字典
            - advantages: (B, L_resp) token-level优势值
            - orm_scalar: (B,) 逐条轨迹的 ±1
    """
    if hyper is None:
        hyper = PRMHyper()

    # ---- 1. 数据准备阶段：提取必要字段 ----
    # 获取设备信息，确保所有张量在同一设备上
    responses = batch.batch["responses"]
    device = responses.device if torch.is_tensor(responses) else torch.as_tensor(responses).device

    # 提取step_ids和group_ids，并确保数据类型正确
    step_ids = _ensure_tensor(batch.batch["step_ids"], device=device, dtype=torch.long)      # (B, L_resp) with -1 for non-response
    group_ids = _ensure_tensor(batch.batch["group_ids"], device=device, dtype=torch.long).view(-1)

    # ---- 2. 提取token-level奖励 ----
    # 尝试多种可能的字段名获取token-level奖励
    token_keys_try = ["token_level_rewards", "response_token_level_rewards", "token_rewards"]
    token_level_rewards = None
    for k in token_keys_try:
        if k in batch.batch:
            token_level_rewards = _ensure_tensor(batch.batch[k], device=device, dtype=torch.float32)
            break
    if token_level_rewards is None:
        raise KeyError("token-level rewards not found in batch (tried keys: token_level_rewards / response_token_level_rewards / token_rewards)")

    # ---- 3. ORM处理：计算ORM分数 ----
    # 对token-level奖励求和得到轨迹级ORM分数，用于各个方案的奖励构造
    orm_sum = token_level_rewards.sum(dim=1)   # (B,)
    orm_scores = torch.where(orm_sum > 0, torch.ones_like(orm_sum), -torch.ones_like(orm_sum)).to(dtype=torch.float32)

    # ---- 4. 方案选择阶段：根据scheme选择具体的奖励构造方案 ----
    scheme = (scheme or "allocation_c").lower()
    if scheme == "fix":
        # 方案1：fix —— 固定基数奖励构造 + 轨迹最后step的ORM符号调整
        step_rewards = _build_fix(orm_scores, step_flags, step_ids, group_ids, hyper)
    elif scheme == "allocation":
        # 方案2：allocation —— 一致性权重瓜分 + 组内减均值中心化
        step_rewards = _build_allocation(orm_scores, step_flags, step_ids, group_ids, hyper)
    elif scheme == "allocation_c":
        # 方案3：allocation_c —— 一致性瓜分 → 组内归一化 → 按比例缩放投影
        step_rewards = _build_allocation_c(orm_scores, step_flags, step_ids, group_ids, hyper)
    elif scheme == "decouple":
        # 方案4：decouple —— PRM和ORM分别标准化后组合
        step_rewards = _build_decouple(orm_scores, step_flags, step_ids, group_ids, hyper,)
    else:
        raise ValueError(f"Unknown PRM scheme: {scheme} (expected one of: fix | allocation | allocation_c | decouple)")

    # ---- 5. 优势值计算阶段：step后缀和 + 广播到token ----
    # 对step-level奖励进行后缀和计算得到step-level优势值
    step_adv = suffix_sum_on_steps(step_rewards)
    # 将step-level优势值广播到token-level
    advantages = broadcast_step_adv_to_tokens(step_adv, step_ids)

    # ---- 6. 结果返回阶段：构造返回字典 ----
    # 返回token-level优势值和原始ORM分数
    return {
        "advantages": advantages,        # (B, L_resp) token-level优势值
        "orm_scores": orm_scores,         # (B,) 逐条轨迹的 ±1
    }
