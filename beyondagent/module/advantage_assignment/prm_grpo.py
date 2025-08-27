# -*- coding: utf-8 -*-
# PRM (step-level) → group z-score (step-level) → per-trajectory reprojection → suffix-sum (step-level) → broadcast to token
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch

@dataclass
class PRMHyper:
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2
    neg_unconsistent_scale: float = 0.2
    eps: float = 1e-8
    do_batch_zscore: bool = True
    traj_equal_zscore: bool = True   # ✅ 新增：True=每轨迹等权；False=拉平


# ----------------------------- Utils -----------------------------

def _ensure_tensor(x, device, dtype=None):
    if torch.is_tensor(x):
        t = x.to(device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)

def _num_steps_from_step_ids(step_ids_row: torch.Tensor) -> int:
    """step_ids: shape (L,) with -1 for non-response tokens; contiguous step ids starting at 0."""
    if step_ids_row.numel() == 0:
        return 0
    m = torch.amax(step_ids_row)
    return int(m.item() + 1) if m.item() >= 0 else 0

def _align_flags(flags: List[bool], K: int, is_success: bool) -> List[bool]:
    if len(flags) == K:
        return list(flags)
    default_flag = True if is_success else False
    if len(flags) < K:
        return list(flags) + [default_flag] * (K - len(flags))
    else:
        return list(flags[:K])

# ------------------------- Core (Plan-3) -------------------------
def compute_step_rewards_from_flags_consistent_centered(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    方案3（step 级）：
      1) 一致性权重瓜分：r_raw，逐轨迹 ∑=±1
      2) 组内（group）step-level z-score：r_std  （支持每轨迹等权）
      3) 逐轨迹“比例缩放投影”：r_proj = r_std * (±1 / sum(r_std))（退化时均分） → 逐轨迹 ∑=±1
    返回：r_proj（逐轨迹按 step 的回报）
    """
    device = step_ids.device
    B, _ = step_ids.shape
    assert orms_sign.shape[0] == B and group_ids.shape[0] == B

    # ---- 1) 一致性瓜分（∑=±1）----
    step_rewards_raw: List[List[float]] = []
    Ks: List[int] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i]); Ks.append(K)
        if K == 0:
            step_rewards_raw.append([]); continue

        is_success = bool(orms_sign[i].item() > 0)
        flags_i = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=is_success)
        if is_success:
            w_good, w_bad = hyper.consistent_scale, hyper.pos_unconsistent_scale
        else:
            w_good, w_bad = hyper.neg_unconsistent_scale, hyper.consistent_scale

        weights = torch.tensor([w_good if f else w_bad], device=device, dtype=torch.float32).repeat(K)
        # 注意：上面一行等价于按 flags_i 逐步赋值；若你更偏好逐元素，可改回列表推导
        weights = torch.tensor([w_good if f else w_bad for f in flags_i], device=device, dtype=torch.float32)
        total_w = float(weights.sum().item())
        if total_w <= hyper.eps:
            weights[:] = 1.0; total_w = float(K)
        unit = float(orms_sign[i].item()) / total_w
        step_rewards_raw.append((weights * unit).tolist())  # ∑=±1

    if not hyper.do_batch_zscore:
        return step_rewards_raw

    # ---- 2) 组内 step-level z-score ----
    step_rewards_std: List[List[float]] = [[] for _ in range(B)]
    unique_groups = torch.unique(group_ids)
    for g in unique_groups.tolist():
        idxs = (group_ids == g).nonzero(as_tuple=False).view(-1).tolist()

        if hyper.traj_equal_zscore:
            # ✅ 每轨迹等权：组均值=轨迹均值的均值；组方差=轨迹对组均值的均方差的均值
            traj_means = []
            for i in idxs:
                ri = step_rewards_raw[i]
                if ri: traj_means.append(sum(ri) / len(ri))
            if len(traj_means) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = float(sum(traj_means) / len(traj_means))
                second_moments = []
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri: continue
                    second_moments.append(sum((x - mu_g) * (x - mu_g) for x in ri) / len(ri))
                var_g = float(sum(second_moments) / len(second_moments)) if second_moments else 0.0
                sd_g = float((var_g + hyper.eps) ** 0.5)
        else:
            # 旧：拉平成所有 step 统计
            flat_vals = []
            for i in idxs:
                flat_vals.extend(step_rewards_raw[i])
            if len(flat_vals) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                t = torch.tensor(flat_vals, device=device, dtype=torch.float32)
                mu_g = float(t.mean().item())
                sd_g = float(max(t.std(unbiased=False).item(), hyper.eps))

        inv = 1.0 / (sd_g + 1e-12)
        for i in idxs:
            ri = step_rewards_raw[i]
            if not ri:
                step_rewards_std[i] = []
            else:
                step_rewards_std[i] = [float((x - mu_g) * inv) for x in ri]

    # ---- 3) 逐轨迹“比例缩放投影”（∑=±1）----
    step_rewards_proj: List[List[float]] = []
    for i in range(B):
        K = Ks[i]
        if K == 0:
            step_rewards_proj.append([]); continue
        ri = step_rewards_std[i]
        current_sum = sum(ri)
        target_sum = float(orms_sign[i].item())  # ±1
        if abs(current_sum) <= hyper.eps:
            # 退化：均分到每个 step
            step_rewards_proj.append([target_sum / K for _ in ri])
        else:
            scale = target_sum / current_sum
            step_rewards_proj.append([float(x * scale) for x in ri])

    return step_rewards_proj

def suffix_sum_on_steps(step_rewards: List[List[float]]) -> List[List[float]]:
    """对每个样本的 step 回报做后缀和，输出同形状的 step-adv。"""
    adv: List[List[float]] = []
    for r in step_rewards:
        if not r:
            adv.append([])
            continue
        t = torch.tensor(r, dtype=torch.float32)
        s = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])
        adv.append([float(x) for x in s])
    return adv

def broadcast_step_adv_to_tokens(
    step_adv: List[List[float]],
    step_ids: torch.Tensor,
) -> torch.Tensor:
    """把 step-adv 按 step_ids 广播到 token 上。step_ids 为 -1 的位置填 0。"""
    device = step_ids.device
    B, L = step_ids.shape
    out = torch.zeros((B, L), device=device, dtype=torch.float32)
    for i in range(B):
        if not step_adv[i]:
            continue
        adv_i = torch.tensor(step_adv[i], device=device, dtype=torch.float32)
        # mask for response tokens
        sid_row = step_ids[i]
        valid = sid_row >= 0
        if torch.any(valid):
            sids = sid_row[valid]
            out[i, valid] = adv_i[sids]
    return out

# ----------------------------- Entry -----------------------------

def compute_prm_grpo_advantages(
    batch,                          # DataProto 或兼容结构：batch.batch[...] 可索引
    step_flags: List[List[bool]],   # 每条轨迹的 GOOD/BAD 标志，长度与 step 数匹配（不足则按 ORM 符号补齐）
    hyper: Optional[PRMHyper] = None,
) -> Dict[str, torch.Tensor]:
    """
    方案3 + ORM 强制为 ±1 的版本：
      - ORM_sign = sign(sum(token_level_rewards))
      - 在 step 上瓜分、标准化、再投影，得到 step-adv
      - 将 step-adv 按 step_ids 广播到 token 得到 (B, L) 的 advantages
    返回：
      - advantages: (B, L) token-level advantages
      - orm_scalar: (B,) 逐条轨迹的 ±1
    """
    if hyper is None:
        hyper = PRMHyper()

    # ---- 取必要字段 ----
    device = None
    # responses 仅用于确定设备/长度
    responses = batch.batch["responses"]
    if torch.is_tensor(responses):
        device = responses.device
    else:
        responses = torch.as_tensor(responses)
        device = responses.device

    step_ids = _ensure_tensor(batch.batch["step_ids"], device=device, dtype=torch.long)  # (B, L_resp) with -1 for non-response
    group_ids = _ensure_tensor(batch.batch["group_ids"], device=device, dtype=torch.long).view(-1)

    # 取 token-level reward（可能字段名不同，做兜底）
    token_keys_try = ["token_level_rewards", "response_token_level_rewards", "token_rewards"]
    token_level_rewards = None
    for k in token_keys_try:
        if k in batch.batch:
            token_level_rewards = _ensure_tensor(batch.batch[k], device=device, dtype=torch.float32)
            break
    if token_level_rewards is None:
        raise KeyError("token-level rewards not found in batch (tried keys: token_level_rewards / response_token_level_rewards / token_rewards)")

    # ---- ORM_sign = ±1 ----
    orm_sum = token_level_rewards.sum(dim=1)   # (B,)
    orms_sign = torch.where(orm_sum > 0, torch.ones_like(orm_sum), -torch.ones_like(orm_sum)).to(dtype=torch.float32)

    # ---- Step-level pipeline ----
    step_rewards_proj = compute_step_rewards_from_flags_consistent_centered(
        orms_sign=orms_sign,
        step_flags=step_flags,
        step_ids=step_ids,
        group_ids=group_ids,
        hyper=hyper,
    )
    step_adv = suffix_sum_on_steps(step_rewards_proj)
    advantages = broadcast_step_adv_to_tokens(step_adv, step_ids)

    return {
        "advantages": advantages,        # (B, L_resp)
        "orm_scalar": orms_sign,         # (B,)
    }
