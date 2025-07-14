import torch
import verl.utils.torch_functional as verl_F
from openai import OpenAI
import os
from loguru import logger
from openai import OpenAI, APIStatusError, APITimeoutError, RateLimitError

__all__ = [
    "evaluate_step_flags",      # 调用 LLM 得到各 step 的 GOOD / BAD 判断
    "apply_step_mask",          # 按规则缩放 token-level advantage
]

# ————————————————————————————————————————————————————————————————
# 1. 调用评估 LLM，判定每个 step 是否 GOOD
# ————————————————————————————————————————————————————————————————
def _build_prompt(query: str,
                  rollout: str,
                  step: str,
                  overall_adv: float) -> list[dict]:
    """
    构造对话消息，要求 LLM 输出 'GOOD' 或 'BAD'（大小写皆可）。
    """
    polarity = "positive" if overall_adv > 0 else "negative"
    sys   = "You are an expert reward-model evaluator.  \nReply with **exactly one word**, either **GOOD** or **BAD** – no explanations."
    user  = """
────────────────────────────────
USER QUERY
{query}

ASSISTANT FULL ANSWER
{rollout}

CURRENT ASSISTANT STEP
{step}
────────────────────────────────

The total advantage (quality score) of the full answer is **{overall_adv:+.4f}**  → this is {polarity} (positive if > 0, negative if < 0).

**Task**  
Does the *current assistant step* improve (GOOD) or harm (BAD) the final answer given the user query and the overall advantage?
    """
    print("user",user)
    return [{"role": "system", "content": sys},
            {"role": "user",   "content": user}]

def _safe_query(client, model, messages):
    """
    调 OpenAI SDK；若报错则返回 None
    """
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            timeout=30,                # 可视情况调整
        )
        return rsp.choices[0].message.content.strip()
    except (APIStatusError, APITimeoutError, RateLimitError) as e:
        # 打印一次即可，防止刷屏
        print(f"[semantic_eval] LLM call failed: {e.__class__.__name__} — {e}", flush=True)
        return None
    except Exception as e:
        # 兜底，不让任何异常往 Ray 外层冒
        print(f"[semantic_eval] Unexpected error: {type(e).__name__}: {e}", flush=True)
        return None
    
def evaluate_step_flags(tokenizer,
                        batch,
                        good_words: tuple[str, ...] = ("GOOD",),
                        bad_words:  tuple[str, ...] = ("BAD",),
                        model_name: str = "qwen-max") -> list[list[bool]]:
    """
    对于 batch 中的每条样本，返回 bool 列表，长度 = step 数，
    True 表示 step 为 GOOD，False 表示 BAD。
    依赖 env_manager ➜ to_dataproto 时写入：
        • batch.non_tensor_batch['steps'] : list[list[str]]
    """
    client = OpenAI(
        api_key = os.getenv("DASHSCOPE_API_KEY"),        # 也可换成你自己的阿里云 key
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    flags_per_sample: list[list[bool]] = []
    try:
        for idx in range(len(batch.batch["prompts"])):
            query   = tokenizer.decode(batch.batch["prompts"][idx],   skip_special_tokens=True)
            rollout = tokenizer.decode(batch.batch["responses"][idx], skip_special_tokens=True)
            steps   = batch.non_tensor_batch["steps"][idx]            # 已由 env_manager 提供
            overall_adv = batch.batch["advantages"][idx].sum().item()

            step_flags = []
            for step_text in steps:
                msgs = _build_prompt(query, rollout, step_text, overall_adv)
                # rsp  = client.chat.completions.create(
                #     model     = model_name,
                #     messages  = msgs,
                #     temperature = 0.0,
                # )
                # word = rsp.choices[0].message.content.strip().upper()
                # step_flags.append(word in good_words if word in good_words + bad_words else False)  # 异常默认为 BAD
                answer = _safe_query(client, model_name, msgs)
                if answer is None:
                    # API 调用失败 → 默认 BAD
                    step_flags.append(False)
                else:
                    answer = answer.upper()
                    step_flags.append(answer.startswith("G"))
                
            flags_per_sample.append(step_flags)
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        logger.exception("LLM evaluation failed")
        # 如果 LLM 调用失败，返回全 False 的 flags
        flags_per_sample = [[False] * len(batch.non_tensor_batch["steps"][0]) for _ in range(len(batch.batch["prompts"]))]
    return flags_per_sample


# ————————————————————————————————————————————————————————————————
# 2. 根据 GOOD/BAD 结果对 token-level advantage 做缩放
# ————————————————————————————————————————————————————————————————
def apply_step_mask(batch,
                    step_flags: list[list[bool]],
                    good_scale: float = 1.0,
                    bad_scale:  float = 0.2):
    """
    * 需要 env_manager ➜ to_dataproto 写入
        • batch.batch['step_ids'] : LongTensor (bs, resp_len)
          - 每个 response token 的 step 索引，从 0 开始；填充位置 = -1
    * 根据整体 Advantage 的正负号切换缩放系数（题主给出的 4 条规则）
    """
    adv      = batch.batch["advantages"]          # (bs, resp_len)
    step_ids = batch.batch["step_ids"].to(adv.device)

    bs, resp_len = adv.shape
    scale = torch.ones_like(adv)

    for b in range(bs):
        overall_pos = adv[b].sum().item() > 0      # True → overall 正
        for s, is_good in enumerate(step_flags[b]):
            tok_mask = step_ids[b] == s            # (resp_len,)
            if not tok_mask.any():
                continue
            if overall_pos:
                factor = good_scale if is_good else bad_scale
            else:
                factor = -bad_scale if is_good else good_scale
            scale[b].masked_fill_(tok_mask, factor)
    print("****************scale",scale.shape, scale)
    print('batch.batch["advantages"]',batch.batch["advantages"].shape)
    print('batch.batch["advantages"][0] ori:',batch.batch["advantages"][0])
    print("****************scale",scale.shape, scale[0])
    # 填充 token（step_id == -1）不变
    batch.batch["advantages"] = adv * scale
    print('&&&&&&&&&&&&&&&&&batch.batch["advantages"][0] final:',batch.batch["advantages"][0])
    
    # 为后续诊断留一个可视化字段
    batch.batch["semantic_scale"] = scale
