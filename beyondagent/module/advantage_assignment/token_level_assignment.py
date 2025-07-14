import torch
import verl.utils.torch_functional as verl_F

def _add_entropy_mask(entropy, rho, response_mask, adv=None, mode="pos-high-neg-high"):
        # print('response_mask',response_mask)
        new_mask = response_mask.bool()
        flat_entropy = entropy[new_mask]  # 只统计有效 token

        # print('new_mask',new_mask.shape)
        # print('flat_entropy',flat_entropy)
        # print('flat_entropy shape',flat_entropy.shape)
        # exit(0)
        # print('maskori', response_mask.bool())
        def _quantile_threshold(flat_values: torch.Tensor, q: float) -> float:
            return torch.quantile(flat_values, q=q, interpolation="higher").item()

        if adv is None:
            raise ValueError("`adv` must be provided when mode='pos-high-neg-low'")

        adv_row = adv[:, 0]
        pos_rows = adv_row > 0
        neg_rows = adv_row < 0

        pos_mask_rows = pos_rows.unsqueeze(-1).expand_as(new_mask)
        pos_mask = pos_mask_rows & new_mask
        entropy_pos = verl_F.masked_mean(entropy, pos_mask)

        neg_mask_rows = neg_rows.unsqueeze(-1).expand_as(new_mask)
        neg_mask = neg_mask_rows & new_mask
        entropy_neg = verl_F.masked_mean(entropy, neg_mask)

        entropy_pos_neg = (entropy_pos.detach().item(), entropy_neg.detach().item())
        print("entropy_pos_neg", entropy_pos_neg)
        # exit(0)

        if mode == "pos-high-neg-high":
            flat_entropy = entropy[new_mask]  # 只统计有效 token
            tau = _quantile_threshold(flat_entropy, q=rho)
            new_mask &= entropy >= tau  # 保留高熵
            tau_rho = (tau, tau)

        elif mode == "pos-high-neg-low":

            if isinstance(rho, (list, tuple)):
                rho_pos, rho_neg = float(rho[0]), float(rho[1])
            else:
                rho_pos = rho_neg = float(rho)  # 若只给一个值，默认两边都用同一个

            tau_pos = _quantile_threshold(flat_entropy, q=rho_pos)
            new_mask &= (~pos_mask_rows) | (entropy >= tau_pos)
            tau_neg = _quantile_threshold(
                flat_entropy, q=1.0 - rho_neg
            )  # 负样本要屏蔽高熵 ⇒ 取 (1-ρ_neg) 分位阈值
            new_mask &= (~neg_mask_rows) | (entropy <= tau_neg)

            tau_rho = (tau_pos, tau_neg)
            print("tau_rho", tau_rho)

        elif mode == None:
            new_mask = response_mask
            tau_rho = None
        else:
            raise
        return new_mask.to(response_mask.dtype), tau_rho, entropy_pos_neg