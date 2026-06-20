"""拒绝采样器（对齐 vLLM V1 `v1/sample/rejection_sampler.py`）。

投机解码的验证步：目标模型对"当前 token + k 个草案 token"共 k+1 个位置并行前向，得到每个
位置的目标输出（贪心即 argmax）。逐位比对草案与目标，接受最长正确前缀，并在首个分歧处用
目标 token 修正（全部接受时追加 1 个奖励 token）。

贪心版与逐 token 自回归**数值等价**（投机只加速、不改变贪心输出）；随机版需按目标/草案概率
比做接受-修正（本实现给出贪心版，随机版接口预留）。
"""


class RejectionSampler:

    @staticmethod
    def verify_greedy(draft: list[int], target: list[int]) -> list[int]:
        """贪心验证。

        draft:  proposer 提议的 k 个 token。
        target: 目标模型在 k+1 个位置的 argmax（前 k 个验证草案，第 k+1 个为奖励位）。
        返回接受的 token 序列（长度 1..k+1）：

          - 逐位接受 target[i]；若 target[i] != draft[i]（分歧），接受该修正 token 并停止；
          - 全部草案被接受时，追加 target[k]（奖励 token）。
        """
        assert len(target) == len(draft) + 1, "target 须比 draft 多 1（奖励位）"
        accepted = []
        for i, d in enumerate(draft):
            accepted.append(target[i])
            if target[i] != d:
                return accepted          # 分歧：修正 token 已加入，停止
        accepted.append(target[len(draft)])   # 全接受：追加奖励 token
        return accepted
