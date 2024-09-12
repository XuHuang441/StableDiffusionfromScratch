import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # attention的输入，一个input复制3份变成q，k，v三个矩阵
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, seq_len, dim)

        input_size = x.shape

        batch_size, seq_len, d_embed = input_size

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_size, seq_len, dim) -> (Batch_size, seq_len ,n_heads, d_head) -> (Batch_size, n_heads, seq_len, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_size, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-2, -1)

        if causal_mask:
            # 使用 torch.ones_like 创建一个形状和 weight 相同的新张量，所有元素都是1
            # .triu(1)：从 mask 张量中取出上三角部分（不包括主对角线），将这部分的值设为 True，其余部分设为 False。
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # 使用掩码 mask 填充 weight 张量。对于 mask 中为 True 的位置，weight 对应的元素会被填充为 -torch.inf（负无穷大）
            weight.masked_fill_(mask, -torch.inf)

        # 根据公式
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (Batch_size, n_head, seq_len, seq_len) @ (Batch_size, n_head, seq_len, d_head)
        # ->(Batch_size, n_head, seq_len, d_head)
        output = weight @ v

        # (Batch_size, seq_len, n_head, d_head)
        output = output.transpose(1, 2).contiguous()

        # (Batch_size, seq_len, dim)
        output = output.reshape(input_size)

        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    """
    Cross attention的代码基本和self attention一样
    """
    def __init__(self, n_heads: int, d_embed: int, d_cross:int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj_q = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.in_proj_k = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.in_proj_v = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x:latent (Batch_size, seq_len_q, dim_q)
        # y:context (Batch_size_kv, seq_len_kv, dim_kv) = (Batch_size, 77, 768)
        input_size = x.shape

        batch_size, seq_len, d_embed = input_size

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.in_proj_q(x)
        k = self.in_proj_k(y)
        v = self.in_proj_v(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-2, -1)

        # 根据公式
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (Batch_size, n_head, seq_len, seq_len) @ (Batch_size, n_head, seq_len, d_head)
        # ->(Batch_size, n_head, seq_len, d_head)
        output = weight @ v

        # (Batch_size, seq_len, n_head, d_head)
        output = output.transpose(1, 2).contiguous()

        # (Batch_size, seq_len, dim)
        output = output.reshape(input_size)

        output = self.out_proj(output)

        return output
