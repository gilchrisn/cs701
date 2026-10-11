"""LoRA and MoE-LoRA adapters for the attention layers of a frozen backbone, and Mona adapters on its residual stream."""
import torch
import torch.nn.functional as F
from torch import nn


class LoRA(nn.Module):
    """y = W x + (alpha / r) * B A x, where W is the wrapped layer (frozen by add_adapters)."""

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0: training starts from exactly the pretrained layer
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


class MoELoRA(nn.Module):
    """y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x, with gates g(x) = softmax(router[task](x)) per token.

    The experts' A_e are stacked in `down` and their B_e side by side in `up`, so gating each
    expert's r hidden units and applying `up` sums the experts.
    """

    def __init__(self, base: nn.Linear, experts=4, rank=4, alpha=8, tasks=2):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, experts * rank, bias=False)  # A_1 ... A_E
        self.up = nn.Linear(experts * rank, base.out_features, bias=False)  # B_1 ... B_E
        nn.init.zeros_(self.up.weight)
        self.routers = nn.ModuleList(nn.Linear(base.in_features, experts, bias=False) for _ in range(tasks))
        self.rank = rank
        self.scale = alpha / rank
        self.task = 0  # which router to use; see set_task

    def forward(self, x):
        gates = self.routers[self.task](x).softmax(dim=-1)
        gates = gates.repeat_interleave(self.rank, dim=-1)  # expert e's gate on each of its r hidden units
        return self.base(x) + self.scale * self.up(self.down(x) * gates)


class Mona(nn.Module):
    """Mona adapter (arXiv 2408.08345, Eqs. 3-5) on the residual stream: x = x0 + U gelu(f_pw(f_dw(D(s1 LN(x0) + s2 x0)))),
    f_dw(z) = z + mean of depth-wise 3x3/5x5/7x7 convs of z, f_pw(z) = z + 1x1 conv of z, on the patch-token grid.
    Prefix (CLS) tokens pass through unchanged (our choice; the paper's Swin has none). U = 0: starts as the identity."""

    def __init__(self, dim=768, hidden=64, prefix=1, convs=True):
        super().__init__()
        self.prefix = prefix
        self.convs = convs  # False: the same serial adapter without f_dw and f_pw (--adapt adapter, H22 control)
        self.norm = nn.LayerNorm(dim)
        self.s1 = nn.Parameter(torch.ones(()))
        self.s2 = nn.Parameter(torch.ones(()))
        self.mona_down = nn.Linear(dim, hidden)
        if convs:
            self.dw = nn.ModuleList(nn.Conv2d(hidden, hidden, k, padding=k // 2, groups=hidden, bias=False) for k in (3, 5, 7))
            self.pw = nn.Conv2d(hidden, hidden, 1, bias=False)  # no conv biases: matches the paper's parameter count, Eq. 6
        self.mona_up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.mona_up.weight)
        nn.init.zeros_(self.mona_up.bias)

    def forward(self, x):
        cls, x0 = x[:, :self.prefix], x[:, self.prefix:]
        b, n, _ = x0.shape
        h = w = int(n ** 0.5)
        assert h * w == n, f'Mona needs a square token grid, got {n} tokens'
        z = self.mona_down(self.s1 * self.norm(x0) + self.s2 * x0)
        if self.convs:
            z = z.transpose(1, 2).reshape(b, -1, h, w).contiguous()  # a strided view here fails in cuDNN for batch > 1
            z = z + sum(conv(z) for conv in self.dw) / 3
            z = z + self.pw(z)
            z = z.reshape(b, -1, n).transpose(1, 2)
        return torch.cat([cls, x0 + self.mona_up(F.gelu(z))], dim=1)


class MonaBlock(nn.Module):
    """A frozen timm ViT Block with one Mona after the attention residual and one after the MLP residual."""

    def __init__(self, blk, dim=768, prefix=1, convs=True):
        super().__init__()
        self.blk = blk
        self.mona1 = Mona(dim, prefix=prefix, convs=convs)
        self.mona2 = Mona(dim, prefix=prefix, convs=convs)

    def forward(self, x, attn_mask=None):
        b = self.blk
        x = x + b.drop_path1(b.ls1(b.attn(b.norm1(x), attn_mask=attn_mask)))
        x = self.mona1(x)
        x = x + b.drop_path2(b.ls2(b.mlp(b.norm2(x))))
        return self.mona2(x)


def add_adapters(backbone, kind):
    """Freeze the backbone, then wrap attn.qkv and attn.proj of every block (kind: 'lora' or 'moelora'), or (kind 'mona')
    replace every timm ViT block by a MonaBlock (kind 'adapter': without its convolutions)."""
    backbone.requires_grad_(False)
    if kind in ('mona', 'adapter'):  # 'adapter': the same MonaBlock without convolutions
        vit = backbone.vit  # timm ViT only
        for i, blk in enumerate(vit.blocks):
            vit.blocks[i] = MonaBlock(blk, vit.embed_dim, vit.num_prefix_tokens, convs=kind == 'mona')
        return
    adapter = {'lora': LoRA, 'moelora': MoELoRA}[kind]
    for blk in backbone.blocks:
        blk.attn.qkv = adapter(blk.attn.qkv)
        blk.attn.proj = adapter(blk.attn.proj)


def set_task(module, task):
    """Route every MoELoRA inside `module` through the router of `task` (0 = classification, 1 = detection)."""
    for layer in module.modules():
        if isinstance(layer, MoELoRA):
            layer.task = task
