import math

import torch
from torch import nn

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift, scale, gate = self.modulation(y).chunk(3, dim=-1)
        h = self.in_norm(x) * (scale + 1) + shift
        h = self.mlp(h)
        return x + gate * h

class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * (scale + 1) + shift
        x = self.linear(x)
        return x

class DiffusionMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        num_resblks,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.time_embed = TimestepEmbedder(mid_channels)
        self.cond_embed = nn.Linear(384, mid_channels)


        self.in_layer = nn.Linear(in_channels, mid_channels)
        self.out_layer = FinalLayer(mid_channels, out_channels)
        self.res_blocks = nn.ModuleList([])

        for _ in range(num_resblks):
            self.res_blocks.append(ResBlock(mid_channels))
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.modulation[-1].weight, 0)
            nn.init.constant_(block.modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.out_layer.modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def forward(self, x, t, c):
        assert x.dim() == 2, f"Input shape should be (B*L, in_channels), got {x.shape}"
        assert c.dim() == 2, f"Condition shape should be (B*L, in_channels), got {c.shape}"
        # x: (B * L, in_channels)
        # t: (B * L,)
        # c: (B * L, in_channels)

        x = self.in_layer(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        for block in self.res_blocks:
            x = block(x, y)
        o = self.out_layer(x, y)
        return o
    
if __name__ == "__main__":
    model = DiffusionMLP(in_channels=256, out_channels=256, mid_channels=512, num_resblks=3).cuda()
    model.initialize_weights()
    x = torch.randn(6, 256).cuda()
    t = torch.randint(0, 1000, (6,), device=x.device).long()
    c = torch.randn(6, 256).cuda()
    out = model(x, t, c)
    print(out.shape)
