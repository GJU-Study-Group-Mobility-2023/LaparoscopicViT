import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels = in_chans,
            out_channels = embed_dim,
            kernel_size = patch_size,
            stride = patch_size # we put both the kernel_size and stride to be the same so the kernel will never overlap and the kernal will exactly fall into patches.
        )


    def forward(self, x):

        x = self.proj( # We run the input layer through the conv layer to get a 3 dimensional tensor
            x
        )

        x = x.flatten(2) # we flatten the 4D x tensor into a 1d vector
        x = x.transpose(1, 2)

        return x
    
class Attention(nn.module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads # number of attetion heads
        self.dim = dim # dimensions of the input and output per token features.
        self.head_dim = dim // n_heads # attention head dimensions
        self.scale = self.head_dim ** -0.5 # prevent feeding extremely large values into the softmax

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # linear mapping, where is takes in a token embedding and it generates a query key and a value.
        self.attn_drop = nn.Dropout(attn_p) # dropout to avoid overfitting problems.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        # the input and the output tensors will have the same shape.
        n_samples, n_tokens, dim = x.shape

        assert dim == self.dim, 'Dimensions are not the same'

        # we take the input tensor and turn it into q, k and v.
        qkv = self.qkv(x) # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) # (n_samples, n_patches  +1 , 3, self.n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ) # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) # transpose the keys 

        dp = (
            q @ k_t
        ) * self.scale # computer the dot product

        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(
            1, 2
        )
        weighted_avg = weighted_avg.flatten(2)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x
