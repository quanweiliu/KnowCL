"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DynamicGraphConvolution(nn.Module):
    # 这是正确的，需要记住
    # [B, vector_length, class_num] 分别为 batch_size, 向量长度， 向量维度
    # 一维卷积（in_channels, out_channels）在最后一维上进行计算， 所以按照上面的逻辑，需要先用permute 函数反转 1，2维
    # 得到 [B, class_num, vector_length], 然后，得到
    # batch_size * out_channels * vector_length
    # 如果不反转，就直接计算维度上的信息了, 得到
    # batch_size * out_channels * class_num
    def __init__(self, in_features, out_features, num_nodes):    # 37, 64, 64
        super(DynamicGraphConvolution, self).__init__()
        self.num_nodes = num_nodes                               # num_nodes = class_num
        

        # 使用这个得到邻接矩阵 A, 好好分析一下为什么邻接矩阵可以这样表示？
        # 发现：这个邻接矩阵不是 n*n 的，而是 n*c 的

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),      # 64, 64
            nn.LeakyReLU(0.2))
        # 前向传播的全连接层，也就是 W
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),               # 26, 64
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        """
            - 4, 16, 1024 * 1024, 1024
            - 所谓静态图就是全连接网络
        """
        # print("num_nodes", self.num_nodes)                      # 64
        # print("x6 ", x.shape)                                   # 64, 26, 64
        # 首先和邻接矩阵相乘
        x = self.static_adj(x.transpose(1, 2))                  # 64 * 64 @ 64 * 64 * 26
        # print("static adj", x.shape)                            # 64, 64, 26
        # 然后和权重相乘
        x = self.static_weight(x.transpose(1, 2))               # 26 * 64 @ 64 * 64 * 26
        # print("static weight", x.shape)                         # 64, 64, 64
        return x

    # 这里值得好好研究一下，惊喜！！
    def forward_construct_dynamic_graph(self, x):
        # print("x7", x.shape)                                      # 64, 26, 64
        ### Model global representations ###
        x_glb = self.gap(x)
        # print("gap", x_glb.shape)                                 # ([B, 26, 1])
        x_glb = self.conv_global(x_glb)                           #    
        # print("conv_global", x_glb.shape)                         # ([B, 26, 1])
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        # print("expand ", x_glb.shape)                             # ([B, 26, class_num])
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        # print("x8", x.shape)                                      # ([B, 52, class_num])
        dynamic_adj = self.conv_create_co_mat(x)                  # 
        # print("dynamic_adj1", dynamic_adj.shape)                  # ([B, class_num, class_num])
        dynamic_adj = torch.sigmoid(dynamic_adj)                  
        # print("dynamic_adj2", dynamic_adj.shape)                  # ([B, class_num, class_num])
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        # print("x", x.shape, "dynamic_adj", dynamic_adj.shape)   # ([B, 26, class_num]) ([4, class_num, class_num])
        x = torch.matmul(x, dynamic_adj)
        # print("x9", x.shape)                                      # ([B, 26, class_num])
        x = self.relu(x)
        x = self.dynamic_weight(x)
        # print("x10", x.shape)                                     # ([B, 64, class_num])

        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        - 动态图是加入注意力机制的全连接网络
        """
        # print("DynamicGraphConvolution_input", x.shape)            # 64, 26, 64
        out_static = self.forward_static_gcn(x) 
        # print('static output', out_static.shape)                   # 64, 64, 64
        # x = x + out_static  # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        # print("dynamic_adj" , dynamic_adj.shape)                   # 
        x = self.forward_dynamic_gcn(x, dynamic_adj)          
        # print('dynamic output', x.shape)                           # 
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# 默认为 ViT-B
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)               # 224, 224
        patch_size = (patch_size, patch_size)         # 卷积核大小
        self.img_size = img_size                      # 224
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16=14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 14*14

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)   # 3，768，16，16
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # print("x.shape", x.shape)

        # ViT 模型的传入图片的大小是固定的
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW], 从第二个位置展平
        # transpose: [B, C, HW] -> [B, HW, C]
        # 调换位置，变成  [batch_size, token，dimension]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5         # 指 根号下维度分支一
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 可以用三个全连接层分别得到 qkv
        self.attn_drop = nn.Dropout(attn_drop_ratio)      # 
        self.proj = nn.Linear(dim, dim)                   # 对 concat 拼接的head 传入全连接层
        self.proj_drop = nn.Dropout(proj_drop_ratio)      # 

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 为了后面进行计算，不太好理解，可以好好看看
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 通过切片的方式，得到 q，k，v
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        # 多维数据的矩阵乘法，只对最后两位进行操作
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对每行进行softmax 处理
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    hidden_features 一般为 in_feature  四倍
    in_feature 等于 out_feature
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    '''
    Encoder Block, 组合几个定义好的组件，并添加其它组件
    '''
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,      # mlp 升维的倍数
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # GCN 
        # self.gcn = DynamicGraphConvolution(64, 26, 64)          # in, out, num_nodes
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # print(x.shape)      #11 [64, 5, 64],             #21 [64, 64, 64]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.gcn(self.norm2(x)))
        return x


class VisionTransformerGCN(nn.Module):
    def __init__(self, img_size=29, patch_size=4, in_c=5, num_classes=16,
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        整合以上定义的所有组件
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer： Encoder Block 的个数
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformerGCN, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1

        # GCN
        self.gcn = DynamicGraphConvolution(37, 64, embed_dim)          # in, out, num_nodes

        # partial ?
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # 制作patch
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # class_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 构建等差序列，构建 drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        
        # 堆叠 Encoder Block
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        # 没有执行这一步 distilled, 所以 没有 self.head_dist
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # print("forward features shape", x.shape)
        x = torch.squeeze(x)
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            # 直接走这里
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        # print("x_forward_features", x.shape)                     # 64, 26, 64
        x = self.gcn(x)
        # x = x + v
        # print("x_forward_features", x.shape)                     # 64, 26, 64

        if self.dist_token is None:
            # 提取 class_token，分类依据
            return self.pre_logits(x[:, 0])     # 64, 64
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        # print("input shape", x.shape)
        x = self.forward_features(x)
        # print("x_forward", x.shape)                     # 64, 64
        # if self.head_dist is not None:
        #     x, x_dist = self.head(x[0]), self.head_dist(x[1])
        #     if self.training and not torch.jit.is_scripting():
        #         # during inference, return the average of both classifier predictions
        #         return x, x_dist
        #     else:
        #         return (x + x_dist) / 2
        # else:
        #     # 直接走到这
        #     x = self.head(x) 
        #     # print("x", x.shape)               # 64, 16
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

if __name__ == "__main__":
    input = torch.rand(128,10,27,27)
    model =  VisionTransformerGCN(img_size=27, patch_size=4, in_c=10, 
                                    num_classes=17, embed_dim=128, depth=4, 
                                    num_heads=4, mlp_ratio=4.0)
    output = model(input)
    print(output.shape)

