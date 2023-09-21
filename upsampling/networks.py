import os, sys
sys.path.append("../")
import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d
import torch.nn.functional as F
from torch import einsum
from einops import repeat

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class RA_Layer(nn.Module):
    def __init__(self, channels):
        super(RA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rel_pos_emb):
        x_q = self.q_conv(x).permute(0, 2, 1)# b, n, c

        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)

        # b, n, n
        energy = torch.bmm(x_q, x_k)

        # attention = self.softmax(energy)
        attention = self.softmax(energy + rel_pos_emb)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class feature_extraction(nn.Module):
    def __init__(self, channels=128):
        super(feature_extraction, self).__init__()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(648)
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 128, kernel_size=1, bias=False),
                                  self.bn1,
                                  nn.LeakyReLU(negative_slope=0.2))

        self.localattn1 = LocalAttention(
            dim=128,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=16  # only the 16 nearest neighbors would be attended to for each point
        )
        self.localattn2 = LocalAttention(
            dim=128,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=16  # only the 16 nearest neighbors would be attended to for each point
        )
        self.ra3 = RA_Layer(channels)
        self.ra4 = RA_Layer(channels)
        self.mlp2 = nn.Sequential(nn.Conv1d(512, 648, kernel_size=1, bias=False),
                                  self.bn2,
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, rel_pos_emb, pos):
        x = self.mlp1(x)
        x1 = self.localattn1(x, pos)
        x2 = self.localattn2(x1, pos)
        x3 = self.ra3(x2, rel_pos_emb)
        x4 = self.ra4(x3, rel_pos_emb)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.mlp2(x)
        return x

class Generator(nn.Module):
    def __init__(self, params=None):
        super(Generator, self).__init__()
        self.feature_extractor = feature_extraction(128)
        self.up_projection_unit = up_projection_unit()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
        )

    def forward(self, input):
        pos = input.permute(0, 2, 1)
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = torch.clamp(rel_pos.sum(-1), -5, 5)
        features = self.feature_extractor(input, rel_pos_emb, pos)  # b,648,n#

        H = self.up_projection_unit(features, pos)  # b,128,4*n

        coord = self.conv1(H)
        coord = self.conv2(coord)
        return coord


class Generator_recon(nn.Module):
    def __init__(self, params):
        # 坐标重建
        super(Generator_recon, self).__init__()
        self.feature_extractor = feature_extraction(128)
        self.up_ratio = params['up_ratio']
        self.num_points = params['patch_num_point']

        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=648, out_channels=128, kernel_size=1),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)
        )

    def forward(self, input):
        features = self.feature_extractor(input)  # b,648,n
        coord = self.conv0(features)
        coord = self.conv1(coord)
        coord = self.conv2(coord)
        return coord

class attention_unit(nn.Module):
    def __init__(self, in_channels=130):
        super(attention_unit, self).__init__()
        self.convF = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1),
            nn.ReLU()
        )
        self.convG = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1),
            nn.ReLU()
        )
        self.convH = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU()
        )
        self.gamma = nn.Parameter(torch.zeros([1]).clone().detach()).cuda()

    def forward(self, inputs):
        f = self.convF(inputs)
        g = self.convG(inputs)  # b,32,n
        h = self.convH(inputs)
        s = torch.matmul(g.permute(0, 2, 1), f)  # b,n,n
        beta = F.softmax(s, dim=2)  # b,n,n

        o = torch.matmul(h, beta)  # b,130,n
        x = self.gamma * o + inputs
        return x

class up_block(nn.Module):
    def __init__(self,up_ratio=4,in_channels=130):
        super(up_block,self).__init__()
        self.up_ratio = up_ratio
        self.conv1 = nn.Sequential(
            Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.grid = self.gen_grid(up_ratio).clone().detach().requires_grad_(True)
        self.attention_unit = attention_unit(in_channels=in_channels)
        self.localattn = LocalAttention(
            dim=130,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=16  # only the 16 nearest neighbors would be attended to for each point
        )
        self.edge_conv = nn.Sequential(nn.Conv2d(128 * 2, 128 * self.up_ratio, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(512),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, inputs, pos):
        net = inputs  # b,128,n
        grid = self.grid.clone()
        grid = grid.unsqueeze(0).repeat(net.shape[0], 1, net.shape[2])  # b,4,2*n
        grid = grid.view([net.shape[0], -1, 2])  # b,4*n,2

        x = get_graph_feature(net, k=20)
        net = self.edge_conv(x)
        net = net.max(dim=-1, keepdim=False)[0].permute(0, 2, 1)
        net = torch.reshape(net, (net.shape[0], net.shape[1] * self.up_ratio, -1))

        net = torch.cat([net, grid.cuda()], dim=2)  # b,n*4,130
        net = net.permute(0, 2, 1)  # b,130,n*4

        net = self.localattn(net, pos)

        net = self.conv1(net)
        net = self.conv2(net)

        return net

    def gen_grid(self,up_ratio):
        import math
        sqrted=int(math.sqrt(up_ratio))+1
        for i in range(1,sqrted+1).__reversed__():
            if (up_ratio%i)==0:
                num_x=i
                num_y=up_ratio//i
                break
        grid_x=torch.linspace(-0.2,0.2,num_x)
        grid_y=torch.linspace(-0.2,0.2,num_y)

        x,y=torch.meshgrid([grid_x,grid_y])
        grid=torch.stack([x,y],dim=-1) # 2,2,2
        grid=grid.view([-1,2])#4,2
        return grid

class down_block(nn.Module):
    def __init__(self, up_ratio=4, in_channels=128):
        super(down_block, self).__init__()
        self.conv =  nn.Sequential(
            nn.Conv1d(in_channels=in_channels*up_ratio, out_channels=in_channels, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1, padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.up_ratio = up_ratio

    def forward(self, inputs):
        net = inputs  # b,128,n*4

        net = net.reshape(net.shape[0],net.shape[1]*self.up_ratio,-1)#b,128,4,n
        net = self.conv(net)#b,256,1,n
        net = self.conv1(net)
        net = self.conv2(net)
        return net

class up_projection_unit(nn.Module):
    def __init__(self, up_ratio=4):
        super(up_projection_unit, self).__init__()
        self.conv1 = nn.Sequential(
            Conv1d(in_channels=648, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.up_block1 = up_block(up_ratio=4, in_channels=128 + 2)
        self.up_block2 = up_block(up_ratio=4, in_channels=128 + 2)
        self.down_block = down_block(up_ratio=4, in_channels=128)

    def forward(self, input, pos):
        L = self.conv1(input)  # b,128,n
        H0 = self.up_block1(L, pos)  # b,128,n*4
        L0 = self.down_block(H0)  # b,128,n

        E0 = L0 - L  # b,128,n
        H1 = self.up_block2(E0)  # b,128,4*n
        H2 = H0 + H1  # b,128,4*n
        return H2

class mlp_conv(nn.Module):
    def __init__(self, in_channels, layer_dim):
        super(mlp_conv, self).__init__()
        self.conv_list = nn.ModuleList()
        for i, num_out_channel in enumerate(layer_dim[:-1]):
            if i == 0:
                sub_module = nn.Sequential(
                    Conv1d(in_channels=in_channels, out_channels=num_out_channel, kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
            else:
                sub_module = nn.Sequential(
                    Conv1d(in_channels=layer_dim[i - 1], out_channels=num_out_channel, kernel_size=1),
                    nn.ReLU()
                )
                self.conv_list.append(sub_module)
        self.conv_list.append(
            Conv1d(in_channels=layer_dim[-2], out_channels=layer_dim[-1], kernel_size=1)
        )

    def forward(self, inputs):
        net = inputs
        for module in self.conv_list:
            net = module(net)
        return net

class mlp(nn.Module):
    def __init__(self, in_channels, layer_dim):
        super(mlp, self).__init__()
        self.mlp_list = nn.ModuleList()
        for i, num_outputs in enumerate(layer_dim[:-1]):
            if i == 0:
                sub_module = nn.Sequential(
                    nn.Linear(in_channels, num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
            else:
                sub_module = nn.Sequential(
                    nn.Linear(layer_dim[i - 1], num_outputs),
                    nn.ReLU()
                )
                self.mlp_list.append(sub_module)
        self.mlp_list.append(
            nn.Linear(layer_dim[-2], layer_dim[-1])
        )

    def forward(self, inputs):
        net = inputs
        for sub_module in self.mlp_list:
            net = sub_module(net)
        return net

class Discriminator(nn.Module):
    def __init__(self, params, in_channels):
        super(Discriminator, self).__init__()
        self.params = params
        self.start_number = 32
        self.mlp_conv1 = mlp_conv(in_channels=in_channels, layer_dim=[self.start_number, self.start_number * 2])
        self.attention_unit = attention_unit(in_channels=self.start_number * 4)
        self.mlp_conv2 = mlp_conv(in_channels=self.start_number * 4,
                                  layer_dim=[self.start_number * 4, self.start_number * 8])
        self.mlp = mlp(in_channels=self.start_number * 8, layer_dim=[self.start_number * 8, 1])

    def forward(self, inputs):
        features = self.mlp_conv1(inputs)
        features_global = torch.max(features, dim=2)[0]  ##global feature
        features = torch.cat([features, features_global.unsqueeze(2).repeat(1, 1, features.shape[2])], dim=1)
        features = get_graph_feature(features, k=20)
        features = self.attention_unit(features)

        features = self.mlp_conv2(features)
        features = torch.max(features, dim=2)[0]

        output = self.mlp(features)

        return output

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def exists(val):
    return val is not None

def max_value(t):
    return torch.finfo(t.dtype).max

def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

class LocalAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=None
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos, mask=None):
        x = x.permute(0, 2, 1)  # transpose
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i=n)  #  (x,y,z)->(x,n,y,z)

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest=False)

            v = batched_index_select(v, indices, dim=2)
            qk_rel = batched_index_select(qk_rel, indices, dim=2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim=2)
            mask = batched_index_select(mask, indices, dim=2) if exists(mask) else None

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim=-2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        agg = agg.permute(0, 2, 1)
        return agg

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = {
        "up_ratio": 4,
        "patch_num_point": 100
    }
    generator = Generator(params).cuda()
    point_cloud = torch.rand(4, 3, 100).cuda()
    output = generator(point_cloud)
    print(output.shape)
    discriminator = Discriminator(params, in_channels=3).cuda()
    dis_output = discriminator(output)
    print(dis_output.shape)
