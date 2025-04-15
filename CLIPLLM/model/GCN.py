import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

device = "cuda" if torch.cuda.is_available() else "cpu"

class GNNSimilarity(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNSimilarity, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels).to(device)
        self.conv2 = GCNConv(hidden_channels, out_channels).to(device)

    def forward(self, x, edge_index):
        # x: 节点特征, shape [num_nodes, num_features]
        # edge_index: 图的边的索引，形状为 [2, num_edges]
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def get_similarity_with_gnn(que_clip_feat, supp_clip_feat, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate((mask == 1).float(), que_clip_feat.shape[-2:])
    cosine_eps = 1e-7
    supp_clip_feat = supp_clip_feat * mask

    # 假设每个像素是一个节点，构造图的邻接关系
    # 这里简单地假设所有节点都相邻，实际应用中可以根据空间关系或距离来构建图
    bsize, ch_sz, sp_sz, _ = que_clip_feat.size()
    num_nodes = sp_sz * sp_sz  # 假设每个像素点是一个节点
    # que_feat_flat = que_clip_feat.view(bsize, ch_sz, -1).permute(0, 2, 1)  # Flatten 特征
    que_feat_flat = que_clip_feat.view(bsize, ch_sz, -1)  # Flatten 特征
    supp_feat_flat = supp_clip_feat.view(bsize, ch_sz, -1)  # Flatten 支持特征

    # 构造边：假设这里使用所有的像素对之间的连接
    edge_index = (torch.combinations(torch.arange(num_nodes), r=2).T).to(device)  # 完全图（所有节点都有边连接）

    # 合并查询和支持特征，作为 GNN 的输入
    x = torch.cat([que_feat_flat, supp_feat_flat], dim=1).view(bsize * num_nodes, -1).to(device)

    # 初始化 GNN 模型
    gnn = GNNSimilarity(in_channels=x.size(1), hidden_channels=256, out_channels=1)

    # 计算 GNN 相似度
    gnn_out = gnn(x, edge_index)

    # 计算查询和支持特征之间的相似度
    similarity = gnn_out.view(bsize, num_nodes)

    # 可能需要做一些归一化或后处理
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity
