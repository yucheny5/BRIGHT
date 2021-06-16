import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv, SAGEConv



class GCN(torch.nn.Module):
    def __init__(self, feat_dim, hid_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class BRIGHT_A(torch.nn.Module):

    def __init__(self, dim, rwr_dim, g1_feat_dim):
        super(BRIGHT_A, self).__init__()
        self.lin = torch.nn.Linear(rwr_dim, dim)
        self.gcn1 = GCN(g1_feat_dim, dim)
        self.combine1 = torch.nn.Linear(2*dim, dim)


    def forward(self, rwr1_emd, rwr2_emd, data1, data2):
            pos_emd1 = self.lin(rwr1_emd)
            pos_emd2 = self.lin(rwr2_emd)
            gcn_emd1 = self.gcn1(data1)
            gcn_emd2 = self.gcn1(data2)
            pos_emd1 = F.normalize(pos_emd1, p=1, dim=1)
            pos_emd2 = F.normalize(pos_emd2, p=1, dim=1)
            gcn_emd1 = F.normalize(gcn_emd1, p=1, dim=1)
            gcn_emd2 = F.normalize(gcn_emd2, p=1, dim=1)
            emd1 = torch.cat([pos_emd1, gcn_emd1], 1)
            emd2 = torch.cat([pos_emd2, gcn_emd2], 1)
            emd1 = self.combine1(emd1)
            emd1 = F.normalize(emd1, p=1, dim=1)
            emd2 = self.combine1(emd2)
            emd2 = F.normalize(emd2, p=1, dim=1)
            return emd1, emd2


class BRIGHT_U(torch.nn.Module):

    def __init__(self, dim, rwr_dim):
        super(BRIGHT_U, self).__init__()
        self.lin1 = torch.nn.Linear(rwr_dim, dim)

    def forward(self, rwr1_emd, rwr2_emd):
        pos_emd1 = self.lin1(rwr1_emd)
        pos_emd2 = self.lin1(rwr2_emd)
        pos_emd1 = F.normalize(pos_emd1, p=1, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=1, dim=1)
        return pos_emd1, pos_emd2

"""
BRIGHT with just GCN
"""
class BRIGHT_gcn(torch.nn.Module):

    def __init__(self, dim, g1_feat_dim, g2_feat_dim):
        super(BRIGHT_gcn, self).__init__()
        self.gcn1 = GCN(g1_feat_dim, dim)
        self.gcn2 = GCN(g2_feat_dim, dim)

    def forward(self, data1, data2):
        gcn_emd1 = self.gcn1(data1)
        gcn_emd2 = self.gcn1(data2)
        gcn_emd1 = F.normalize(gcn_emd1, p=1, dim=1)
        gcn_emd2 = F.normalize(gcn_emd2, p=1, dim=1)
        return gcn_emd1, gcn_emd2


class ranking_loss_L1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out1, out2, anchor1, anchor2, neg1, neg2, gamma):
        anchor1_vec = out1[anchor1]
        anchor2_vec = out2[anchor2]
        neg1_vec = out2[neg1]
        neg2_vec = out1[neg2]

        A = torch.sum(torch.abs(anchor1_vec - anchor2_vec), 1)
        D = A + gamma
        B1 = -torch.sum(torch.abs(anchor1_vec-neg1_vec), 1)
        L1 = torch.sum(F.relu(B1 + D))
        B2 = -torch.sum(torch.abs(anchor2_vec - neg2_vec), 1)
        L2 = torch.sum(F.relu(B2 + D))
        return (L1 + L2)/len(anchor1)






