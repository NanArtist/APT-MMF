import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class Attribution(nn.Module):
    def __init__(self, 
            nlt_in_size, ft_out_dim,
            emb_dim, type_dim, dropout_ioc, num_heads, num_meta_paths, hidden_size, out_size, dropout_mpneigh,
            cuda):
        super(Attribution, self).__init__()
        self.nlt_ft = FC(nlt_in_size, ft_out_dim)
        self.multilevel_att_net = Triple_Attention(emb_dim, type_dim, dropout_ioc, num_heads, num_meta_paths, hidden_size, out_size, dropout_mpneigh, cuda)

    def forward(self, g, inputs):
        heterG_adj, report_node, attribute_type_feat, nlt_feat, topo_relation_feat, node_vec_type = inputs
        nlt_feat_ft = self.nlt_ft(nlt_feat)
        multimodal_node_feat = torch.cat([attribute_type_feat, nlt_feat_ft, topo_relation_feat], dim=1)
        report_sparse_feat = torch.matmul(report_node, multimodal_node_feat)
        ioc_ft = heterG_adj, report_sparse_feat, multimodal_node_feat, node_vec_type 
        # heterG_adj: num_report*num_node, report_sparse_feat: num_report*embed_dim, multimodal_node_feat: num_node*embed_dim, node_vec_type: num_node*num_ioc_type
        predict = self.multilevel_att_net(g, ioc_ft)
        return predict


class FC(nn.Module):
    """a linear layer"""
    def __init__(self, input_size, output_size, dropout=0.05):
        super(FC, self).__init__()
        self.output = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        outputs = self.output(inputs)
        return self.dropout(outputs)


class Triple_Attention(nn.Module):
    def __init__(self, emb_dim, type_dim, dropout_ioc, num_heads, num_meta_paths, hidden_size, out_size, dropout_mpneigh, cuda):
        super(Triple_Attention, self).__init__()
        # IOC type-level attention layer
        self.ioc_type_att = IOC_Type_Level_Attention(type_dim=type_dim, dropout=dropout_ioc, activation=F.elu, num_heads=num_heads[0], cuda=cuda)
        # Metapath-based neighbor node-level attention layer
        self.mp_neighbor_att = Metapath_Neighbor_Level_Attention(num_meta_paths, emb_dim, hidden_size, num_heads[1], dropout_mpneigh)
        # Metapath semantic-level attention layer
        self.mp_semantic_att = Metapath_Semantic_Level_Attention(in_size=hidden_size*num_heads[1])
        # Classification layer
        self.classifier = nn.Linear(hidden_size*num_heads[1], out_size)

    def forward(self, g, inputs):
        adj, emb_dest, feat_src, emb_type = inputs
        completed_feats = self.ioc_type_att(adj, emb_dest, feat_src, emb_type)
        mp_specific_feat_stack = self.mp_neighbor_att(g, completed_feats)
        final_feats = self.mp_semantic_att(mp_specific_feat_stack)
        results = self.classifier(final_feats)
        return results


class IOC_Type_Level_Attention(nn.Module):
    def __init__(self, type_dim, dropout, activation, num_heads, cuda=False):
        super(IOC_Type_Level_Attention, self).__init__()
        self.dropout = dropout
        self.attentions_type = [AttentionLayer_Type(type_dim, dropout, activation, cuda) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions_type):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, bias, emb_dest, feat_src, emb_type):
        adj = F.dropout(bias, self.dropout, training=self.training)
        x = torch.cat([att(adj, emb_dest, emb_type, feat_src).unsqueeze(0) for att in self.attentions_type], dim=0) if self.attentions_type!=[] else None
        return torch.add(torch.mean(x, dim=0, keepdim=False), emb_dest)


class AttentionLayer_Type(nn.Module):
    def __init__(self, in_dim, dropout, activation, cuda=False):
        super(AttentionLayer_Type, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.W = nn.Parameter(nn.init.normal_(torch.zeros(in_dim).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, bias, emb_dest, emb_type, feature_src):
        emb_dest = emb_dest.to(torch.float32)
        emb_type = emb_type.to(torch.float32)
        feature_src = feature_src.to(torch.float32)

        e = self.leakyrelu(bias * torch.mv(emb_type, self.W))
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(bias > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention_dp = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention_dp, feature_src)

        return self.activation(h_prime)


class Metapath_Neighbor_Level_Attention(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(Metapath_Neighbor_Level_Attention, self).__init__()
        self.gat_layers = nn.ModuleList() # One GAT layer for each metapath based adjacency matrix
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu, allow_zero_in_degree=True))

    def forward(self, gs, h):
        mp_specific_feat_list = []
        for i, g in enumerate(gs):
            mp_specific_feat_list.append(self.gat_layers[i](g, h).flatten(1))
        mp_specific_feat_stack = torch.stack(mp_specific_feat_list, dim=1)
        return mp_specific_feat_stack # (num_reports, num_metapaths, out_size*layer_num_heads)


class Metapath_Semantic_Level_Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Metapath_Semantic_Level_Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (num_metapaths, 1)
        beta = torch.softmax(w, dim=0)                 # (num_metapaths, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (num_reports, num_metapaths, 1)
        return (beta * z).sum(1)                       # (num_reports, z.shape[-1])

