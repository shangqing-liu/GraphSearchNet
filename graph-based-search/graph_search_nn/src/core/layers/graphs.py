import torch
import torch.nn as nn
from ..utils.generic_utils import to_cuda
from .common import GRUStep, GatedFusion
import torch.nn.functional as F
from .GAT import GAT, GraphAttentionLayer
from .GCN import GCN


class GraphNN(nn.Module):
    def __init__(self, config):
        super(GraphNN, self).__init__()
        print('[ Using {}-hop GraphNN ]'.format(config['graph_hops']))
        self.device = config['device']
        hidden_size = config['graph_hidden_size']
        self.hidden_size = hidden_size
        self.graph_direction = config['graph_direction']
        self.graph_type = config['graph_type']
        self.graph_hops = config['graph_hops']
        self.word_dropout = config['word_dropout']
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.graph_type == 'ggnn_bi':
            self.static_graph_mp = GraphMessagePassing(config)
            self.static_gru_step = GRUStep(hidden_size, hidden_size)
            if self.graph_direction == 'all':
                self.static_gated_fusion = GatedFusion(hidden_size)
            self.graph_update = self.static_graph_update
        elif self.graph_type == 'gcn':
            self.gcn = GCN(config)
            self.graph_update = self.graph_gcn_update
        elif self.graph_type == 'gat':
            self.gat = GraphAttentionLayer(hidden_size, hidden_size, 0.6, 0.2)
            self.graph_update = self.graph_attention_update
        else:
            raise RuntimeError('Unknown graph_type: {}'.format(self.graph_type))

        print('[ Using graph type: {} ]'.format(self.graph_type))
        print('[ Using graph direction: {} ]'.format(self.graph_direction))

    def forward(self, node_feature, edge_vec, adj):
        node_state = self.graph_update(node_feature, edge_vec, adj)
        return node_state

    def static_graph_update(self, node_feature, edge_vec, adj):
        '''Static graph update'''
        node2edge, edge2node = adj
        # Shape: (batch_size, num_edges, num_nodes)
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)
        for _ in range(self.graph_hops):
            bw_agg_state = self.static_graph_mp.mp_func(node_feature, edge_vec, node2edge, edge2node)  # (num_nodes, dim)
            fw_agg_state = self.static_graph_mp.mp_func(node_feature, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            if self.graph_direction == 'all':
                agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state)
                node_feature = self.static_gru_step(node_feature, agg_state)
            elif self.graph_direction == 'forward':
                node_feature = self.static_gru_step(node_feature, fw_agg_state)
            else:
                node_feature = self.static_gru_step(node_feature, bw_agg_state)
        return node_feature

    def graph_attention_update(self, node_state, edge_vec, adj, node_mask=None):
        node2edge, edge2node = adj
        node2edge = to_cuda(torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0), self.device)
        edge2node = to_cuda(torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0), self.device)
        adj = torch.bmm(edge2node, node2edge)
        node_weight_list = []
        for index in range(node_state.size(0)):
            current_node_state = node_state[index, :, :]
            current = adj[index, :, :]
            node_weight_index = self.gat(current_node_state, current)
            node_weight_list.append(node_weight_index)
        node_weight = torch.stack(node_weight_list)
        node_state = torch.bmm(node_weight, node_state)
        return node_state

    def graph_gcn_update(self, node_state, edge_vec, adj, node_mask=None):
        node_state = self.gcn.graph_update(node_state, edge_vec, adj, node_mask)
        return node_state


class GraphMessagePassing(nn.Module):
    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config['graph_hidden_size']
        if config['message_function'] == 'edge_mm':
            self.edge_weight_tensor = torch.Tensor(config['num_edge_types'], hidden_size * hidden_size)
            self.edge_weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.edge_weight_tensor))
            self.mp_func = self.msg_pass_edge_mm
        elif config['message_function'] == 'edge_network':
            self.edge_network = torch.Tensor(config['edge_embed_dim'], hidden_size, hidden_size)
            self.edge_network = nn.Parameter(nn.init.xavier_uniform_(self.edge_network))
            self.mp_func = self.msg_pass_edge_network
        elif config['message_function'] == 'edge_pair':
            self.linear_edge = nn.Linear(config['edge_embed_dim'], hidden_size, bias=False)
            self.mp_func = self.msg_pass
        elif config['message_function'] == 'no_edge':
            self.mp_func = self.msg_pass
        else:
            raise RuntimeError('Unknown message_function: {}'.format(config['message_function']))

    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)                      # batch_size x num_edges x hidden_size
        if edge_vec is not None and self.config['message_function'] == 'edge_pair':
            node2edge_emb = node2edge_emb + self.linear_edge(edge_vec)
        agg_state = torch.bmm(edge2node, node2edge_emb)                         # consider self-loop if preprocess not igore
        return agg_state

    def msg_pass_edge_mm(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = F.embedding(edge_vec[:, i], self.edge_weight_tensor).view(-1, node_state.size(-1), node_state.size(-1)) # batch_size x hidden_size x hidden_size
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1) # batch_size x num_edges x hidden_size
        agg_state = torch.bmm(edge2node, new_node2edge_emb)
        return agg_state

    def msg_pass_edge_network(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = torch.mm(edge_vec[:, i], self.edge_network.view(self.edge_network.size(0), -1)).view((-1,) + self.edge_network.shape[-2:])
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1) # batch_size x num_edges x hidden_size
        agg_state = torch.bmm(edge2node, new_node2edge_emb)
        return agg_state