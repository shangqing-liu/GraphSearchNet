from ..layers.common import dropout
from ..layers.attention import *
from ..layers.graphs import GraphNN
from ..utils.generic_utils import to_cuda, create_mask
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class Output(object):

    def __init__(self, loss=0, loss_value=0, mrr=0):
        self.loss = loss  # scalar
        self.loss_value = loss_value  # float value, excluding coverage loss
        self.mrr = mrr
        self.top1 = 0
        self.top5 = 0
        self.top10 = 0
        self.ndcg1 = 0
        self.ndcg5 = 0
        self.ndcg10 = 0


class Graph2Search(nn.Module):

    def __init__(self, config, word_embedding, word_vocab):
        super(Graph2Search, self).__init__()
        self.name = 'Graph2Search'
        self.device = config['device']
        self.word_dropout = config['word_dropout']
        self.graph_hidden_size = config['graph_hidden_size']
        self.word_embed = word_embedding
        self.message_function = config['message_function']
        if config['fix_word_embed']:
            print('[ Fix word embeddings ]')
            for param in self.word_embed.parameters():
                param.requires_grad = False
        self.edge_embed = nn.Embedding(config['num_edge_types'], config['edge_embed_dim'], padding_idx=0)
        self.code_graph_encoder = GraphNN(config)
        self.sequence_graph_encoder = GraphNN(config)
        self.linear_max = nn.Linear(self.graph_hidden_size, self.graph_hidden_size, bias=False)
        self.heads = config.get('heads', 4)
        self.global_sequence_att = MultiHeadedAttention(self.heads, self.graph_hidden_size, config, config['word_dropout'])
        self.global_code_att = MultiHeadedAttention(self.heads, self.graph_hidden_size, config, config['word_dropout'])
        self.linear_proj = nn.Linear(self.graph_hidden_size, 2 * self.graph_hidden_size, bias=False)
        self.code_info_type = config['code_info_type']
        self.des_info_type = config['des_info_type']

    def forward(self, ex, criterion=None):
        batch_size = ex['batch_size']
        code_graphs = ex['code_graphs']
        doc_graphs = ex['doc_graphs']
        doc_words = ex['targets']
        if self.message_function == 'edge_mm':
            code_edge_vec = code_graphs['edge_features']
            doc_edge_vec = doc_graphs['edge_features']
        else:
            code_edge_vec = self.edge_embed(code_graphs['edge_features'])
            doc_edge_vec = self.edge_embed(doc_graphs['edge_features'])
        code_node_mask = create_mask(code_graphs['node_num'], code_graphs['max_node_num_batch'], self.device)
        doc_node_mask = create_mask(doc_graphs['node_num'], doc_graphs['max_node_num_batch'], self.device)

        node_embedded = self.word_embed(code_graphs['node_index'])
        node_embedded = dropout(node_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
        doc_words_embedded = self.word_embed(doc_words)
        doc_words_embedded = dropout(doc_words_embedded, self.word_dropout, shared_axes=[-2], training=self.training)

        if self.code_info_type in ['all', 'local']:
            code_node_embedding = self.code_graph_encoder(node_embedded, code_edge_vec,
                                                          (code_graphs['node2edge'], code_graphs['edge2node']))
            local_code_state = self.graph_maxpool(code_node_embedding, code_node_mask).squeeze()
        if self.des_info_type in ['all', 'local']:
            doc_node_embedding = self.sequence_graph_encoder(doc_words_embedded, doc_edge_vec,
                                                             (doc_graphs['node2edge'], doc_graphs['edge2node']))
            local_doc_state = self.graph_maxpool(doc_node_embedding, doc_node_mask).squeeze()
        code_sequence_embedded = self.word_embed(ex['sequences'])
        code_sequence_embedded_mask = create_mask(ex['sequence_lens'], ex['max_code_lens'], self.device)
        doc_sequence_embedded_mask = create_mask(doc_graphs['node_num'], doc_graphs['max_node_num_batch'], self.device)
        if self.code_info_type in ['all', 'global']:
            weighted_code = self.global_code_att(code_sequence_embedded, code_sequence_embedded, code_sequence_embedded,
                                                 code_sequence_embedded_mask.unsqueeze(1))
            global_code_state = torch.div(torch.sum(weighted_code, dim=1), ex['sequence_lens'].unsqueeze(1).float())
        if self.des_info_type in ['all', 'global']:
            weighted_doc = self.global_sequence_att(doc_words_embedded, doc_words_embedded, doc_words_embedded,
                                                    doc_sequence_embedded_mask.unsqueeze(1))
            global_doc_state = torch.div(torch.sum(weighted_doc, dim=1), ex['target_lens'].unsqueeze(1).float())
        if self.code_info_type in ['all']:
            src_state = torch.cat([local_code_state, global_code_state], dim=-1)
        elif self.code_info_type in ['global']:
            src_state = global_code_state
        else:
            src_state = local_code_state

        if self.des_info_type in ['all']:
            tgt_state = torch.cat([local_doc_state, global_doc_state], dim=-1)
        elif self.des_info_type in ['global']:
            tgt_state = global_doc_state
        else:
            tgt_state = local_doc_state

        if self.code_info_type in ['all'] and self.des_info_type not in ['all']:
            tgt_state = self.linear_proj(tgt_state)

        if self.des_info_type in ['all'] and self.code_info_type not in ['all']:
            src_state = self.linear_proj(src_state)

        r = Output()
        nll_loss, cosine_similarities = self.softmax_loss(src_state, tgt_state, criterion, batch_size)
        r.loss = torch.sum(nll_loss)
        r.loss_value = r.loss.item()
        correct_scores = torch.diag(cosine_similarities, 0)
        compared_scores = cosine_similarities >= torch.unsqueeze(correct_scores, dim=-1)
        sum = torch.sum(compared_scores, dim=1)
        ones = to_cuda(torch.ones(sum.size(), dtype=torch.long), self.device)
        mrr = torch.div(ones.double(), sum.double())
        r.mrr = torch.sum(mrr).item()
        r.top1 = torch.sum(sum == ones).item()
        r.top5 = torch.sum(sum <= 5. * ones).item()
        r.top10 = torch.sum(sum <= 10. * ones).item()
        r.ndcg1 = ndcg_score(np.arange(batch_size), cosine_similarities.detach().cpu().numpy(), k=1)
        r.ndcg5 = ndcg_score(np.arange(batch_size), cosine_similarities.detach().cpu().numpy(), k=5)
        r.ndcg10 = ndcg_score(np.arange(batch_size), cosine_similarities.detach().cpu().numpy(), k=10)
        return r

    def softmax_loss(self, src_state, tgt_state, criterion, batch_size):
        logits = torch.matmul(tgt_state, src_state.transpose(0, 1))
        label = to_cuda(torch.arange(batch_size, dtype=torch.long), self.device)
        loss = criterion(logits, label)
        return loss, logits

    def graph_maxpool(self, node_state, node_mask=None):
        node_mask = node_mask.unsqueeze(-1)
        node_state = node_state * node_mask.float()
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1))
        return graph_embedding


def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=1):
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)
    scores = []
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)
    return np.sum(scores)

