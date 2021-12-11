import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .models.Graph2Search import Graph2Search
from .utils.vocab_utils import VocabModel
from .utils import constants as Constants
from .utils.generic_utils import create_mask, to_cuda
from .layers.common import dropout


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        self.config = config
        if config['model_name'] in ['Graph2Search']:
            self.net_module = Graph2Search
        else:
            raise RuntimeError('Unknown model_name: {}'.format(config['model_name']))
        print('[ Running {} model ]'.format(config['model_name']))

        self.vocab_model = VocabModel.build(self.config['saved_vocab_file'], train_set, config)
        self.config['num_edge_types'] = len(self.vocab_model.edge_vocab)
        if self.config['pretrained']:
            state_dict_opt = self.init_saved_network(self.config['pretrained'])
        else:
            assert train_set is not None
            # Building network.
            self._init_new_network()

        num_params = 0
        for name, p in self.network.named_parameters():
            print('{}: {}'.format(name, str(p.size())))
            num_params += p.numel()

        print('#Parameters = {}\n'.format(num_params))

        self.criterion = nn.CrossEntropyLoss()
        self._init_optimizer()

    def init_saved_network(self, saved_dir):
        fname = os.path.join(saved_dir, Constants._SAVED_WEIGHTS_FILE)
        print('[ Loading saved model %s ]' % fname)
        saved_params = torch.load(fname, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        self.saved_epoch = saved_params.get('epoch', 0)

        word_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                              pretrained_vecs=self.vocab_model.word_vocab.embeddings)
        self.network = self.net_module(self.config, word_embedding, self.vocab_model.word_vocab)

        # Merge the arguments
        if state_dict:
            merged_state_dict = self.network.state_dict()
            for k, v in state_dict['network'].items():
                if k in merged_state_dict:
                    merged_state_dict[k] = v
            self.network.load_state_dict(merged_state_dict)

        return state_dict.get('optimizer', None) if state_dict else None

    def _init_new_network(self):
        word_embedding = self._init_embedding(len(self.vocab_model.word_vocab), self.config['word_embed_dim'],
                                           pretrained_vecs=self.vocab_model.word_vocab.embeddings)
        self.network = self.net_module(self.config, word_embedding, self.vocab_model.word_vocab)

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adagrad':
            self.optimizer = optim.Adagrad(parameters, lr=self.config['learning_rate'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    def _init_embedding(self, vocab_size, embed_size, pretrained_vecs=None):
        """Initializes the embeddings
        """
        return nn.Embedding(vocab_size, embed_size, padding_idx=0, _weight=torch.from_numpy(pretrained_vecs).float()
                            if pretrained_vecs is not None else None)

    def save(self, dirname, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            'config': self.config,
            'dir': dirname,
            'epoch': epoch
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')

    def predict(self, batch, step, update=True, mode='train'):
        self.network.train(update)

        if mode == 'train':
            loss, loss_value, metrics = train_batch(batch, self.network, self.criterion)
            # Accumulate gradients
            loss = loss / self.config['grad_accumulated_steps']  # Normalize our loss (if averaged)
            # Run backward
            loss.backward()

            if (step + 1) % self.config['grad_accumulated_steps'] == 0:  # Wait for several backward steps
                if self.config['grad_clipping']:
                    # Clip gradients
                    parameters = [p for p in self.network.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(parameters, self.config['grad_clipping'])
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
            output = {
                'loss': loss_value,
                'metrics': metrics
            }
        elif mode == 'dev':
            loss_value, metrics = dev_batch(batch, self.network, criterion=self.criterion)
            output = {
                'loss': loss_value,
                'metrics': metrics
            }
        elif mode == 'test':
            loss_value, metrics = test_batch(batch, self.network, criterion=self.criterion)
            output = {
                'loss': loss_value,
                'metrics': metrics
            }
        else:
            code_state, code_funcs, file_names, urls = build_batch(batch, self.network)
            loss_value = None
            output = {
                'code_state': code_state,
                'code_funcs': code_funcs,
                'file_names': file_names,
                'urls': urls
            }
        return output


def train_batch(batch, network, criterion):
    network.train(True)
    metrics = {}
    with torch.set_grad_enabled(True):
        out = network(batch, criterion)
        loss = out.loss
        loss_value = out.loss_value
        metrics['MRR'] = out.mrr
        metrics['TOP1'] = out.top1
        metrics['TOP5'] = out.top5
        metrics['TOP10'] = out.top10
        metrics['NDCG1'] = out.ndcg1
        metrics['NDCG5'] = out.ndcg5
        metrics['NDCG10'] = out.ndcg10
    return loss, loss_value, metrics


# Development phase
def dev_batch(batch, network, criterion=None):
    metrics = {}
    network.train(False)
    out = eval_decode_batch(batch, network, criterion)
    metrics['MRR'] = out.mrr
    metrics['TOP1'] = out.top1
    metrics['TOP5'] = out.top5
    metrics['TOP10'] = out.top10
    metrics['NDCG1'] = out.ndcg1
    metrics['NDCG5'] = out.ndcg5
    metrics['NDCG10'] = out.ndcg10
    return out.loss_value, metrics


# Test phase
def test_batch(batch, network, criterion=None):
    metrics = {}
    network.train(False)
    out = eval_decode_batch(batch, network, criterion)
    metrics['MRR'] = out.mrr
    metrics['TOP1'] = out.top1
    metrics['TOP5'] = out.top5
    metrics['TOP10'] = out.top10
    metrics['NDCG1'] = out.ndcg1
    metrics['NDCG5'] = out.ndcg5
    metrics['NDCG10'] = out.ndcg10
    return out.loss_value, metrics


# Building phase
def build_batch(batch, network):
    network.train(False)
    with torch.no_grad():
        code_state, code_funcs, file_names, urls = cal_code_features(network, batch)
    return code_state, code_funcs, file_names, urls


def cal_code_features(network, ex):
    batch_size = ex['batch_size']
    code_graphs = ex['code_graphs']
    if network.message_function == 'edge_mm':
        code_edge_vec = code_graphs['edge_features']
    else:
        code_edge_vec = network.edge_embed(code_graphs['edge_features'])
    code_node_mask = create_mask(code_graphs['node_num'], code_graphs['max_node_num_batch'], network.device)
    node_embedded = network.word_embed(code_graphs['node_index'])
    node_embedded = dropout(node_embedded, network.word_dropout, shared_axes=[-2], training=network.training)
    code_node_embedding = network.code_graph_encoder(node_embedded, code_edge_vec,
                                                     (code_graphs['node2edge'], code_graphs['edge2node']))
    code_sequence_embedded = network.word_embed(ex['sequences'])
    code_sequence_embedded_mask = create_mask(ex['sequence_lens'], ex['max_code_lens'], network.device)
    weighted_code = network.global_code_att(code_sequence_embedded, code_sequence_embedded, code_sequence_embedded,
                                            code_sequence_embedded_mask.unsqueeze(1))
    global_code_state = torch.div(torch.sum(weighted_code, dim=1), ex['sequence_lens'].unsqueeze(1).float())
    local_code_state = network.graph_maxpool(code_node_embedding, code_node_mask).squeeze()

    if network.code_info_type in ['all']:
        src_state = torch.cat([local_code_state, global_code_state], dim=-1)
    elif network.code_info_type in ['global']:
        src_state = global_code_state
    else:
        src_state = local_code_state
    return src_state.detach().cpu().numpy(), ex['code_func'], ex['file_names'], ex['urls']


def cal_query_features(network, ex):
    doc_graphs = ex['doc_graphs']
    doc_words = ex['targets']
    if network.message_function == 'edge_mm':
        doc_edge_vec = doc_graphs['edge_features']
    else:
        doc_edge_vec = network.edge_embed(doc_graphs['edge_features'])
    doc_node_mask = create_mask(doc_graphs['node_num'], doc_graphs['max_node_num_batch'], network.device)
    doc_words_embedded = network.word_embed(doc_words)
    doc_sequence_embedded_mask = create_mask(doc_graphs['node_num'], doc_graphs['max_node_num_batch'], network.device)
    doc_node_embedding = network.sequence_graph_encoder(doc_words_embedded, doc_edge_vec,
                                                        (doc_graphs['node2edge'], doc_graphs['edge2node']))
    weighted_doc = network.global_sequence_att(doc_words_embedded, doc_words_embedded, doc_words_embedded,
                                               doc_sequence_embedded_mask.unsqueeze(1))
    global_doc_state = torch.div(torch.sum(weighted_doc, dim=1), ex['target_lens'].unsqueeze(1).float())
    local_doc_state = network.graph_maxpool(doc_node_embedding, doc_node_mask).squeeze()
    if network.des_info_type in ['all']:
        tgt_state = torch.cat([local_doc_state, global_doc_state], dim=-1)
    elif network.des_info_type in ['global']:
        tgt_state = global_doc_state
    else:
        tgt_state = local_doc_state
    return tgt_state.detach().cpu().numpy()


def eval_decode_batch(batch, network, criterion=None):
    """Test the `network` on the `batch`, return the decoded textual tokens and the Output."""
    with torch.no_grad():
        out = network(batch, criterion)
    return out
