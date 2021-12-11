# -*- coding: utf-8 -*-
"""
Module to handle getting data loading classes and helper functions.
"""
import torch
import json
from random import shuffle
import numpy as np
from scipy.sparse import *
from . import padding_utils
import gzip
from tqdm import tqdm
import multiprocessing
import regex as re


def vectorize_input(batch, training=True, device=None, mode='train'):
    if not batch:
        return None
    srcs = torch.LongTensor(batch.sent1_word)
    src_lens = torch.LongTensor(batch.sent1_length)
    if batch.sent2_word is not None:
        targets = torch.LongTensor(batch.sent2_word)
        target_lens = torch.LongTensor(batch.sent2_length)

    with torch.set_grad_enabled(training):
        example = {'batch_size': batch.batch_size,
                   'code_graphs': batch.code_graph,
                   'doc_graphs': batch.doc_graph,
                   'sequences': srcs.to(device) if device else srcs,
                   'sequence_lens': src_lens.to(device) if device else src_lens,
                   'code_token_indexes': batch.code_token_indexes,
                   'code_func': batch.funcs,
                   'file_names': batch.filenames,
                   'urls': batch.urls,
                   'max_code_lens': batch.max_sent1_length
                   }
        if batch.sent2_word is not None:
            example['targets'] = targets.to(device) if device else targets
            example['target_lens'] = target_lens.to(device) if device else target_lens
            example['target_src'] = batch.sent2_src
        return example


def prepare_datasets(config):
    if config['trainset'] is not None:
        train_set, train_src_len, train_tgt_len = read_all_Datasets(config['trainset'], isLower=True)
        print('# of training examples: {}'.format(len(train_set)))
        print('Training source sentence length: {}'.format(train_src_len))
        print('Training target sentence length: {}'.format(train_tgt_len))
    else:
        train_set = None

    if config['devset'] is not None:
        dev_set, dev_src_len, dev_tgt_len = read_all_Datasets(config['devset'], isLower=True)
        print('# of dev examples: {}'.format(len(dev_set)))
        print('Dev source sentence length: {}'.format(dev_src_len))
        print('Dev target sentence length: {}'.format(dev_tgt_len))
    else:
        dev_set = None
    return {'train': train_set, 'dev': dev_set}


def read_all_Datasets(inpath, isLower=True):
    all_instances = []
    code_graph_len = []
    doc_token_len = []
    with gzip.GzipFile(inpath, 'r') as f:
        lines = list(f)
    results = parallel_process(lines, single_instance_process, args=(isLower,))
    for result in results:
        if type(result) is tuple:
            (sent1, sent2) = result
            code_graph_len.append(sent1.get_token_length())
            doc_token_len.append(sent2.get_token_length())
            all_instances.append((sent1, sent2))
    code_graph_len_stats = {'min': np.min(code_graph_len), 'max': np.max(code_graph_len), 'mean': np.mean(code_graph_len)}
    doc_token_len_stats = {'min': np.min(doc_token_len), 'max': np.max(doc_token_len), 'mean': np.mean(doc_token_len)}
    return all_instances, code_graph_len_stats, doc_token_len_stats


def single_instance_process(line, isLower, mode='train'):
    instance = json.loads(line)
    code_graph = instance['code_graph']
    if len(code_graph['nodes']) > 200:
        return False
    sent1 = Graph(instance, codeGraph=True, isLower=isLower)
    if mode == 'train':
        sent2 = Graph(instance, docGraph=True, isLower=isLower)
        if sent1.get_node_length() > 200 or sent2.get_token_length() > 200:
           return False
        else:
            return (sent1, sent2)
    else:
        if sent1.get_node_length() > 200:
            return False
        else:
            return (sent1, None)


def parallel_process(array, single_instance_process, args=(), n_cores=None):
    if n_cores is 1:
        return [single_instance_process(x, *args) for x in tqdm(array)]
    with tqdm(total=len(array)) as pbar:
        def update(*args):
            pbar.update()
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=n_cores) as pool:
            jobs = [
                pool.apply_async(single_instance_process, (x, *args), callback=update) for x in array
            ]
            results = [job.get() for job in jobs]
        return results


def read_db(inpath, isLower=True):
    all_instances = []
    code_graph_len = []
    with gzip.GzipFile(inpath, 'r') as f:
        lines = list(f)
    results = parallel_process(lines, single_instance_process, args=(isLower, 'build'))
    for result in results:
        if type(result) is tuple:
            (sent1, sent2) = result
            code_graph_len.append(sent1.get_token_length())
            all_instances.append((sent1, sent2))
    code_graph_len_stats = {'min': np.min(code_graph_len), 'max': np.max(code_graph_len),
                            'mean': np.mean(code_graph_len)}
    return all_instances, code_graph_len_stats


class DataStream(object):
    def __init__(self, all_instances, word_vocab, edge_vocab, config=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.config = config
        if batch_size == -1: batch_size = config['batch_size']
        if isSort:
            all_instances = sorted(all_instances, key=lambda instance: (instance[0].get_node_length()))
        else:
            random.shuffle(all_instances)
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute srcs into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for (batch_start, batch_end) in tqdm(batch_spans):
            cur_instances = all_instances[batch_start: batch_end]
            cur_batch = Batch(cur_instances, config, word_vocab, edge_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]


class Batch(object):
    def __init__(self, instances, config, word_vocab, edge_vocab):
        self.instances = instances
        self.batch_size = len(instances)
        # Create word representation and length
        self.sent2_src = []
        self.sent2_word = []    # [batch_size, sent2_len]
        self.sent2_length = []  # [batch_size]
        self.sent1_word = []  # [batch_size, sent2_len]
        self.sent1_length = []
        self.max_sent1_length = 0
        self.code_token_indexes = []
        self.funcs = []
        self.filenames = []
        self.urls = []
        batch_code_graph = []
        batch_doc_graph = []
        for (sent1, sent2) in instances:
            batch_code_graph.append(sent1.graph)
            if sent2 is not None:
                batch_doc_graph.append(sent2.graph)
        if config['model_name'] in ['Graph2Search']:
            batch_code_graphs = cons_batch_graph(batch_code_graph, word_vocab)
            self.code_graph = vectorize_batch_graph(batch_code_graphs, config, edge_vocab)
            if len(batch_doc_graph) > 0:
                batch_doc_graphs = cons_batch_graph(batch_doc_graph, word_vocab)
                self.doc_graph = vectorize_batch_graph(batch_doc_graphs, config, edge_vocab)
            else:
                self.doc_graph = None
        else:
            self.code_graph = None
            self.doc_graph = None
        for i, (sent1, sent2) in enumerate(instances):
            src_idx = []
            for word in sent1.graph['backbone_sequence']:
                idx = word_vocab.getIndex(word)
                src_idx.append(idx)
            self.sent1_length.append(sent1.get_token_length())
            self.sent1_word.append(src_idx)
            if sent1.get_token_length() > self.max_sent1_length:
                self.max_sent1_length = sent1.get_token_length()
            if sent2 is not None:
                sent2_idx = []
                for word in sent2.graph['backbone_sequence']:
                    idx = word_vocab.getIndex(word)
                    sent2_idx.append(idx)
                self.sent2_word.append(sent2_idx)
                self.sent2_src.append(' '.join(sent2.graph['backbone_sequence']))
                self.sent2_length.append(sent2.get_token_length())
            self.code_token_indexes.append(sent1.seq_token_in_node)
            self.funcs.append(sent1.function)
            self.filenames.append(sent1.filename)
            self.urls.append(sent1.url)
        self.sent1_word = padding_utils.pad_2d_vals_no_size(self.sent1_word)
        self.sent1_length = np.array(self.sent1_length, dtype=np.int32)
        if instances[0][1] is not None:
            self.sent2_word = padding_utils.pad_2d_vals_no_size(self.sent2_word)
            self.sent2_length = np.array(self.sent2_length, dtype=np.int32)


class Graph(object):
    def __init__(self, instance, codeGraph=False, docGraph=False, isLower=False):
        if codeGraph:
            code_graph = instance['code_graph']
            backbone_sequence = code_graph['tokens']
            self.seq_token_in_node = code_graph['seq_token_in_node']
            if isLower:
                backbone_sequence = [token.lower() for token in backbone_sequence]
            filter_code_nodes, edges = self.build_code_graph(code_graph, isLower)
            self.graph = {'nodes': filter_code_nodes, 'edges': edges,
                          'backbone_sequence': backbone_sequence}
            if 'original_string' in instance.keys():
                self.function = instance['original_string']
            else:
                self.function = instance['function']
            if 'func_name' in instance.keys():
                self.filename = instance['func_name']
            else:
                self.filename = instance['identifier']
            self.func_name_list = self.subtokenizer(self.filename)
            self.url = instance['url']
        if docGraph:
            doc_graph = instance['doc_graph']
            backbone_sequence = doc_graph['backbone_sequence']
            if isLower:
                backbone_sequence = [token.lower() for token in backbone_sequence]
            doc_nodes = self.build_doc_graph(backbone_sequence)
            self.graph = {'nodes': doc_nodes, 'edges': doc_graph['edges'],
                          'backbone_sequence': backbone_sequence}

    def build_code_graph(self, code_graph, isLower):
        filter_code_nodes = []
        edges = []
        node_id_list = []
        for index, node in enumerate(code_graph['nodes']):
            if isLower:
                node_content = node['contents'].lower()
            else:
                node_content = node['contents']
            node_id = node['id_sorted']
            filter_code_nodes.append({'id': node_id, 'content': node_content, 'type': node['type']})
            node_id_list.append(node_id)
        for edge in code_graph['edges']:
            if edge['sourceId'] in node_id_list and edge['destinationId'] in node_id_list:
                edges.append([edge['type'], edge['sourceId'], edge['destinationId']])
        return filter_code_nodes, edges

    def build_doc_graph(self, backbone_sequence):
        doc_nodes = []
        for id, doc_token in enumerate(backbone_sequence):
            doc_nodes.append({'id': id, 'content': doc_token})
        return doc_nodes

    def get_node_length(self):
        return len(self.graph['nodes'])

    def get_token_length(self):
        return len(self.graph['backbone_sequence'])

    def subtokenizer(self, identifier):
        splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        identifiers = re.split('[._\-]', identifier)
        subtoken_list = []
        for identifier in identifiers:
            matches = splitter_regex.finditer(identifier)
            for subtoken in [m.group(0) for m in matches]:
                subtoken_list.append(subtoken)
        return subtoken_list


def cons_batch_graph(graphs, word_vocab):
    num_nodes = max([len(g['nodes']) for g in graphs])
    num_edges = max([len(g['edges']) for g in graphs])
    batch_edges = []
    batch_node2edge = []
    batch_edge2node = []
    batch_node_num = []
    batch_node_index = []
    for example_id, g in enumerate(graphs):
        edges = {}
        graph_node_index = cons_node_features(g['nodes'], word_vocab)
        node2edge = lil_matrix(np.zeros((num_edges, num_nodes)), dtype=np.float32)
        edge2node = lil_matrix(np.zeros((num_nodes, num_edges)), dtype=np.float32)
        edge_index = 0
        for edge, src_node, dest_node in g['edges']:
            if src_node == dest_node:  # Ignore self-loops for now
                continue
            edges[edge_index] = edge
            node2edge[edge_index, dest_node] = 1
            edge2node[src_node, edge_index] = 1
            edge_index += 1
        batch_edges.append(edges)
        batch_node2edge.append(node2edge)
        batch_edge2node.append(edge2node)
        batch_node_num.append(len(g['nodes']))
        batch_node_index.append(graph_node_index)
    batch_graphs = {'max_num_edges': num_edges,
                    'edge_features': batch_edges,
                    'node2edge': batch_node2edge,
                    'edge2node': batch_edge2node,
                    'node_num': batch_node_num,
                    'max_num_nodes': num_nodes,
                    'node_word_index': batch_node_index
                    }
    return batch_graphs


def cons_node_features(nodes, word_vocab):
    graph_node_index = []
    for node in nodes:
        idx = word_vocab.getIndex(node['content'])
        graph_node_index.append(idx)
    return graph_node_index


def vectorize_batch_graph(graph, config, edge_vocab):
    edge_features = []
    for edges in graph['edge_features']:
        edges_v = []
        for idx in range(len(edges)):
            edges_v.append(edge_vocab.getIndex(edges[idx]))
        for _ in range(graph['max_num_edges'] - len(edges_v)):
            edges_v.append(edge_vocab.PAD)
        edge_features.append(edges_v)
    edge_features = torch.LongTensor(np.array(edge_features))
    node_indexes = torch.LongTensor(padding_utils.pad_2d_vals_no_size(graph['node_word_index']))
    node_num = torch.LongTensor(np.array(graph['node_num']))
    gv = {'edge_features': edge_features.to(config['device']) if config['device'] else edge_features,
          'node2edge': graph['node2edge'],
          'edge2node': graph['edge2node'],
          'node_num': node_num,
          'max_node_num_batch': graph['max_num_nodes'],
          'node_index': node_indexes.to(config['device']) if config['device'] else node_indexes
          }
    return gv