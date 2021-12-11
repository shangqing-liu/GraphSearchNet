from parsers.sourcecode.java.build_java_graph import build_desc_graph, normalize_des_graph
from graph_search_nn.src.core.utils.data_utils import Graph, cons_batch_graph, vectorize_batch_graph
from graph_search_nn.src.core.utils.padding_utils import pad_2d_vals_no_size
from graph_search_nn.src.core.model import cal_query_features
from graph_search_nn.src.core.utils import constants
import numpy as np
import torch
from elasticsearch import Elasticsearch
import pandas as pd
import os


class search_engine:
    def __init__(self, model_handle, config):
        self.model = model_handle.model
        self.config = config
        self.model_handle = model_handle
        self.save_file = config['answer_file']
        if not os.path.exists(os.path.dirname(self.save_file)):
            os.makedirs(os.path.dirname(self.save_file))

    def search(self, str_searchs, search_size=10):
        instances = []
        answers = []
        for str_search in str_searchs:
            instance = {}
            str_search = ' '.join(str_search)
            instance['doc_graph'] = build_desc_graph(str_search, file=None)
            instance['doc_graph'] = normalize_des_graph(instance['doc_graph'])
            instances.append(Graph(instance, docGraph=True, isLower='True'))
        ex = self.build_batch_data(instances)
        query_embedded = cal_query_features(self.model.network, ex)
        client = Elasticsearch()
        for index in range(query_embedded.shape[0]):
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, doc['code_state']) + 1.0",
                        "params": {"query_vector": query_embedded[index].tolist()}
                    }
                }
            }

            response = client.search(index=self.config['index_name'], body={
                    "size": search_size,
                    "query": script_query,
                    "_source": {"includes": ['code_func', 'identifier', 'url']}
                }
            )
            for hit in response["hits"]["hits"]:
                answers.append({'query': ' '.join(str_searchs[index]), 'function': hit["_source"]['code_func'],
                                'identifier': hit["_source"]['identifier'], 'url': hit["_source"]['url'],
                                'score': hit['_score']})
        df = pd.DataFrame(answers, columns=['query', 'function', 'identifier', 'url', 'score'])
        df.to_csv(self.save_file, index=False)
        print('Answer query finished')

    def build_batch_data(self, instances):
        doc_word_lengths = []
        doc_words = []
        doc_graphs = []
        for i, doc_graph in enumerate(instances):
            doc_idx = []
            for word in doc_graph.graph['backbone_sequence']:
                idx = self.model.vocab_model.word_vocab.getIndex(word)
                doc_idx.append(idx)
            doc_word_lengths.append(len(doc_idx))
            doc_words.append(doc_idx)
            doc_graphs.append(doc_graph.graph)
        batch_doc_graphs = cons_batch_graph(doc_graphs, self.model.vocab_model.word_vocab)
        doc_words = pad_2d_vals_no_size(doc_words)
        doc_word_lengths = np.array(doc_word_lengths, dtype=np.int32)
        doc_words = torch.LongTensor(doc_words)
        doc_word_lengths = torch.LongTensor(doc_word_lengths)
        batch_doc_graphs = vectorize_batch_graph(batch_doc_graphs, self.config, self.model.vocab_model.edge_vocab)
        with torch.set_grad_enabled(False):
            example = {'batch_size': len(instances),
                       'doc_graphs': batch_doc_graphs,
                       'targets': doc_words.to(self.model_handle.device) if self.model_handle.device else doc_words,
                       'target_lens': doc_word_lengths.to(self.model_handle.device) if self.model_handle.device
                       else doc_word_lengths
                       }
        return example

    def test(self, query_embedded):
        client = Elasticsearch()
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['code_vector']) + 1.0",
                    "params": {"query_vector": query_embedded}
                }
            }
        }

        response = client.search(index=self.config['index_name'], body={
            "size": 1,
            "query": script_query,
            "_source": {"includes": ["code_func"]}
        }
        )

        for hit in response["hits"]["hits"]:
            print("score: {}".format(hit["_score"]))
            print(hit["_source"]['code_func'])
            print('--------------------------')