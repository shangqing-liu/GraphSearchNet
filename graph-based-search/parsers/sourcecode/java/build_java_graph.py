import gzip
import codecs
import os
import json
from subprocess import Popen, PIPE
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
from spacy.tokens import Doc
import multiprocess as mp
import pickle
import re
import spacy
import multiprocessing
from pathlib import Path
import pickle
environ = os.environ.copy()
environ['JAVA_HOME'] = '/usr/local/jdk-11.0.2'
environ['PATH'] += ':/usr/local/jdk-11.0.2/bin'

dir_path = os.path.dirname(os.path.realpath(__file__))
home_path = str(Path.home())
EXTRACTOR_JAR = os.path.join(dir_path, 'features-javac/extractor/target/'
                                       'features-javac-extractor-1.0.0-SNAPSHOT-jar-with-dependencies.jar')
DOT_JAR = os.path.join(dir_path, 'features-javac/dot/target/features-javac-dot-1.0.0-SNAPSHOT-jar-with-dependencies.jar')
RAW_FILE = os.path.join(home_path, 'data/code_search_data/CodeSearchNet/resources/data/java/final/jsonl')
JAVA_BASE = os.path.join(home_path, 'data/code_search_data/CodeSearchNet/resources/data/java_dedupe_definitions_v2.pkl')
JAVA_BASE_DIR = os.path.join(home_path, 'data/code_search_data/CodeSearchNet/resources/data/java_base')
BASE_GRAPH_FILE = os.path.join(home_path, 'data/code_search_data/CodeSearchNet/resources/data/java_base_gz')


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
NLP = spacy.load('en')
NLP.tokenizer = WhitespaceTokenizer(NLP.vocab)


def load_jsonl_gz(file_path):
    reader = codecs.getreader('utf-8')
    data = []
    with gzip.open(file_path) as f:
        json_list = list(reader(f))
    data.extend([json.loads(jline) for jline in json_list])
    return data


def check_existed(sample, java_func_dir):
    couple = sample['url'].split('/')[-1].split('#')
    class_name = couple[0].split('.java')[0]
    start = couple[1].split('-')[0].replace('L', '')
    end = couple[1].split('-')[1].replace('L', '')
    if 'repo' in sample.keys():
        project = sample['repo'].replace('/', '-')
    else:
        project = sample['nwo'].replace('/', '-')
    file_name = os.path.join(java_func_dir, project + '_' + class_name + '_' + str(start) + '_' + str(end))
    if os.path.exists(file_name):
        return True
    else:
        return False


def write_sample_to_java_file(sample, java_func_dir):
    couple = sample['url'].split('/')[-1].split('#')
    class_name = couple[0].split('.java')[0]
    start = couple[1].split('-')[0].replace('L', '')
    end = couple[1].split('-')[1].replace('L', '')
    if 'repo' in sample.keys():
        project = sample['repo'].replace('/', '-')
    else:
        project = sample['nwo'].replace('/', '-')
    file_name = os.path.join(java_func_dir, project + '_' + class_name + '_' + str(start) + '_' + str(end))
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    if 'code' in sample.keys():
        function_text = sample['code']
    else:
        function_text = sample['function']
    function_text = wrap_function_dummy_class(function_text, class_name)
    wrapped_file_path = write_wrapped_function(function_text, class_name, file_name)
    write_json(sample, os.path.join(file_name, class_name + '.json'))
    return wrapped_file_path, file_name


def write_wrapped_function(function_text, class_name, file_name):
    wrapped_file_path = os.path.join(file_name, class_name + '.java')
    with open(wrapped_file_path, 'w', encoding='utf-8') as writer:
        writer.write(function_text)
    return wrapped_file_path


def wrap_function_dummy_class(function_text, class_name):
    function_text = 'public class ' + class_name + ' { \n' + function_text + '\n }'
    return function_text


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def build_java_graphs():
    types = os.listdir(RAW_FILE)
    for type in types:
        samples = []
        type_dir = os.path.join(RAW_FILE, type)
        for file in os.listdir(type_dir):
            if file.endswith('.gz') and file.startswith('java'):
                samples.extend(load_jsonl_gz(os.path.join(type_dir, file)))
        java_func_dir = os.path.join(type_dir, 'java_funcs')
        if not os.path.exists(java_func_dir):
            os.makedirs(java_func_dir)
        results = parallel_process(samples, build_single_graph, args=(java_func_dir, ))
        succeed = 0
        for result in results:
            if result:
                succeed += 1
        print('%s built %d graphs totally' % (type, succeed))


def build_java_base_graphs():
    definitions = pickle.load(open(JAVA_BASE, 'rb'))
    if not os.path.exists(JAVA_BASE_DIR):
        os.makedirs(JAVA_BASE_DIR)
    results = parallel_process(definitions, build_single_graph, args=(JAVA_BASE_DIR, ))
    succeed = 0
    for result in results:
        if result:
            succeed += 1
    print('%s built %d graphs totally' % (type, succeed))


def build_single_graph(sample, java_func_dir):
    try:
        if 'function_tokens' in sample.keys():
            if len(sample['function_tokens']) > 200:
                return False
        if check_existed(sample, java_func_dir):
            return False
        wrapped_file_path, file_name = write_sample_to_java_file(sample, java_func_dir)
        generate_proto_file(wrapped_file_path)
        if os.path.exists(wrapped_file_path + '.proto'):
            generate_dot_json_file(wrapped_file_path)
            return True
        return False
    except:
        return False


def parallel_process(array, function, args=(), n_cores=None):
    if n_cores is 1:
        return [function(x, *args) for x in tqdm(array)]
    with tqdm(total=len(array)) as pbar:
        def update(*args):
            pbar.update()
        if n_cores is None:
            n_cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=n_cores) as pool:
            jobs = [
                pool.apply_async(function, (x, *args), callback=update) for x in array
            ]
            results = [job.get() for job in jobs]
        return results


def generate_proto_file(wrapped_file_path):
    command = '/usr/local/jdk-11.0.2/bin/javac -cp %s -Xplugin:FeaturePlugin %s' % (EXTRACTOR_JAR, wrapped_file_path)
    p1 = Popen(command, cwd='/', shell=True, env=environ, stdout=PIPE, stderr=PIPE)
    p1.communicate()


def generate_dot_json_file(wrapped_file_path):
    command = '/usr/local/jdk-11.0.2/bin/java -jar %s -i %s -o %s -j %s' % (DOT_JAR,
                                                                            os.path.join(wrapped_file_path + '.proto'),
                                                                            os.path.join(wrapped_file_path + '.dot'),
                                                                            os.path.join(wrapped_file_path + '.json'))
    p1 = Popen(command, cwd='/', shell=True, env=environ, stdout=PIPE, stderr=PIPE)
    p1.communicate()


def normalize_graph(graph, func_nodes):
    method_nodes = []
    method_edges = []
    sorted_nodes = sorted(graph['node'], key=lambda i: int(i['id']))
    mapping = {}
    index = 0
    nodes_to_subtokenize = {}
    token_sequential_list = []
    seq_token_in_node = []
    try:
        for sorted_node in sorted_nodes:
            if sorted_node['id'] in func_nodes:
                if 'TOKEN' in sorted_node['type']:
                    subtokens = subtokenizer(sorted_node['contents'])
                    if len(subtokens) > 1:
                        for subtoken in subtokens:
                            dummy_node = sorted_node.copy()
                            dummy_node['contents'] = subtoken
                            dummy_node['id_sorted'] = index
                            dummy_node['subtoken'] = True
                            dummy_node['ori_token'] = sorted_node['contents']
                            token_sequential_list.append(subtoken)
                            seq_token_in_node.append(index)
                            method_nodes.append(dummy_node)
                            if sorted_node['id'] not in mapping.keys():
                                mapping[sorted_node['id']] = index
                            if sorted_node['id'] not in nodes_to_subtokenize.keys():
                                nodes_to_subtokenize[sorted_node['id']] = [index]
                            else:
                                nodes_to_subtokenize[sorted_node['id']].append(index)
                            index += 1
                    else:
                        sorted_node['id_sorted'] = index
                        sorted_node['subtoken'] = False
                        method_nodes.append(sorted_node)
                        mapping[sorted_node['id']] = index
                        seq_token_in_node.append(index)
                        token_sequential_list.append(sorted_node['contents'])
                        index += 1
                else:
                    sorted_node['id_sorted'] = index
                    method_nodes.append(sorted_node)
                    mapping[sorted_node['id']] = index
                    index += 1
        for edge in graph['edge']:
            if edge['sourceId'] in func_nodes and edge['destinationId'] in func_nodes and edge['type'] in \
                    ['AST_CHILD', 'NEXT_TOKEN', 'COMPUTED_FROM', 'LAST_USE', 'LAST_WRITE']:
                edge['sourceId'] = mapping[edge['sourceId']]
                edge['destinationId'] = mapping[edge['destinationId']]
                method_edges.append(edge)
        for key in nodes_to_subtokenize.keys():
            for index in range(1, len(nodes_to_subtokenize[key])):
                edge = {}
                edge['sourceId'] = nodes_to_subtokenize[key][index-1]
                edge['destinationId'] = nodes_to_subtokenize[key][index]
                edge['type'] = 'SUB_TOKEN'
                method_edges.append(edge)
        if not len(seq_token_in_node) == len(token_sequential_list):
            code_graph = {}
        else:
            code_graph = {'nodes': method_nodes, 'edges': method_edges, 'seq_token_in_node': seq_token_in_node,
                          'tokens': token_sequential_list}
    except:
        code_graph = {}
    return code_graph


def subtokenizer(identifier):
    if identifier == 'MONKEYS_AT':
        return [identifier]
    splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    identifiers = re.split('[._\-]', identifier)
    subtoken_list = []
    for identifier in identifiers:
        matches = splitter_regex.finditer(identifier)
        for subtoken in [m.group(0) for m in matches]:
            subtoken_list.append(subtoken)
    return subtoken_list


def single_post_process(key, chunk):
    file_graph_name = os.path.join(os.path.join(RAW_FILE, key), 'graph_' + key + '_gnn.jsonl.gz')
    count = 0
    with gzip.GzipFile(file_graph_name, 'wb') as gnn_file:
        for file in tqdm(chunk):
            class_name = file.split('_')[1]
            if os.path.exists(os.path.join(RAW_FILE, key, 'java_funcs', file, class_name + '.java.json')):
                try:
                    sub_graph_nodes = clip_dot_graph(os.path.join(RAW_FILE, key,
                                                                  'java_funcs', file, class_name + '.java.dot'))
                    with open(os.path.join(RAW_FILE, key, 'java_funcs', file, class_name + '.java.json')) as reader:
                        graph = json.load(reader)
                    with open(os.path.join(RAW_FILE, key, 'java_funcs', file, class_name + '.json')) as reader:
                        raw_sample = json.load(reader)
                    raw_sample['code_graph'] = normalize_graph(graph, sub_graph_nodes)
                    if raw_sample['docstring_tokens']:
                        doc_summary = ' '.join(re.sub(r'[^A-Za-z0-9 ]+', ' ',
                                                      ' '.join(raw_sample['docstring_tokens'])).split())
                        if doc_summary:
                            doc_graph = build_desc_graph(doc_summary)
                            if doc_graph:
                                raw_sample['doc_graph'] = normalize_des_graph(doc_graph)
                            else:
                                continue
                    else:
                        raw_sample['doc_graph'] = {}
                    if raw_sample['code_graph'] and raw_sample['doc_graph']:
                        save_sample_to_jsonl_gz(raw_sample, gnn_file)
                        count += 1
                    del raw_sample
                except:
                    continue
        print('there are %d samples have code graph and doc graph in %s' % (count, key))
        gnn_file.close()


def post_process():
    files = os.listdir(RAW_FILE)
    chunk_files = dict()
    process_list = []
    for file in files:
        raw_funcs_dir = os.path.join(RAW_FILE, file, 'java_funcs')
        chunk_files[file] = os.listdir(raw_funcs_dir)
    for key in chunk_files.keys():
        process = mp.Process(target=single_post_process,
                             args=(key, chunk_files[key]))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()
    # single_post_process('test', chunk_files['test'])


def build_desc_graph(desc, file=None):
    try:
        if str(desc).endswith('.'):
            desc = desc[0: len(desc)-1]
        desc = ' '.join(desc.split())
        doc = NLP(desc)
        g_features = []
        dep_tree = defaultdict(list)
        boundary_nodes = []
        for sent in doc.sents:
            boundary_nodes.append(sent[-1].i)
            for each in sent:
                g_features.append(each.text)
                if each.i != each.head.i:                   # Not a root
                    dep_tree[each.head.i].append({'node': each.i, 'edge': each.dep_})

        for i in range(len(boundary_nodes) - 1):
            # Add connection between neighboring dependency trees
            dep_tree[boundary_nodes[i]].append({'node': boundary_nodes[i] + 1, 'edge': 'neigh'})
            dep_tree[boundary_nodes[i] + 1].append({'node': boundary_nodes[i], 'edge': 'neigh'})
        edges = []
        for key, values in dep_tree.items():
            for value in values:
                edges.append((value['edge'], key, value['node']))
        if edges:
            des_graph = {'backbone_sequence': g_features, 'edges': edges}
        else:
            des_graph = {}
    except:
        des_graph = {}
    return des_graph


def normalize_des_graph(des_graph):
    new_tokens = []
    new_edges = []
    nodes_to_subtokenize = {}
    mapping = {}
    count = 0
    for index, token in enumerate(des_graph['backbone_sequence']):
        subtokens = subtokenizer(token)
        if len(subtokens) > 1:
            for subtoken in subtokens:
                if index not in nodes_to_subtokenize.keys():
                    nodes_to_subtokenize[index] = [count]
                    mapping[index] = count
                else:
                    nodes_to_subtokenize[index].append(count)
                new_tokens.append(subtoken)
                count += 1
        else:
            mapping[index] = count
            new_tokens.extend(subtokens)
            count += 1
    for edge in des_graph['edges']:
        new_edges.append((edge[0].upper(), mapping[edge[1]], mapping[edge[2]]))
    for key in nodes_to_subtokenize.keys():
        for index in range(1, len(nodes_to_subtokenize[key])):
            new_edges.append(('SUB_TOKEN', nodes_to_subtokenize[key][index - 1], nodes_to_subtokenize[key][index]))
    for index in range(len(new_tokens) - 1):
        new_edges.append(('NEXT_TOKEN', index, index + 1))
    return {'backbone_sequence': new_tokens, 'edges': new_edges}


def clip_dot_graph(dot_file_path):
    nx_g = nx.drawing.nx_agraph.read_dot(dot_file_path)
    method_node_id = 0
    for node_id in nx_g.nodes():
        label = nx_g.nodes[node_id]['label']
        if label == 'MEMBERS':
            method_node_id = node_id
            break
        else:
            continue
    sub_graph_nodes = nx.algorithms.dag.descendants(nx_g, method_node_id)
    sub_graph_nodes = sorted(sub_graph_nodes, key=lambda i: int(i))
    sub_graph_nodes = sub_graph_nodes[:-1]
    return sub_graph_nodes


def save_sample_to_jsonl_gz(function, out_file):
    writer = codecs.getwriter('utf-8')
    writer(out_file).write(json.dumps(function))
    writer(out_file).write('\n')


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def post_base_process():
    files = os.listdir(JAVA_BASE_DIR)
    chunk_files = list(chunks(files, 100000))
    print('there are %d chunks' % len(chunk_files))
    process_list = []
    for index in range(len(chunk_files)):
        process = mp.Process(target=single_post_base_process,
                             args=(chunk_files[index], index))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()
    # single_post_base_process(chunk_files[0], 0)


def single_post_base_process(files, index):
    count = 0
    if not os.path.exists(BASE_GRAPH_FILE):
        os.makedirs(BASE_GRAPH_FILE)
    with gzip.GzipFile(os.path.join(BASE_GRAPH_FILE, 'base_graph_gnn_' + str(index) + '.jsonl.gz'), 'wb') as gnn_file:
        for file in tqdm(files):
            class_name = file.split('_')[1]
            try:
                sub_graph_nodes = clip_dot_graph(
                    os.path.join(JAVA_BASE_DIR, file, class_name + '.java.dot'))
                with open(os.path.join(JAVA_BASE_DIR, file, class_name + '.java.json')) as reader:
                    graph = json.load(reader)
                with open(os.path.join(JAVA_BASE_DIR, file, class_name + '.json')) as reader:
                    raw_sample = json.load(reader)
                raw_sample['code_graph'] = normalize_graph(graph, sub_graph_nodes)
                if raw_sample['code_graph']:
                    save_sample_to_jsonl_gz(raw_sample, gnn_file)
                    count += 1
                del raw_sample
            except:
                continue
        print('there are %d samples in chunk %d' % (count, index))
        gnn_file.close()



if __name__ == '__main__':
    build_java_graphs()
    build_java_base_graphs()
    post_process()
    post_base_process()

