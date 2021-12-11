import torch
import torch.backends.cudnn as cudnn
from .model import Model
from .utils.data_utils import read_db, DataStream, read_all_Datasets
from .utils import Timer, DummyLogger, AverageMeter
from .model_handler import ModelHandler


class ModelHandlerExtend(ModelHandler):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config):
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        self._dev_loss = AverageMeter()
        self._dev_metrics = {'MRR': AverageMeter(), 'TOP1': AverageMeter(), 'TOP5': AverageMeter(),
                             'TOP10': AverageMeter(), 'NDCG1': AverageMeter(), 'NDCG5': AverageMeter(),
                             'NDCG10': AverageMeter()}

        self.model = Model(config, None)
        self.model.network = self.model.network.to(self.device)
        self.config = self.model.config
        self.is_test = False
        self.is_building = False

    def prepare_vector_db(self, file_path):
        build_set, build_code_graph_len_stats = read_db(file_path)
        print('# of database examples: {}'.format(len(build_set)))
        print('Database code graph node length: {}'.format(build_code_graph_len_stats))
        self.build_loader = DataStream(build_set, self.model.vocab_model.word_vocab,
                                       self.model.vocab_model.edge_vocab, config=self.config,
                                       isShuffle=False, isLoop=False, isSort=True,
                                       batch_size=self.config['test_batch_size'])
        self._n_build_batches = self.build_loader.get_num_batch()
        self._n_build_examples = len(build_set)

    def build_code_vec_database(self, client):
        if self.build_loader is None:
            print("No building set specified -- skipped testing.")
            return
        self.is_building = True
        timer = Timer("Building")
        for param in self.model.network.parameters():
            param.requires_grad = False
        self._run_epoch(self.build_loader, training=False, verbose=0, client=client)
        timer.finish()
        self.logger.close()

    def test(self):
        self.is_test = True
        test_set, test_src_len, test_tgt_len = read_all_Datasets(self.config['testset'], isLower=True)
        print('# of testing examples: {}'.format(len(test_set)))
        print('Test source sentence length: {}'.format(test_src_len))
        print('Test target sentence length: {}'.format(test_tgt_len))
        self.test_loader = DataStream(test_set, self.model.vocab_model.word_vocab,
                                      self.model.vocab_model.edge_vocab, config=self.config,
                                      isShuffle=False, isLoop=False, isSort=True,
                                      batch_size=self.config['test_batch_size'])
        self._n_test_batches = self.test_loader.get_num_batch()
        self._n_test_examples = len(test_set)
        timer = Timer("Test")
        for param in self.model.network.parameters():
            param.requires_grad = False
        self._run_epoch(self.test_loader, training=False, verbose=0)
        format_str = "Test -- Loss: {:0.5f}".format(self._dev_loss.mean())
        format_str += self.metric_to_str(self._dev_metrics)
        self.logger.write_to_file(format_str)
        print(format_str)
        timer.finish()
        self.logger.close()
        return self.metric_format(self._dev_metrics)

    def metric_format(self, metrics):
        for k in metrics:
            metrics[k] = metrics[k].mean()
        return metrics