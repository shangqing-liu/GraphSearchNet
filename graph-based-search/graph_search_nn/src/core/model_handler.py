import time
import torch
import torch.backends.cudnn as cudnn
from .model import Model
from .utils.data_utils import prepare_datasets, DataStream, vectorize_input
from .utils import Timer, DummyLogger, AverageMeter
from elasticsearch.helpers import bulk


class ModelHandler(object):
    """High level model_handler that trains/validates/tests the network,
    tracks and logs metrics.
    """
    def __init__(self, config):
        # Evaluation Metrics:
        self._train_loss = AverageMeter()
        self._dev_loss = AverageMeter()
        self._train_metrics = {'MRR': AverageMeter(), 'TOP1': AverageMeter(), 'TOP5': AverageMeter(),
                               'TOP10': AverageMeter(), 'NDCG1': AverageMeter(), 'NDCG5': AverageMeter(),
                               'NDCG10': AverageMeter()}
        self._dev_metrics = {'MRR': AverageMeter(), 'TOP1': AverageMeter(), 'TOP5': AverageMeter(),
                             'TOP10': AverageMeter(), 'NDCG1': AverageMeter(), 'NDCG5': AverageMeter(),
                             'NDCG10': AverageMeter()}
        self.logger = DummyLogger(config, dirname=config['out_dir'], pretrained=config['pretrained'])
        self.dirname = self.logger.dirname
        if not config['no_cuda'] and torch.cuda.is_available():
            print('[ Using CUDA ]')
            self.device = torch.device('cuda' if config['cuda_id'] < 0 else 'cuda:%d' % config['cuda_id'])
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        config['device'] = self.device

        # Prepare datasets
        datasets = prepare_datasets(config)
        self.train_set = datasets['train']
        self.dev_set = datasets['dev']

        self._n_train_examples = 0
        self.model = Model(config, self.train_set)
        self.model.network = self.model.network.to(self.device)

        self.train_loader = DataStream(self.train_set, self.model.vocab_model.word_vocab,
                                       self.model.vocab_model.edge_vocab, config=config,
                                       isShuffle=True,
                                       isLoop=True, isSort=True)
        self._n_train_batches = self.train_loader.get_num_batch()

        self.dev_loader = DataStream(self.dev_set, self.model.vocab_model.word_vocab,
                                     self.model.vocab_model.edge_vocab, config=config,
                                     isShuffle=False,
                                     isLoop=True, isSort=True)
        self._n_dev_batches = self.dev_loader.get_num_batch()

        self.config = self.model.config
        self.is_test = False
        self.is_building = False

    def train(self):
        self.is_test = False
        timer = Timer("Train")
        if self.config['pretrained']:
            self._epoch = self._best_epoch = self.model.saved_epoch
        else:
            self._epoch = self._best_epoch = 0


        self._best_metrics = {}
        for k in self._dev_metrics:
            self._best_metrics[k] = self._dev_metrics[k].mean()
        self._reset_metrics()

        while self._stop_condition(self._epoch, self.config['patience']):
            self._epoch += 1
            print("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self.logger.write_to_file("\n>>> Train Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.train_loader, training=True, verbose=self.config['verbose'])
            train_epoch_time = timer.interval("Training Epoch {}".format(self._epoch))
            format_str = "Training Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
            self.logger.write_to_file(format_str)
            print(format_str)
            print("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self.logger.write_to_file("\n>>> Dev Epoch: [{} / {}]".format(self._epoch, self.config['max_epochs']))
            self._run_epoch(self.dev_loader, training=False, verbose=self.config['verbose'])
            timer.interval("Validation Epoch {}".format(self._epoch))
            format_str = "Validation Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
            self.logger.write_to_file(format_str)
            print(format_str)
            self.model.scheduler.step(self._dev_metrics[self.config['early_stop_metric']].mean())
            if self._best_metrics[self.config['early_stop_metric']] < self._dev_metrics[self.config['early_stop_metric']].mean():
                self._best_epoch = self._epoch
                for k in self._dev_metrics:
                    self._best_metrics[k] = self._dev_metrics[k].mean()

                if self.config['save_params']:
                    self.model.save(self.dirname, self._epoch)
                    print('Saved model to {}'.format(self.dirname))
                format_str = "!!! Updated: " + self.best_metric_to_str(self._best_metrics)
                self.logger.write_to_file(format_str)
                print(format_str)

            self._reset_metrics()

        timer.finish()
        self.training_time = timer.total

        print("Finished Training: {}".format(self.dirname))
        print(self.summary())
        return self._best_metrics

    def _run_epoch(self, data_loader, training=True, verbose=10, client=None):
        start_time = time.time()
        if training:
            mode = 'train'
        elif self.is_building:
            mode = 'building'
        elif self.is_test:
            mode = 'test'
        else:
            mode = 'dev'
        code_states = []
        code_funcs = []
        file_names = []
        code_urls = []
        if training:
            self.model.optimizer.zero_grad()
        for step in range(data_loader.get_num_batch()):
            input_batch = data_loader.nextBatch()
            x_batch = vectorize_input(input_batch, training=training, device=self.device, mode=mode)
            if not x_batch:
                continue  # When there are no examples in the batch

            res = self.model.predict(x_batch, step, update=training, mode=mode)
            if 'loss' in res.keys() and 'metrics' in res.keys():
                loss = res['loss']
                metrics = res['metrics']
                self._update_metrics(loss, metrics, x_batch['batch_size'], training=training)
            if training:
                self._n_train_examples += x_batch['batch_size']

            if (verbose > 0) and (step > 0) and (step % verbose == 0):
                summary_str = self.self_report(step, mode)
                self.logger.write_to_file(summary_str)
                print(summary_str)
                print('used_time: {:0.2f}s'.format(time.time() - start_time))

            if mode == 'building':
                code_states.extend(res['code_state'])
                code_funcs.extend(res['code_funcs'])
                file_names.extend(res['file_names'])
                code_urls.extend(res['urls'])
        if mode == 'building':
            self.index_data(client, code_states, code_funcs, file_names, code_urls)

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "dev":
            format_str = "[predict-{}] step: [{} / {}] | loss = {:0.5f}".format(
                    self._epoch, step, self._n_dev_batches, self._dev_loss.mean())
            format_str += self.metric_to_str(self._dev_metrics)
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(self._best_metrics)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                if not k in metrics:
                    continue
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._dev_loss.update(loss)
            for k in self._dev_metrics:
                if not k in metrics:
                    continue
                self._dev_metrics[k].update(metrics[k], batch_size)

    def _reset_metrics(self):
        self._train_loss.reset()
        self._dev_loss.reset()

        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._dev_metrics:
            self._dev_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        """
        Checks have not exceeded max epochs and has not gone patience epochs without improvement.
        """
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.config['max_epochs']
        return False if exceeded_max_epochs or no_improvement else True

    def index_data(self, client, code_states, code_funcs, file_names, code_urls):
        requests = []
        for index in range(len(code_states)):
            request = {}
            code_state = code_states[index]
            code_func = code_funcs[index]
            file_name = file_names[index]
            url = code_urls[index]
            request['_op_type'] = 'index'
            request['_index'] = self.config['index_name']
            request['code_state'] = code_state.tolist()
            request['code_func'] = code_func
            request['identifier'] = file_name
            request['url'] = url
            requests.append(request)
            if (index + 1) % 10000 == 0:
                bulk(client, requests)
                requests = []
        if requests:
            bulk(client, requests)