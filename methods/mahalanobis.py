import os
import os.path as path
import timeit
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import global_vars as Global
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger
from methods.base_threshold import ProbabilityThreshold
from methods.binary_classifier import BinaryClassifier
from methods import AbstractModelWrapper, SVMLoss
from datasets import SubDataset, MirroredDataset
from models.classifiers import NIHDenseBinary
from collections import OrderedDict, defaultdict

class MahaModelWrapper(nn.Module):
    """
    Module that classifies data based on class conditional gaussians
    """
    def __init__(self, base_model:NIHDenseBinary, num_class=2):
        super(MahaModelWrapper, self).__init__()
        self.base_model = base_model
        self.num_class = num_class

        #self.activations = OrderedDict()
        #intermediate_nodes = [3, 5, 7, 9, 11]

        #for i in intermediate_nodes:
        #    self.track_channel_mean(self.base_model.densenet121.features[i], self.activations)
        self.trained = False

    @staticmethod
    def track_channel_mean(module, cache):
        def hook(model, input, output):
            cache[module] = output.mean(dim=(2,3))
        return module.register_forward_hook(hook)

    def collect_states(self, data_loader, device):
        all_xs = defaultdict(list)
        all_ys = []

        for x, y in tqdm(data_loader):
            x = x.to(device)
            all_ys.append(y)

            activations = OrderedDict()
            intermediate_nodes = [3, 5, 7, 9, 11]
            hooks = []
            for i in intermediate_nodes:
                hook = self.track_channel_mean(self.base_model.densenet121.features[i], activations)
                hooks.append(hook)
            _ = self.base_model.forward(x)
            for module, activation in activations.items():
                all_xs[module].append(activation.detach().cpu())
                del activation
            for hook in hooks:
                hook.remove()
            del activations
            del _
            torch.cuda.empty_cache()

        all_ys = torch.cat(all_ys, dim=0)
        self.mus = defaultdict(list)
        self.Ts = defaultdict(list)
        for layer, xs in all_xs.items():
            xs = torch.cat(xs, dim=0)
            for c in range(self.num_class):
                if len(all_ys.shape) == 1:
                    this_class_indices = torch.eq(all_ys, all_ys.new([c])).nonzero()
                else:
                    this_class_indices = torch.eq(all_ys[:, c], all_ys.new([1])).nonzero()
                this_class_x = torch.index_select(xs, 0, this_class_indices.squeeze())
                mu = torch.mean(this_class_x, dim=0)
                x_demean = (this_class_x - mu.view(1, -1))
                covar = torch.matmul(x_demean.transpose(0, 1), x_demean) / (x_demean.shape[0]-1)
                covar = covar + torch.eye(covar.shape[0])* 1e-5
                try:
                    T = torch.inverse(covar)
                except:
                    T = torch.pinverse(covar)
                self.mus[layer].append(mu)
                self.Ts[layer].append(T)

    def forward(self, x, softmax=False):
        if len(self.mus) ==0:
            return self.base_model.forward(x, softmax=softmax)
        activations = OrderedDict()
        intermediate_nodes = [3, 5, 7, 9, 11]
        hooks = []
        for i in intermediate_nodes:
            hook = self.track_channel_mean(self.base_model.densenet121.features[i], activations)
            hooks.append(hook)
        _ = self.base_model.forward(x)      # run the model once to populate hooks
        all_LL = []
        del _

        for module, activation in activations.items():
            n = activation.size(0)
            d = activation.size(1)
            activation = activation.unsqueeze(1).expand(n, 1, d)
            zs = []
            for c in range(self.num_class):
                mu = self.mus[module][c].view(1, -1).cuda()
                T = self.Ts[module][c].cuda()
                m = mu.size(0)      # should be 1
                assert m == 1
                mu = mu.unsqueeze(0).expand(n, 1, d)
                cLL = - torch.mul(torch.tensordot(activation - mu, T, dims=1), activation - mu).sum(2)  # n by 1
                zs.append(cLL)
            LL = torch.cat(zs, dim=1)
            if softmax:
                LL = nn.Softmax(LL)
            all_LL.append(LL)

        for hook in hooks:
            hook.remove()

        return all_LL



class MahaODModelWrapper(AbstractModelWrapper):
    """ The wrapper class for Mahalanobis distance OOD detection
    """

    def __init__(self, base_model:MahaModelWrapper, num_class, num_layers, epsilon=0.0012):
        """

        :param base_model: A classifier model
        :param num_class:
        :param num_layers:

        :param epsilon:
        """
        super(MahaODModelWrapper, self).__init__(base_model)

        self.num_class=num_class
        self.num_layers = num_layers
        self.H = nn.Module()
        self.H.regressor = nn.Sequential(nn.BatchNorm1d(num_layers),
                                         nn.Linear(num_layers, 1),)
        # register params under H for storage.
        self.H.register_buffer('epsilon', torch.FloatTensor([epsilon]))
        
        self.criterion = nn.CrossEntropyLoss()

    def subnetwork_eval(self, x):
        # We have to backpropagate through the input.
        # The model must be fixed in the eval mode.
        new_x = x.clone()
        cur_x = x.clone()

        grad_input_x = None
        with torch.set_grad_enabled(True):
            cur_x.requires_grad = True
            if cur_x.grad is not None:
                cur_x.grad.zero_()
            base_outputs = self.base_model(cur_x, softmax=False)
            all_layer_outputs = []
            for i, base_output in enumerate(base_outputs):
                y_hat = base_output.max(1)[1].detach()
                loss = self.criterion(base_output, y_hat)
                if i == len(base_outputs)-1:
                    grad_input_x = autograd.grad([loss], [cur_x], retain_graph=False, only_inputs=True)[0]
                else:
                    grad_input_x = autograd.grad([loss], [cur_x], retain_graph=True, only_inputs=True)[0]
                with torch.set_grad_enabled(False):
                    new_input = (new_x.detach() - self.H.epsilon * (grad_input_x.detach().sign()))
                    new_input.detach_()
                    new_input.requires_grad = False

                    # second evaluation.
                    new_output = self.base_model(new_input, softmax=False)[i].detach()
                    input = base_output.max(1)[0].detach().unsqueeze_(1)
                    all_layer_outputs.append(input)
                del base_output
            del base_outputs
            all_layer_outputs = torch.cat(all_layer_outputs, dim=1)     # batchsize x num_layers
        return all_layer_outputs


        # new_input = (new_x.detach() - self.H.epsilon * (grad_input_x.detach().sign()))
        # new_input.detach_()
        # new_input.requires_grad = False
        #
        # # second evaluation.
        # new_output = self.base_model(new_input, softmax=False).detach()
        #
        # new_output.mul_(self.H.temperature)

        #probabilities = F.softmax(new_output, dim=1)

        # Get the max probability out
        #input = probabilities.max(1)[0].detach().unsqueeze_(1)

        #return input.detach()

    def wrapper_eval(self, x):
        pred = self.H.regressor(x)
        return pred

    def classify(self, x):
        return (x > 0).long()


class MahalanobisDetector(ProbabilityThreshold):
    def method_identifier(self):
        output = "MAHA"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def propose_H(self, dataset):
        config = self.get_base_config(dataset)

        from models import get_ref_model_path
        h_path = get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name)
        best_h_path = path.join(h_path, 'model.best.pth')

        if not path.isfile(best_h_path):
            raise NotImplementedError("Please use model_setup to pretrain the networks first!")
        else:
            print(colored('Loading H1 model from %s' % best_h_path, 'red'))
            config.model.load_state_dict(torch.load(best_h_path))

        # trainer.run_epoch(0, phase='all')
        # test_average_acc = config.logger.get_measure('all_accuracy').mean_epoch(epoch=0)
        # print("All average accuracy %s"%colored('%.4f%%'%(test_average_acc*100), 'red'))

        self.base_model = MahaModelWrapper(config.model, 2)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True,
                            num_workers=self.args.workers, pin_memory=True)
        self.base_model.collect_states(loader, self.args.device)
        self.base_model.eval()

    def get_H_config(self, train_ds, valid_ds, will_train=True, epsilon=0.0012):
        print("Preparing training D1+D2 (H)")

        # Initialize the multi-threaded loaders.
        train_loader = DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=self.args.workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=self.args.workers, pin_memory=True)

        # Set up the criterion
        criterion = nn.BCEWithLogitsLoss().cuda()
        # Set up the model
        model = MahaODModelWrapper(self.base_model, epsilon=epsilon, num_class=2, num_layers=5).to(self.args.device)

        old_valid_loader = valid_loader
        if will_train:
            # cache the subnetwork for faster optimization.
            from methods import get_cached
            from torch.utils.data.dataset import TensorDataset

            trainX, trainY = get_cached(model, train_loader, self.args.device)
            validX, validY = get_cached(model, valid_loader, self.args.device)

            new_train_ds = TensorDataset(trainX, trainY)
            new_valid_ds = TensorDataset(validX, validY)

            # Initialize the new multi-threaded loaders.
            train_loader = DataLoader(new_train_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)
            valid_loader = DataLoader(new_valid_ds, batch_size=2048, shuffle=True, num_workers=0, pin_memory=False)

            # Set model to direct evaluation (for cached data)
            model.set_eval_direct(True)

        # Set up the config
        config = IterativeTrainerConfig()

        base_model_name = self.base_model.__class__.__name__
        if hasattr(self.base_model, 'preferred_name'):
            base_model_name = self.base_model.preferred_name()

        config.name = '_%s[%s](%s-%s)' % (self.__class__.__name__, base_model_name, self.args.D1, self.args.D2)
        config.train_loader = train_loader
        config.valid_loader = valid_loader
        config.phases = {
            'train': {'dataset': train_loader, 'backward': True},
            'test': {'dataset': valid_loader, 'backward': False},
            'testU': {'dataset': old_valid_loader, 'backward': False},
        }
        config.criterion = criterion
        config.classification = True
        config.cast_float_label = True
        config.stochastic_gradient = True
        config.visualize = not self.args.no_visualize
        config.model = model
        config.optim = optim.Adagrad(model.H.parameters(), lr=1e-3)
        config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(config.optim, patience=5, threshold=1e-1, min_lr=1e-6,
                                                                factor=0.1, verbose=True)
        config.logger = Logger()
        config.max_epoch = 100
        return config

    def train_H(self, dataset):
        # Wrap the (mixture)dataset in SubDataset so to easily
        # split it later.
        dataset = SubDataset('%s-%s' % (self.args.D1, self.args.D2), dataset, torch.arange(len(dataset)).int())

        # 80%, 20% for local train+test
        train_ds, valid_ds = dataset.split_dataset(0.8)

        if self.args.D1 in Global.mirror_augment:
            print(colored("Mirror augmenting %s" % self.args.D1, 'green'))
            new_train_ds = train_ds + MirroredDataset(train_ds)
            train_ds = new_train_ds

        # As suggested by the authors.
        all_epsilons = torch.linspace(0, 0.004, 21)
        total_params = len(all_epsilons)
        best_accuracy = -1

        h_path = path.join(self.args.experiment_path, '%s' % (self.__class__.__name__),
                           '%d' % (self.default_model),
                           '%s-%s.pth' % (self.args.D1, self.args.D2))
        h_parent = path.dirname(h_path)
        if not path.isdir(h_parent):
            os.makedirs(h_parent)

        done_path = h_path + '.done'
        trainer, h_config = None, None

        if self.args.force_train_h or not path.isfile(done_path):
            # Grid search over the temperature and the epsilons.
            for i_eps, eps in enumerate(all_epsilons):
                so_far = i_eps + 1
                print(colored('Checking eps=%.2e (%d/%d)' % (eps, so_far, total_params), 'green'))
                start_time = timeit.default_timer()
                h_config = self.get_H_config(train_ds=train_ds, valid_ds=valid_ds,
                                             epsilon=eps)

                trainer = IterativeTrainer(h_config, self.args)

                print(colored('Training from scratch', 'green'))
                trainer.run_epoch(0, phase='test')

                for epoch in range(1, h_config.max_epoch):
                    trainer.run_epoch(epoch, phase='train')
                    trainer.run_epoch(epoch, phase='test')

                    train_loss = h_config.logger.get_measure('train_loss').mean_epoch()
                    h_config.scheduler.step(train_loss)

                    # Track the learning rates and threshold.
                    lrs = [float(param_group['lr']) for param_group in h_config.optim.param_groups]
                    h_config.logger.log('LRs', lrs, epoch)
                    h_config.logger.get_measure('LRs').legend = ['LR%d' % i for i in range(len(lrs))]

                    if h_config.visualize:
                        # Show the average losses for all the phases in one figure.
                        h_config.logger.visualize_average_keys('.*_loss', 'Average Loss', trainer.visdom)
                        h_config.logger.visualize_average_keys('.*_accuracy', 'Average Accuracy', trainer.visdom)
                        h_config.logger.visualize_average('LRs', trainer.visdom)

                    test_average_acc = h_config.logger.get_measure('test_accuracy').mean_epoch()

                    # Save the logger for future reference.
                    torch.save(h_config.logger.measures,
                               path.join(h_parent, 'logger.%s-%s.pth' % (self.args.D1, self.args.D2)))

                    if best_accuracy < test_average_acc:
                        print('Updating the on file model with %s' % (colored('%.4f' % test_average_acc, 'red')))
                        best_accuracy = test_average_acc
                        torch.save(h_config.model.state_dict(), h_path)

                    if test_average_acc > 1 - 1e-4:
                        break

                elapsed = timeit.default_timer() - start_time
                print('Hyper-param check %.2e in %.2fs' % (eps, elapsed))

            torch.save({'finished': True}, done_path)

        # If we load the pretrained model directly, we will have to initialize these.
        if trainer is None or h_config is None:
            h_config = self.get_H_config(train_ds=train_ds, valid_ds=valid_ds,
                                         epsilon=0, temperature=1, will_train=False)
            # don't worry about the values of epsilon or temperature. it will be overwritten.
            trainer = IterativeTrainer(h_config, self.args)

        # Load the best model.
        print(colored('Loading H model from %s' % h_path, 'red'))
        state_dict = torch.load(h_path)
        for key, val in state_dict.items():
            if val.shape == torch.Size([]):
                state_dict[key] = val.view((1,))
        h_config.model.H.load_state_dict(state_dict)
        h_config.model.set_eval_direct(False)
        print('Epsilon %s' % (colored(h_config.model.H.epsilon.item(), 'red')))

        trainer.run_epoch(0, phase='testU')
        test_average_acc = h_config.logger.get_measure('testU_accuracy').mean_epoch(epoch=0)
        print("Valid/Test average accuracy %s" % colored('%.4f%%' % (test_average_acc * 100), 'red'))
        self.H_class = h_config.model
        self.H_class.eval()
        self.H_class.set_eval_direct(False)
        return test_average_acc
