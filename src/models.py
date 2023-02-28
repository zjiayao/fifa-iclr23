import logging
import numpy as np
import pandas as pd
import scipy
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.datasets
from .resnet import resnet18,resnet34,resnet50,resnet101, NormedLinear
from .variable_width_resnet import resnet10vw,resnet18vw,resnet50vw
from .vggnet import vgg11, vgg13, vgg16, vgg19
from pathlib import Path

logger = logging.getLogger(__name__)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        logger.debug("using NormedLinear")
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class MLP(nn.Module):
    def __init__(self, input_shape, n_classes, n_hiddens, dropout=False, **kwargs):
        super(MLP, self).__init__()
        self.n = len(n_hiddens)
        d0 = np.prod(input_shape)
        self.d0 = d0
        n_hiddens = [d0] + n_hiddens
        self.layers = []
        for i in range(1, self.n+1):
            self.layers += [
                nn.Linear(n_hiddens[i-1], n_hiddens[i],), nn.ReLU(inplace=False)
            ]
            if dropout:
                self.layers += [nn.Dropout(p=0.2)]

        self.features = nn.Sequential(*self.layers)
        self.fc1 = nn.Linear(n_hiddens[-1], n_classes)

    
    def forward(self, x):
        x = x.view(-1,self.d0)
        return self.fc1(self.features(x))
    

class TwoLayerNN(nn.Module):
    def __init__(self, input_shape, n_classes, n_hidden=1000, dropout=False, **kwargs):
        super(TwoLayerNN, self).__init__()
        self.d0 = np.prod(input_shape)
        self.d1 = n_hidden
        self.d2 = n_classes
        self.fc0 = nn.Linear(self.d0, self.d1,)
        self.fc1 = nn.Linear(self.d1, self.d2,)
        self.do = nn.Dropout(p=0.2)
        self.dropout = dropout

    def partial_fwd(self, x):
        x = F.relu(self.fc0(x.view(-1,self.d0)))
        return x
    
    def forward(self, x):
        x = F.relu(self.fc0(x.view(-1,self.d0)))
        if self.dropout:
            x = self.do(x)
        x = self.fc1(x)
        return x

    
class OneLayerNN(nn.Module):
    def __init__(self, input_shape, n_classes, **kwargs):
        super(OneLayerNN, self).__init__()
        self.d0 = np.prod(input_shape)
        self.d1 = n_classes
        self.fc0 = NormedLinear(self.d0, self.d1)

    def forward(self, x):
        x = self.fc0(x.view(-1, self.d0))
        return x
    
class ThreeLayerNN(nn.Module):
    def __init__(self, input_shape, n_classes, n_hidden=128, **kwargs):
        super(ThreeLayerNN, self).__init__()
        self.d0 = np.prod(input_shape)
        self.d1 = n_hidden
        self.d2 = n_hidden
        self.d3 = n_classes
        self.fc0 = nn.Linear(self.d0, self.d1)
        self.fc1 = nn.Linear(self.d1, self.d2)
        self.fc2 = nn.Linear(self.d2, self.d3)

    def forward(self, x):
        x = F.relu(self.fc0(x.view(-1,self.d0)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SimpleConv(nn.Module):
    def __init__(self, input_shape, n_classes, **kwargs):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, n_classes, bias=False) #120)

    def partial_fwd(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        return x
    
class SimpleConv3d(nn.Module):
    def __init__(self, input_shape, n_classes, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLPWrapper:
    import sklearn.metrics as skm
    import scipy.special
    
    def clone(self):
        atrs = [i for i in self.__dict__.keys() if i[:1] != '_']
        return MLPWrapper(**{k:self.__dict__[k] for k in atrs})


    def __init__(self, *,  model_func, loss_func, opt_func, metric_func, model_kwargs={}, loss_kwargs={}, opt_kwargs={}, 
                 max_iter=200, min_iters=100, init_func=torch.nn.init.xavier_normal_,
                 device='cpu:0',verbose=True,log_metrics=True,
                 log_metrics_func=lambda itr: False,
                 stop_at_tr_err=None, manual_init=True,
                 pred_batch_size=None, save_path=None,
                 wandb_run=None,
                 **kwargs
                ):

        self.model_func = model_func
        self.model_kwargs = model_kwargs
        self.loss_func = loss_func
        self.loss_kwargs = loss_kwargs
        self.opt_func = opt_func
        self.opt_kwargs = opt_kwargs
        
        self.init_func = init_func
        self.model = None
        self.opt = None

        self.last_history = None
        self.device = device
        self.verbose = verbose
        self.max_iter = max_iter
        self.min_iters = min_iters
        self.metric_func = metric_func
        
        self.softmax = scipy.special.softmax
        self.log_metrics = log_metrics
        self.log_metrics_func = log_metrics_func
        
        self.stop_at_tr_err = stop_at_tr_err
        
        self.manual_init = manual_init
        self.pred_batch_size=pred_batch_size
        self.save_path = save_path
        
        self.wandb_run = wandb_run
        
        
    def change_save_path(self, path):
        self.save_path = path
        
        
    def _hist_save_dict(self, iloc):
        ret = {}
        try:
            dt = self.last_history.iloc[iloc]
            ret['itr'] = dt['itr']
            ret['loss'] = dt['loss']
            ret['acc'] = dt['acc']
            ret['type'] = dt['type']
        except:
            pass
        return ret
    
    def _hist_save_dict_last(self):
        return {**self._hist_save_dict(-1), **self._hist_save_dict(-2)}
        
    def save(self, path):
        if not isinstance(path, Path):
            try:
                path = Path(path)
            except Exception as e:
                logger.critical(f"Error saving model: {e}")
                
        if self.model is not None:
            save_dict = {**{
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                }, **self._hist_save_dict_last(),
            }
            logger.debug(f"saving model to {path}")
            torch.save(save_dict,path)
    def _init_model(self):
        if self.model is not None and self.manual_init:
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear):
                    self.init_func(m.weight)
         
    def _prepare_model(self):
        if self.verbose:
            print("Generating new model")
        self.model = self.model_func(**self.model_kwargs)
        self._init_model()
        self.model.to(self.device)
        self.opt = self.opt_func(filter(lambda p: p.requires_grad, self.model.parameters()), 
                            **self.opt_kwargs)

    def _prepare_fit_data(self, X, y, A=None):
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(A, pd.Series):
            A = A.values 
        if isinstance(X, src.datasets.PseudoData):
            X = X.to(self.device)
            
        return torch.Tensor(X.values).float().to(self.device) if isinstance(X, pd.DataFrame) else X,  torch.Tensor(y).long().to(self.device) if y is not None else y, A 

    def eval(self, X, y, sample_weight=None, group=None, label=None, size_map=None):
        orig_data = [X, y, group]
        _, y, group = self._prepare_fit_data(None, y, group)

        n_samples = len(X)
        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        
        y_pred = self._output(X)
        y = y.to(y_pred.device)
        
        sample_weight = torch.Tensor(sample_weight).to(y_pred.device)
        
        loss = self.loss_func(sample_weight=sample_weight, sample_type='test', **self.loss_kwargs)
        ls = loss(y_pred, y,
                 label=orig_data[1],
                 group=orig_data[2],
                 )

        metric =self._log_metric(y_pred, y, ls.item(), group=orig_data[2], 
                                 label=label,size_map=size_map,
                                 metric_type='test')
        return metric
    
    def _batched_out(self, data, batch_size=100):
        """
        get model outputs of data in a batched fashion
        to avoid out-of-memory errors
        """
        res = []
        nb = data.shape[0] // batch_size
        for i in tqdm.tqdm(range(nb)):
            res.append( self.model(data[i*batch_size:(i+1)*batch_size]).detach().cpu() )

        if (nb * batch_size) < data.shape[0]:
            res.append( self.model(data[nb*batch_size:]).detach().cpu() )
        return torch.vstack(res)

    
    def _output(self, X, batch_size=None):
        if batch_size is None:
            batch_size = self.pred_batch_size
            
        X, _, _ = self._prepare_fit_data(X, y=None, A=None)
        
        self.model.eval()
        if batch_size is not None:
            y_pred = self._batched_out(X, batch_size=batch_size)
        else:
            y_pred = self.model(X)
        return y_pred
    
    def _torch_to_np(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.detach().cpu().numpy()
    
    def _to_prob(self, out):
        return self.softmax(out, axis=1)
    
    def predict_and_proba(self, X):
        y = self._output(X)
        pred = torch.max(y, -1)[1].detach().cpu().numpy()
        proba = self._to_prob(self._torch_to_np(y))
        return pred, proba
        
    
    def predict(self, X):
        return torch.max(self._output(X), -1)[1].detach().cpu().numpy()
        
        
    def predict_proba(self, X):
        return self._to_prob(self._torch_to_np(self._output(X)))
    
    
    def _log_metric(self, y_pred, y_batch, ls_val, group=None, label=None, size_map=None, metric_type="train", ):
        metric = self.metric_func(self._to_prob(self._torch_to_np(y_pred)), 
                                          self._torch_to_np(y_batch), 
                                          group=group,label=label,size_map=size_map)
        metric['loss'] = ls_val
        metric['type'] = metric_type
        try:
            if 'eo' in metric and 'cls_avg_acc' in metric:
                metric['balloss_cls_eo'] = (1.0-metric['cls_avg_acc']) * 0.5 + 0.5 * metric['eo']
                metric['balloss_cls_dp'] = (1.0-metric['cls_avg_acc']) * 0.5 + 0.5 * metric['dp']


        except Exception as e:
            logger.critical(f"error logging eo metric: {e}")
        return metric
                
    def _log_wandb_metric(self, wandb_run, metric, itr, metric_type="train"):
        if wandb_run is not None:
            dt = metric.drop(['itr', 'type'],errors='ignore').iloc[0].to_dict()
            d = {f"{metric_type}_{k}":v for k,v in dt.items()}
            wandb_run.log(d,step=itr)
        
    def _acc(self, y_pred, y_true):
        return torch.sum(torch.max(y_pred, -1)[1] == y_true).item() * 1. / y_pred.shape[0]
        
    def fit(self, X, y, sample_weight=None, 
            y_orig=None,
            group=None,
            X_test=None, y_test=None, group_test=None,
            n_iters=None, min_iters=None,
            continue_training=False, 
            batch_size=None, balance_labels=False, 
            wandb_run=None, imbalance_method='none',
            drw_beta=0., drw_ep=None,
            size_map_handle=lambda **kwargs:None,
            **kwargs):
        if wandb_run is None:
            wandb_run = self.wandb_run
        if n_iters is None:
            n_iters = self.max_iter
        if min_iters is None:
            min_iters = self.min_iters
        if self.model is None or not continue_training:
            self._prepare_model()
        if y_orig is None:
            y_orig = y
            logger.debug("y_orig set to y")
        do_eval = (X_test is not None) and (y_test is not None)
        n_samples = len(X)
        orig_data = [X, y_orig, group]
        
#         logger.debug(f"Entering fitting loop, max_iter: {n_iters}, do_eval: {do_eval}, n_samples: {n_samples}, batch_size: {batch_size}")
        
        X, y, group = self._prepare_fit_data(X, y, group)

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        
        sample_weight = torch.Tensor(sample_weight).to(self.device)
        loss = self.loss_func(sample_weight=sample_weight, sample_type='train', **self.loss_kwargs)
        metrics = []
        if self.verbose:
            tqdmr = tqdm.tqdm(range(n_iters))
        else:
            tqdmr = range(n_iters)
            
        X_batch, y_batch = X, y
        for it in tqdmr:
            ep_per_it = 1 if batch_size is None else batch_size / n_samples
            ep = np.floor(it * ep_per_it)
            self.model.train()
            label_series = orig_data[1]
            group_series = orig_data[2]

            if balance_labels or (batch_size is not None):
                if balance_labels:
                    indices = utils.get_balanced_indices(X, y, batch_size)
                else:
                    indices = np.random.choice(np.arange(n_samples),batch_size)
                
                X_batch = X[indices]
                y_batch = y[indices]
#                 y_orig_batch = y_orig[indices].reset_index(drop=True)

                if orig_data[1] is not None:
                    label_series = orig_data[1].iloc[indices].reset_index(drop=True)
                if group_series is not None:
                    group_series = orig_data[2].iloc[indices].reset_index(drop=True)
                loss = self.loss_func(sample_weight=sample_weight[indices], sample_type='train', **self.loss_kwargs)
                
                
            if drw_ep is not None and ep >= drw_ep:
                if hasattr(loss, 'set_beta'):
#                     logger.debug(f"ep: {ep} >= {drw_ep}")
                    loss.set_beta(drw_beta)

            y_pred = self.model(X_batch)
            ls = loss(y_pred, y_batch, 
                      label=label_series,
                      group=group_series,
                     )

            if self.log_metrics and self.log_metrics_func(it):
                # compute training metrics wrt true labels
                # not the reducted labels
                metric = self._log_metric(y_pred, label_series.values, ls.item(), 
                                          group=group_series, label=label_series,
                                         size_map=size_map_handle(sample_type='train'))
                self._log_wandb_metric(wandb_run, metric=metric, itr=it, metric_type="train")

                if do_eval:
                    val_metric = self.eval(X_test, y_test, group=group_test, label=y_test,
                                          size_map=size_map_handle(sample_type='test'))
                    self._log_wandb_metric(wandb_run, metric=val_metric, itr=it, metric_type="test")
                    
                    metric = pd.concat([metric, val_metric])

                metric['itr'] = it  
                metrics.append(metric)
            
            if self.stop_at_tr_err is not None:
                _tr_err = 1.- self._acc(y_pred, y_batch)

#                 if (it > min_iters) and self.verbose and wandb_run is None:
#                     logger.debug(f"[{it:05d}] Training error: {_tr_err}")
#                     print(f"[{it:05d}] tr acc: {_tr_err}")
                if (it > min_iters) and (_tr_err < self.stop_at_tr_err):
                    logger.debug(f"Early stpping at {it:05d} ")
                    print(f"Early stopping criterion reached at {it:05d}]: tr_err = {_tr_err:.4f} < {self.stop_at_tr_err}")
                    break
            
            ls.backward()
            self.opt.step()
            self.opt.zero_grad()
            
            if self.save_path is not None and (it % 1000 == 0):
                self.save(self.save_path)
            
            
            if (self.log_metrics and self.log_metrics_func(it)) and self.verbose:
                tqdmr.set_description(f"Itr: {it:03d} Ep: {ep:.1f} it/ep: {1./ep_per_it:.2f} tr_acc: {metric['acc'].iloc[0]:.2f} tr_loss: {metric['loss'].iloc[0]:.2f} avg_y_pred: {torch.mean(torch.abs(y_pred)).item():.3f}")
                tqdmr.refresh()
                
        if self.log_metrics:
            y_pred = self.model(X_batch)
            ls = loss(y_pred, y_batch,
                     label=label_series,
                      group=group_series,
                     )

            metric = self._log_metric(y_pred, label_series.values, ls.item(), 
                                          group=group_series, label=label_series,
                                         size_map=size_map_handle(sample_type='train'))
            self._log_wandb_metric(wandb_run, metric=metric, itr=it, metric_type="train")
            if do_eval:
                val_metric = self.eval(X_test, y_test, group=group_test, label=y_test,
                                          size_map=size_map_handle(sample_type='test'))
                self._log_wandb_metric(wandb_run, metric=val_metric, itr=it, metric_type="test")
                metric = pd.concat([metric, val_metric])
            metric['itr'] = it+1
            metrics.append(metric)
            
            metric_df = pd.concat(metrics)
            self.last_history = metric_df.reset_index(drop=True)
        if self.save_path is not None:
            try:
                self.save(self.save_path)
            except:
                pass
            
        