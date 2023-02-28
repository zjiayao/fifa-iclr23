import multiprocessing, os, sys, logging
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
import uuid
import wandb
import torch
import src.utils as utils
import src.models as models
import src.datasets as datasets
import src.metrics
import fairlearn
import fairlearn.metrics
from fairlearn.reductions import ExponentiatedGradient, GridSearch
logger = logging.getLogger(__name__)
def load_model(path, train_layer=None, 
               replace_layer={},
               map_location=torch.device('cpu'),
               **kwargs):
    if not isinstance(path, Path):
        path = Path(path)
    logger.debug(f"load model from {path}")
    model = torch.load(path, map_location=map_location, **kwargs) 
    for ly, new_ly_func in replace_layer.items():
        old_layer = getattr(model, ly)
        logger.debug(f"changing layer {ly} with shape {old_layer.weight.shape}")
        _out, _in = old_layer.weight.shape
        new_layer = new_ly_func(in_features=_in, out_features=_out)
        if new_layer.weight.shape == old_layer.weight.shape:
            new_layer.weight.data = old_layer.weight.data
        else:
            new_layer.weight.data = old_layer.weight.data.T
        setattr(model, ly, new_layer)
        logger.debug(f"changed layer {ly} with shape {new_layer.weight.shape}")
    if train_layer is not None and isinstance(train_layer, str):
        for name, param in model.named_parameters():
            if not name.startswith(train_layer):
                param.requires_grad = False
            else:
                logger.debug(f"only train layer {name} ({train_layer})")
                param.requires_grad = True
    else:
        logger.debug(f"training all layers")

    return model

def console_log(msg, end='\n'):
    os.write(1, ('[LOG/{}]'.format(multiprocessing.current_process().name)+msg+end).encode('utf-8'))
    
def set_seed(seed):
    seed = hash(seed)%(2*32-1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_effectiven_n(map_df, lb, alpha):
    # compute effective sample size
    dt = map_df.xs(lb)
    ns = dt.values.reshape(-1)
    n = np.sum(ns)
    n_prods = np.prod(ns)

    denom = np.sqrt(n_prods) + alpha * np.sqrt(n_prods*n) * np.sum([np.sqrt( 1./nn ) for nn in ns])
    return n*n_prods / denom ** 2
        

def get_n_from_index(map_df, idx,):
    # get sample_size n for the subgroup identified by index idx
    return map_df.loc[idx].iloc[0]

def get_ns_from_lb(map_df, lb):
    # get sample_sizes for each subgroup in label class lb
    return map_df.xs(lb).values.reshape(-1)

def get_idx_from_idx(lb_df:pd.Series,
                     grp_df:pd.Series,
                     lb:float,
                     g:str):
    # get sample indices by selecting
    # the label class and the group class
#     print(lb, g, np.sum(lb_df==lb), np.sum(grp_df==g),
#           np.sum((lb_df==lb) & (grp_df==g))
#              )
    return np.where((lb_df==lb) & (grp_df==g))[0]
    
def get_idx_from_lb(lb_df:pd.Series, lb:float):
    # get sample indices of the label class lb
    return np.where(lb_df.reset_index(drop=True)==lb)[0]

def get_lb_grp_row(df, y, s):
    # similar to get_idx_from_idx
    return df[(df[_LABEL]==y) & (df[_GROUP_ID]==s)].iloc[0]

    
class ConfigArgs(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    
def summary_as_df(name, summary):
    a = summary.by_group
    a['overall'] = summary.overall
    return pd.DataFrame({name: a})

def gen_balanced_indices(X, y, batch_size=None):
    if batch_size is None:
        batch_size = np.min(
            [torch.sum(y==0).item(),
             torch.su(y==1).item()])
    bal_indices_0 = np.random.choice(
        torch.where(y==0)[0].cpu().numpy(),
        batch_size//2
    )
    bal_indices_1 = np.random.choice(
        torch.where(y==1)[0].cpu().numpy(),
        batch_size//2
    )

    return np.hstack([bal_indices_0,bal_indices_1])

def z_normalization(df, df_test=None, excl=[], eps=1e-8):
    cols = df.columns
    for  col in cols:
        if col in excl:
            continue
        mu, std = df[col].describe()[['mean','std']]
        std = 1. if np.abs(std) < eps else std
        df[col] = (df[col]-mu)/std
        if df_test is not None:
            df_test[col] = (df_test[col]-mu)/std
    
def load_saved_lambdas(path, data='celeba'):
    if not isinstance(path, Path):
        path = Path(path)
    grid = None
    grid = pd.read_csv(path).set_index(["sign","event","group_id"])
    return grid
    
    
def fit_simple_bin_expgradient_eps(regressor, constraints_func, epss, n_rep, dataset):
    """
    
    dataset: [X_train, y_train, A_train, X_test, y_test, A_test]
    """
    [X_train, y_train, A_train, X_test, y_test, A_test] = dataset
    As = list(A_train.unique())
    dfs = []

    for eps in tqdm.tqdm(epss):
        for _ in range(n_rep):
            expgrad_model = ExponentiatedGradient(
                regressor,
                constraints=constraints_func(difference_bound=eps), # eps in c
                eps=eps, # L1 ball bounding lambda
                nu=1e-6)

            expgrad_model.fit(
                X_train,
                y_train,
                sensitive_features=A_train,
            )

            
            ## test sores
            expgrad_model_scores_te = pd.Series(expgrad_model.predict(X_test), name="scores")

            te_df = summary_as_df(
                "scores",
                fairlearn.metrics.MetricFrame(metrics=fairlearn.metrics.mean_prediction, y_true=y_test, y_pred=expgrad_model_scores_te, 
                             sensitive_features=A_test)
            )
            te_df.loc['disparity']=(te_df.loc[As[0]]-te_df.loc[As[1]]).abs()
            te_df['err'] = np.mean(y_test!=expgrad_model_scores_te)
            te_df['type'] = 'test'
            

            ## train scores
            expgrad_model_scores_tr = pd.Series(expgrad_model.predict(X_train), name="scores")

            tr_df = summary_as_df(
                "scores", fairlearn.metrics.MetricFrame(metrics=fairlearn.metrics.mean_prediction, y_true=y_train, y_pred=expgrad_model_scores_tr, 
                             sensitive_features=A_train)
            )
            tr_df.loc['disparity']=(tr_df.loc[As[0]]-tr_df.loc[As[1]]).abs()
            tr_df['err'] = np.mean(y_train!=expgrad_model_scores_tr)
            tr_df['type'] = 'train'


            df = pd.concat([tr_df, te_df])
            df['eps'] = eps

            df['eid'] = str(uuid.uuid4()).split('-')[0]
            dfs.append(df)
    return pd.concat(dfs).reset_index()

def log_expg_metrics(expg, dataset, 
                     size_map_handle=lambda x: None,
                    wandb_run=None):
    [X_train, y_train, A_train, X_test, y_test, A_test] = dataset
    y_pred_tr = expg._pmf_predict(X_train)
    y_pred_te = expg._pmf_predict(X_test)
    
    
    met_tr = src.metrics.bin_classification_metrics(y_pred_tr, y_train.values,
            group=A_train,label=y_train,size_map=size_map_handle('train'))
    met_tr['type'] = 'train'
    
    if 'eo' in met_tr and 'cls_avg_acc' in met_tr:
        met_tr['balloss_cls_eo'] = (1.0-met_tr['cls_avg_acc']) * 0.5 + 0.5 * met_tr['eo']
        met_tr['balloss_cls_dp'] = (1.0-met_tr['cls_avg_acc']) * 0.5 + 0.5 * met_tr['dp']
    
    met_te = src.metrics.bin_classification_metrics(y_pred_te, y_test.values,
            group=A_test,label=y_test,size_map=size_map_handle('test'))
    met_te['type'] = 'test'
    if 'eo' in met_te and 'cls_avg_acc' in met_te:
        met_te['balloss_cls_eo'] = (1.0-met_te['cls_avg_acc']) * 0.5 + 0.5 * met_te['eo']
        met_te['balloss_cls_dp'] = (1.0-met_te['cls_avg_acc']) * 0.5 + 0.5 * met_te['dp']
    
    if wandb_run is not None:
        dt_tr = met_tr.drop(['itr', 'type'],errors='ignore').iloc[0].to_dict()
        d_tr = {f"train_{k}":v for k,v in dt_tr.items()}

        dt_te = met_te.drop(['itr', 'type'],errors='ignore').iloc[0].to_dict()
        d_te = {f"test_{k}":v for k,v in dt_te.items()}
        wandb_run.log({**d_tr, **d_te})
    return met_tr, met_te
    

def fit_bin_expgradient_eps(regressor, constraints_func, epss, dataset,
                           wandb_run=None, fit_kwargs={},
                            do_callback=True,
                            **kwargs):
    """
    
    dataset: [X_train, y_train, A_train, X_test, y_test, A_test]
    """
#     print('callback: ', do_callback, 'wb: ', wandb_run is None,
#          ' kwargs: ', fit_kwargs)
    [X_train, y_train, A_train, X_test, y_test, A_test] = dataset
    As = list(A_train.unique())
    for eps in tqdm.tqdm(epss):

        expgrad_model = ExponentiatedGradient(
            regressor,
            constraints=constraints_func(difference_bound=eps), # eps in c
            eps=eps, # L1 ball bounding lambda
            nu=1e-6,fit_kwargs=fit_kwargs)
            
        expgrad_model.fit(
            X_train,
            y_train,
            sensitive_features=A_train,
        )
        
        if do_callback:
            size_map_handle = fit_kwargs['size_map_handle'] if 'size_map_handle' in fit_kwargs else None

            met_tr, met_te = log_expg_metrics(expgrad_model, dataset, size_map_handle=size_map_handle, wandb_run=wandb_run)
#             logger.debug(f"tr: {met_tr}\nte: {met_te}")
    

def fit_bin_gridsearch_eps(regressor, constraints_func, epss, dataset,
                           grid=None, grid_size=50, grid_limit=10, 
                           keep_all_predictors=False,
                           wandb_run=None, fit_kwargs={}, save_path=None,
                           do_callback=True,call_estimator_to_complete=True,
                           size_map_handle=lambda x:None,
                           **kwargs,
                          ):
    """
    
    dataset: [X_train, y_train, A_train, X_test, y_test, A_test]
    """
    logger.debug(f"wandb_run is None: {wandb_run is None}")
    [X_train, y_train, A_train, X_test, y_test, A_test] = dataset
    As = list(A_train.unique())
    dfs = []
    eps_metrics = []
    for eps in tqdm.tqdm(epss):
        model_uuid = str(uuid.uuid4()).split('-')[0]
        
        if save_path is not None:
            try:
                ename = "" if wandb_run is None else wandb_run.config['exp_name']
            except:
                ename = ""
            pth = Path(save_path) / f"{ename}.checkpoint"
            regressor.change_save_path(pth)
        

        sweep = GridSearch(
                regressor,
                constraints=constraints_func(difference_bound=eps), 
                grid=grid,
                grid_size=grid_size, grid_limit=grid_limit,
                keep_all_predictors=keep_all_predictors,  
                fit_kwargs=fit_kwargs,
                call_estimator_to_complete=call_estimator_to_complete
            )

        
        metrics = []
        def met_call_back(predictor, lbd, best=False, probas=None,bal_loss=None, **kwargs):
            if probas is not None:
                y_pred_p = probas
            else:
                y_pred_p = predictor.predict_proba(X_train)
            tr_met = src.metrics.bin_classification_metrics(y_pred=y_pred_p,
                        y_true=y_train.values, group=A_train,
                        label=y_train, size_map=size_map_handle('train'))

    
            tr_met['balloss_cls_eo'] = (1.0-tr_met['cls_avg_acc']) * 0.5 + 0.5 * tr_met['eo']
            tr_met['balloss_cls_dp'] = (1.0-tr_met['cls_avg_acc']) * 0.5 + 0.5 * tr_met['dp']
            
            tr_met['lambda'] = lbd
            tr_met['eps'] = eps
            tr_met['best'] = best
            tr_met['model_id'] = model_uuid
            tr_met['bal_loss'] = bal_loss
            
            y_test_pred_p = predictor.predict_proba(X_test)
            
            te_met = src.metrics.bin_classification_metrics(y_pred=y_test_pred_p,
                                                y_true=y_test.values, group=A_test,
                                                label=y_test, size_map=size_map_handle('test'))
            te_met['balloss_cls_eo'] = (1.0-te_met['cls_avg_acc']) * 0.5 + 0.5 * te_met['eo']
            te_met['balloss_cls_dp'] = (1.0-te_met['cls_avg_acc']) * 0.5 + 0.5 * te_met['dp']
            te_met['lambda'] = lbd
            te_met['eps'] = eps
            te_met['best'] = best
            te_met['model_id'] = model_uuid
            te_met['bal_loss'] = bal_loss
            
            
            
            if wandb_run is not None:
                logger.debug("wandb is not None, logging metrics")
                dt_tr = tr_met.drop(['itr', 'type'],errors='ignore').iloc[0].to_dict()
                d_tr = {f"train_{k}":v for k,v in dt_tr.items()}

                dt_te = te_met.drop(['itr', 'type'],errors='ignore').iloc[0].to_dict()
                d_te = {f"test_{k}":v for k,v in dt_te.items()}
                wandb_run.log({**d_tr, **d_te})

            tr_met['type'] = 'train'
            te_met['type'] = 'test'


            metrics.append(pd.concat([tr_met, te_met]))
            
            
            
        sweep.fit(
            X_train,
            y_train,
            sensitive_features=A_train,
            callback=met_call_back if do_callback else None,
        )
        if wandb_run is not None:
            met_call_back(sweep.best_predictor, lbd=None, best=True)
        if len(metrics) > 0:
            eps_metrics.append(pd.concat(metrics))
            
    return pd.concat(eps_metrics).reset_index() if len(eps_metrics) > 0 else None

  




  