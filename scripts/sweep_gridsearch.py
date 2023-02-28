"""
sweep_gridsearch.py

"""
from __future__ import print_function, absolute_import, division
import sys, os, json, requests, time, datetime, logging, argparse

from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model
import tqdm
import torch
import wandb


import src.example_datasets as egdata 
import src.utils as utils
import src.losses
import src.models
import src.metrics

from fairlearn.reductions import  DemographicParity, EqualizedOdds
## modify below
DATA_PATH = Path("path-to-data")
TMP_DIR = Path("path-to-tmp-dir")
MODEL_SAVE_PATH = Path("path-to-model-checkpoints")

TORCH_DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    
    configs = utils.ConfigArgs({
        'project': 'fairness_gen',
        'group': 'best:grid:eo',
        
        'dataset': 'celeba_224',
        'modelname': 'rn18',
        'normed_linear': 1,

        
        'save_path': MODEL_SAVE_PATH, 
        'model_load_path': None,
        'model_train_ly': None,
        
        'lr_scheduler': 'none',
        'imbalance_method': 'none',
        'loss_type': 'standard',
        
        'n_trials' : 1,
        'data_shape' : (32, 32),
        'n_classes' : 2,
        'n_iters' : 8000,
        'min_iters': 8000,
        'lr': 1e-4,
        'weight_decay': 5e-5,
        'batch_size' : 128, # GD if None
        'batch_balance' : False,
        'verbose' : True,
        'log_metrics': False,
        'log_metrics_every' : None,
        
        'n_hidden': -1,
        
        'stop_at_tr_err': 1e-10,
        
        'grid': "./configs/lambdas/celeba_rn18_best1.csv",
        'grid_size': 20,
        'grid_limit': 2,
        

        'constraint': 'eo', 
        
        'data_label': 'Blond_Hair',
        'data_label_name': 'blond',
        'pred_batch_size': 128, # used when making predictions / eval
        
        'use_drw': 1,
        'drw_ep': 4,
        'drw_beta': .9999,
        'drw_label_only': 0,
        
        'd0m': 0., # 66734
        'd0f': 0., # 71629
        'd1m': 0., # 1387
        'd1f': 0., # 22880
        'violation_eps': 0.01,
        
        
    })
    configs['group'] = configs['group'] + ':' + configs['dataset']
    
            
    wandb_run = wandb.init(project='test',
                       dir=TMP_DIR,
                       reinit=True,
                       config=configs,
              )

    wandb_run.config.update({'exp_id': wandb_run.id, 'exp_name': wandb_run.name})    
    configs = utils.ConfigArgs(wandb_run.config) # for sweep
    
    CelebA_raw = src.datasets.load_celeba_trval(DATA_PATH )
    CelebA_data = src.datasets.load_celeba_sensitive(DATA_PATH, group_kwargs={
        'label': configs.data_label,
        'label_name': configs.data_label_name,
    })
    
    
    if configs.model_load_path is not None:

        model_func = utils.load_model
        model_kwargs = {
        'path': configs.model_load_path,
        'train_layer': configs.model_train_ly,
        'replace_layer': {'fc': src.models.NormedLinear }
        }
    
    else:
        model_func = src.models.resnet18

        model_kwargs = {
            'input_shape': configs.data_shape,
            'n_classes': configs.n_classes,
            'normed_linear': configs.normed_linear == 1,
        }
    
    if 'loss_type' in configs and configs.loss_type.lower() == 'ldam':
        loss_func = lambda sample_weight, **kwargs: src.losses.LDAMLoss(sample_weight=sample_weight, **kwargs)

        logger.debug("using LDAM loss")
    elif 'loss_type' in configs and configs.loss_type.lower() == 'fdam':
        loss_func = lambda sample_weight, **kwargs: src.losses.FDAMLoss(sample_weight=sample_weight, **kwargs)

        logger.debug("using FDAM loss")
    else:
        loss_func = lambda sample_weight, **kwargs: src.losses.WeightedCrossEntropyLoss(sample_weight=sample_weight, **kwargs)
#         loss_kwargs = {}
        logger.debug("using CE loss")
    
    size_map_handle = src.datasets.construct_celeba_size_map(
           y_train=CelebA_raw[1], y_test=CelebA_raw[3],
           attr_list=CelebA_data[3].attr_names,
           label=configs.data_label, label_name=configs.data_label_name,
       )
    loss_kwargs = {
       'size_map':size_map_handle,
       'alpha':configs.alpha,
       's':configs.s,
       'max_m':configs.max_m,

       'delta_map':src.datasets.construct_celeba_delta_map(
           configs.d0m,configs.d0f,configs.d1m,configs.d1f
       ),
       'drw_label_only':configs.drw_label_only==1,
        'use_drw': configs.use_drw==1,
    }
    epss = [configs.violation_eps]

    
    print(f"grid search over: ", epss)
    for n_t in tqdm.tqdm(range(configs.n_trials)):
        log_metrics_func = lambda it : False
        if 'log_metrics_every' in configs and configs.log_metrics_every is not None:
            log_metrics_func = lambda it : (it > configs.n_iters-2) or (it % int(configs.log_metrics_every)==0)
        grid = None
        if 'grid' in configs and configs.grid is not None:
            grid = utils.load_saved_lambdas(configs.grid, data=configs.dataset)
            logger.debug(f"loaded grid from file of shape {grid.shape}")
            
        
        
        
        mlp_model = src.models.MLPWrapper(
           model_func=model_func,
           loss_func=lambda sample_weight, **kwargs: src.losses.WeightedCrossEntropyLoss(sample_weight=sample_weight, **kwargs),
           opt_func=lambda params, **kwargs : torch.optim.Adam(params, lr=configs.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=configs.weight_decay, **kwargs),
           model_kwargs=model_kwargs,
           metric_func=src.metrics.bin_classification_metrics,
           max_iter=configs.n_iters,
           min_iters=configs.min_iters,
           log_metrics=configs.log_metrics,
           log_metrics_func=log_metrics_func,
           verbose=configs.verbose,
           device=TORCH_DEV,
           stop_at_tr_err=configs.stop_at_tr_err,
           manual_init=False,
           pred_batch_size=configs.pred_batch_size,
           wandb_run=wandb_run if configs.log_metrics else None,

    )
        constraints_func = EqualizedOdds if 'eo' in configs.constraint.lower() else DemographicParity
        logger.debug(f"starting procedure, eps: {epss}, grids: {grid.shape}, constraint: {constraints_func.__name__}, log_metrics: {configs.log_metrics}")

            
        res_df = utils.fit_bin_gridsearch_eps(
            regressor=mlp_model,
            constraints_func=constraints_func,
            epss=epss,
            dataset=CelebA_data,
            grid=grid,
            grid_size=configs.grid_size,
            grid_limit=configs.grid_limit,
            wandb_run=None if configs.log_metrics else wandb_run,
            fit_kwargs={
                'batch_size': configs.batch_size,
                'imbalance_method': configs.imbalance_method,
                'y_orig': CelebA_data[1],
                'group': CelebA_data[2],
                'X_test': CelebA_data[3], 
                'y_test': CelebA_data[4],
                'group_test': CelebA_data[5],
                'drw_beta': configs.drw_beta,
                'drw_ep': configs.drw_ep,
                'size_map_handle': size_map_handle,
            },
            save_path=configs.save_path,
            do_callback=False,
            call_estimator_to_complete=False,
            size_map_handle=size_map_handle,
            
        )



