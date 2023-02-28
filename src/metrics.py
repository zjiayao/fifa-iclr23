import torch
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as skm
import src.utils as utils
import logging
logger = logging.getLogger('metrics')
logger.setLevel(logging.DEBUG)

def bin_fair_metrics(y_pred, y_true, group):
    """
    y_pred: (n_sample * 2)
    """
    from fairlearn.metrics import (MetricFrame,
        selection_rate, demographic_parity_difference, demographic_parity_ratio,
        false_positive_rate, false_negative_rate,
        false_positive_rate_difference, false_negative_rate_difference,
        equalized_odds_difference)
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y_pred_lbs = np.argmax(y_pred, axis=1)
    y_pred = y_pred[:,1]
#     print(y_pred_lbs, np.unique(y_true))
    metrics_dict = {
        "overall_sel_rate": (  # Overall selection rate
            lambda x: selection_rate(y_true, x), True),
        "dp": ( # Demographic parity difference
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        "dpr": ( # Demographic parity ratio
            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        "bal_err_rate": ( # Overall balanced error rate
            lambda x: 1-balanced_accuracy_score(y_true, x), True),
#         "bal_err_diff": ( # Balanced error rate difference
#             lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        "fpr": ( # False positive rate difference
            lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        "fnr": ( # False negative rate difference
            lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),
        "eo": ( # Equalized odds difference
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
#         "auc_diff": ( # AUC difference
#             lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        try:
            df_dict[metric_name] = [metric_func(y_pred_lbs) if use_preds else metric_func(y_pred) ]
        except Exception as e:
            logger.debug(f"error computing metric {metric_name}: {e}")
            
    return pd.DataFrame.from_dict(df_dict, )






def bin_classification_metrics(y_pred, y_true, group=None, label=None, size_map=None, **kwargs):
    """
    y_pred: scores not 0/1 predictions
    """
    def acc(yp, yt):
        try:
            return np.mean(np.argmax(yp, axis=1)==yt)
        except:
            return np.nan

    def get_prf(yp, yt):
        try:
            yp = np.argmax(yp, axis=1)
            return skm.precision_recall_fscore_support(yt, yp, average='micro',zero_division=0)
        except:
            return np.nan,np.nan,np.nan,None

    def get_auc(yp, yt):
        try:
            return skm.roc_auc_score(yt, yp[:,1])
        except Exception as e:
            print(f"Error getting AUC: {e}")
            return None

    def get_grp_acc(res, yp, yt, size_map):
        accs = []
        for idx in size_map.index:
            # idx is like (0.0, 'male')

            lb = '_'.join([str(s) for s in idx])
            indices = utils.get_idx_from_idx(label, group, idx[0], idx[1])
            grp_acc = acc(yp[indices], yt[indices])
            accs.append(grp_acc)

            res['acc_'+lb] = grp_acc
        try:
            return np.nanmean(accs)
        except:
            return np.nan
    
    def get_lb_acc(res, yp, yt, size_map):
        label_classes = size_map.index.get_level_values(0).drop_duplicates().tolist()

        accs = []
        for lb_idx in label_classes:
            lb = f"cls_{int(lb_idx)}"
            indices = utils.get_idx_from_lb(label, lb_idx)
            lb_acc = acc(yp[indices], yt[indices])
            accs.append(lb_acc)
            res['acc_'+lb] = lb_acc
        try:
            return np.nanmean(accs)
        except:
            return np.nan
            

    res_dict = {}

    # total
    res_dict['acc'] = acc(y_pred, y_true)
    res_dict['auc'] = get_auc(y_pred, y_true)
    precision, recall, f1, _ = get_prf(y_pred, y_true)
    
    res_dict['precision_total'] = precision
    res_dict['recall_total'] = recall
    res_dict['f1_total'] = f1
    if size_map is not None:
        res_dict['subgrp_avg_acc'] = get_grp_acc(res_dict, y_pred, y_true, size_map)
        res_dict['cls_avg_acc'] = get_lb_acc(res_dict, y_pred, y_true, size_map)
        
    res_df = pd.DataFrame([res_dict]) 
    if group is not None:
        fair_df = bin_fair_metrics(y_true=y_true, y_pred=y_pred, group=group)
        res_df = pd.concat([res_df, fair_df],axis=1)
        

        
    
    return res_df