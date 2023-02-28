import torch
import numpy as np
import pandas as pd
import src.utils as utils
import logging
logger = logging.getLogger(__name__)

class FDAMLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self,
                 sample_weight,
                 size_map: pd.DataFrame,
                 alpha: float,
                 delta_map: pd.DataFrame,
                 sample_type: str='train',
                 max_m: float=0.5,  # tengyu's code
                 s: float=30,       # tengyu's code
                 beta: float=0.,    # for annealing
                 size_average=None, ignore_index: int = -100,
                 drw_label_only=False,
                 use_drw=True,
                 weight=None,       # not used
                 reduce=None) -> None:
        super(FDAMLoss, self).__init__(None, size_average, reduce, 'none')
        self.sample_weight = sample_weight
        self.ignore_index = ignore_index
        if callable(size_map):
            self.size_map = size_map(sample_type=sample_type)
        else:
            self.size_map = size_map
        self.alpha = alpha
        self.delta_map = delta_map
        self.max_m = max_m        # "C" in our notation
        # a multiplicative factor for the logtits that will be cancel out
        # used in tengyu's code with no comments
        # guess it's for increaseing stability
        self.s = s   
        self.weight = weight # per cls weight, not used
        self.beta = beta
        self.drw_label_only = drw_label_only
        self.use_drw = use_drw
        
#         if self.drw_label_only:
#             logger.debug("FDAM w/ 2-DRW")
        
    def set_beta(self, beta):
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor, *,
                label: pd.Series,
               group: pd.Series, **kwargs) -> torch.Tensor:
        # margins to be substracted
        margin_weights = torch.ones(len(input)).float()

        per_lb_ns = []
        per_lb_indices = []
        n_tildes = [] # for annealing
        deltas = []
        
        # for each of the 2x2 subgroup
        for idx in self.size_map.index:
            # idx is like (0.0, 'male')
            # for each label class
            
            # this is the effective class size for the i-th class
            # since we our looping over all subgroups
            # it's the same for different subgroups in the same label class
            n_i = utils.get_effectiven_n(self.size_map, idx[0], self.alpha) # n_i
            # 1 / n^{1/4}
            per_lb_ns.append(1. / np.sqrt(np.sqrt(n_i))) 
            
            # this selects the sample indices from this subgroup
            per_lb_indices.append(utils.get_idx_from_idx(label, group, idx[0], idx[1]))
            
            # subgroup size
            n_ia = utils.get_n_from_index(self.size_map, idx)
            
            # drw used in tengyu's code
            n_tildes.append( 1.-np.power(self.beta, n_ia) ) 
            
            # small deltas, delta_{i,a}
            deltas.append( utils.get_n_from_index(self.delta_map, idx) ) 
            
        # compute Deltas
        # normalization used in tengyu's code
        # here max_m is our C
        per_lb_ns = np.array(per_lb_ns) * (self.max_m / np.max(per_lb_ns))
        
        # for each of the 2x2 subgroup
        for lb_grp_i, lb_grp_idx in enumerate(per_lb_indices):
            # Delta_{i, a}
            # lb_grp_idx: sample indices that belong to subgroup {i, a}
            
            # Delta := C / \tilde{n_i}^{1/4} + delta_{i,a}
            margin_weights[lb_grp_idx] *= (per_lb_ns[lb_grp_i] + deltas[lb_grp_i])
        margin_weights = margin_weights.to(input.device)
        
        # input.shape: (n_samples, n_classes)
        # note that the margin is substracted from the i-th logit
        # for a sample from the i-th label class
        # the other logits are not touched
        # hence we are selecting the second dim of input by the label values
        input[[np.arange(len(input)), label.values.astype(int)]] -= margin_weights
        
        
        out = torch.nn.functional.cross_entropy(input*self.s, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='none',
        )
     
        # DRW
        if self.use_drw:
            an_weights = torch.ones(len(input)).float()
            if self.drw_label_only:
                # get per-label indices
                label_classes = self.size_map.index.get_level_values(0).drop_duplicates().tolist()
                per_lb_indices = []
                n_tildes = []
                for lb in label_classes:
                    n = np.sum( utils.get_ns_from_lb(self.size_map,lb) )
                    n_tildes.append(n)
                    per_lb_indices.append(utils.get_idx_from_lb(label, lb))

                an_wts_per_idx = (1.-self.beta)/np.array(n_tildes)
                an_wts_per_idx = an_wts_per_idx / np.sum(an_wts_per_idx) * len(label_classes)

                for lb_i, lb_idx in enumerate(per_lb_indices):
                    an_weights[lb_idx] *= an_wts_per_idx[lb_i]

            else:
                # get sample indices from each subgroup
                # same as "per_lb_indices" above
                per_lb_grp_indices = [utils.get_idx_from_idx(label, group, idx[0], idx[1])
                                      for idx in self.size_map.index
                                     ]
                # same procedure as in tengyu's code
                # this gives us the annealing weights used by DRW
                an_wts_per_idx = (1.-self.beta)/np.array(n_tildes)
                an_wts_per_idx = an_wts_per_idx / np.sum(an_wts_per_idx) * len(self.size_map.index)

                for lb_grp_i, lb_grp_idx in enumerate(per_lb_grp_indices):
                    an_weights[lb_grp_idx] *= an_wts_per_idx[lb_grp_i]

            an_weights = an_weights.to(input.device)

            # reweighting by annealing weights
            out = (out.T * an_weights).T / torch.sum(an_weights) # annealing
        
        # reweighting by sample_weights given by the fairlearn lib
        # used for weighted classification
        out = (out.T * self.sample_weight).T # weighted classification
        return torch.sum(out/torch.sum(self.sample_weight))

    
             
class LDAMLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, # for weighted classification
                 sample_weight,
                 size_map: pd.DataFrame,
                 alpha: float,
                 delta_map: pd.DataFrame,
                 sample_type: str='train',
                 max_m: float=0.5,  # tengyu's code
                 s: float=30,       # tengyu's code
                 beta: float=0.,    # for annealing
                 size_average=None, ignore_index: int = -100,
                 use_drw=True,
                 weight=None,
                 reduce=None) -> None:
        super(LDAMLoss, self).__init__(None, size_average, reduce, 'none')
        self.sample_weight = sample_weight
        self.ignore_index = ignore_index
        if callable(size_map):
            self.size_map = size_map(sample_type=sample_type)
        else:
            self.size_map = size_map
        self.alpha = alpha
        self.delta_map = delta_map
        self.max_m = max_m
        self.s = s
        self.weight = weight # per cls weight
        self.beta = beta
        self.use_drw = use_drw
        
    def set_beta(self, beta):
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor, *,
                label: pd.Series,
               group: pd.Series, **kwargs) -> torch.Tensor:
        
        margin_weights = torch.ones(len(input)).float()
        
        # per label class
        label_classes = self.size_map.index.get_level_values(0).drop_duplicates().tolist()
        per_lb_ns = []
        per_lb_indices = []
        n_tildes = []
        for lb in label_classes:
            n = np.sum( utils.get_ns_from_lb(self.size_map,lb) )
            n_tildes.append(n)
            per_lb_ns.append(1. / np.sqrt(np.sqrt(n)))
            per_lb_indices.append(utils.get_idx_from_lb(label, lb))
            
        # compute Deltas
        per_lb_ns = np.array(per_lb_ns) * (self.max_m / np.max(per_lb_ns))
        for lb_i, lb_idx in enumerate(per_lb_indices):
            margin_weights[lb_idx] *= per_lb_ns[lb_i]

        # now sample_weight is the same as ``batch_m`` in tengyu's code 
        margin_weights = margin_weights.to(input.device)
        
        # substract Delta
        input[[np.arange(len(input)), label.values.astype(int)]] -= margin_weights
        
        
        out = torch.nn.functional.cross_entropy(input*self.s, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='none',
        )

        if self.use_drw:
            # annealing weights
            # for each label class
            an_wts_per_idx = (1.-self.beta)/np.array(n_tildes)
            an_wts_per_idx = an_wts_per_idx / np.sum(an_wts_per_idx) * len(self.size_map.index.unique(level=0))
            an_weights = torch.ones(len(input)).float()
            for lb_i, lb_idx in enumerate(per_lb_indices):
                an_weights[lb_idx] *= an_wts_per_idx[lb_i]


            an_weights = an_weights.to(input.device)
            out = (out.T * an_weights).T / torch.sum(an_weights) # annealing

        out = (out.T * self.sample_weight).T # weighted classification
        return torch.sum(out/torch.sum(self.sample_weight))

    
    
    
class WeightedCrossEntropyLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, sample_weight, size_average=None, ignore_index: int = -100,
                 size_map: pd.DataFrame={},
                 use_drw=False,
                 drw_label_only=False,
                 beta=0.0,
                 reduce=None, **kwargs) -> None:
        super(WeightedCrossEntropyLoss, self).__init__(None, size_average, reduce, 'none')
        self.sample_weight = sample_weight
        self.ignore_index = ignore_index
        
        self.size_map = size_map
        self.use_drw = use_drw
        self.drw_label_only = drw_label_only
        self.beta = beta
        
    def set_beta(self, beta):
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor,*,
                label: pd.Series=None, group: pd.Series=None, **kwargs) -> torch.Tensor:
        
        out = torch.nn.functional.cross_entropy(input, target, weight=None,
                       ignore_index=self.ignore_index, reduction='none',
        )
        
        
        if self.use_drw:
            an_weights = torch.ones(len(input)).float()
            if self.drw_label_only:
                # get per-label indices
                label_classes = self.size_map.index.get_level_values(0).drop_duplicates().tolist()
                per_lb_indices = []
                n_tildes = []
                for lb in label_classes:
                    n = np.sum( utils.get_ns_from_lb(self.size_map,lb) )
                    n_tildes.append(n)
                    per_lb_indices.append(utils.get_idx_from_lb(label, lb))

                an_wts_per_idx = (1.-self.beta)/np.array(n_tildes)
                an_wts_per_idx = an_wts_per_idx / np.sum(an_wts_per_idx) * len(label_classes)

                for lb_i, lb_idx in enumerate(per_lb_indices):
                    an_weights[lb_idx] *= an_wts_per_idx[lb_i]

            else:
                n_tildes = [] # for annealing

                # for each of the 2x2 subgroup
                for idx in self.size_map.index:
                    # subgroup size
                    n_ia = utils.get_n_from_index(self.size_map, idx)
                    # drw used in tengyu's code
                    n_tildes.append( 1.-np.power(self.beta, n_ia) ) 
                
                per_lb_grp_indices = [utils.get_idx_from_idx(label, group, idx[0], idx[1])
                                      for idx in self.size_map.index
                                     ]

                an_wts_per_idx = (1.-self.beta)/np.array(n_tildes)
                an_wts_per_idx = an_wts_per_idx / np.sum(an_wts_per_idx) * len(self.size_map.index)

                for lb_grp_i, lb_grp_idx in enumerate(per_lb_grp_indices):
                    an_weights[lb_grp_idx] *= an_wts_per_idx[lb_grp_i]

            an_weights = an_weights.to(input.device)

            # reweighting by annealing weights
            out = (out.T * an_weights).T / torch.sum(an_weights) # annealing

        out = (out.T * self.sample_weight).T
        return torch.sum(out/torch.sum(self.sample_weight))
    


class LogitWeightedCrossEntropyLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, sample_weight, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(WeightedCrossEntropyLoss, self).__init__(None, size_average, reduce, reduction)
        self.sample_weight = sample_weight
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        input = (input.T * self.sample_weight).T
        return torch.nn.functional.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
    )
    
