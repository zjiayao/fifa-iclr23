# Obtaining More Generalizable Fair Classifiers on Imbalanced Datasets

This repo contains the code to reproduce the experiments in the paper [*Obtaining More Generalizable Fair Classifiers on Imbalanced Datasets*](https://openreview.net/forum?id=zVrw4OH1Lch). 

```bib
@inproceedings{
	deng2023obtaining,
	title={Obtaining More Generalizable Fair Classifiers on Imbalanced Datasets},
	author={Zhun Deng and Jiayao Zhang and Linjun Zhang and Ting Ye and Yates Coley and Weijie J Su and James Zou},
	booktitle={International Conference on Learning Representations},
	year={2023},
url={https://openreview.net/forum?id=zVrw4OH1Lch}
}
```

## Reproducing Experiments

### (Optional) Sweeps

All experiments are done using ``wandb`` sweeps. The sweep configs are
stored in ``./sweep_configs/`` with self-explanatory filenames.

### Reproducing Figures and Tables

The results of those sweeps are stored in the ``csv`` files in ``./csv_data``.
Please see the notebook ``reproduction.ipynb`` for details.