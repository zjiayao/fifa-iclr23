from pathlib import Path
import numpy as np
import pandas as pd
import skimage.draw as skd
import scipy.ndimage as simg
import torch
import logging

logger = logging.getLogger(__name__)

class PseudoData:
    def __init__(self, torch_dataset, return_tensor='pt'):
        self.dataset = torch_dataset
        self.keys = {
            'float': False,
            'device': 'cpu',
            'view_args': [],
            'view_kwargs': {}
        }
        self.transforms = []
        self.return_tensor = return_tensor
        
        self.return_handle = np.stack
        self._set_return_tensor_handle(return_tensor)
    def _set_return_tensor_handle(self, return_tensor):
        if isinstance(return_tensor, str):
            if return_tensor.lower() == 'pt':
                self.return_handle = torch.stack
    @property
    def return_pt(self):
        return (self.return_tensor is None) or (self.return_tensor=='pt')
    @property
    def attr_names(self):
        return self.dataset.attr_names
    def float(self, val=True):
        self.keys['float'] = val
        return self
    def to(self, device):
        self.keys['device'] = device
        return self
    def add_transform(self, tran):
        self.transforms.append(tran)
        return self
    def clear_transform(self):
        self.transforms = []
        return self
    def view(self, *args, **kwargs):
        self.keys['view_args'] = args
        self.keys['view_kwargs'] = kwargs
        
    def _apply_trans(self, smp):
        if self.keys['float']:
            smp = smp.float()
        if self.return_pt:
            smp = smp.to(self.keys['device'])
        if len(self.keys['view_args']) > 0 or len(self.keys['view_kwargs']) > 0:
            smp = smp.view(*self.keys['view_args'], **self.keys['view_kwargs'])
        for tran in self.transforms:
            smp = tran(smp)
        
        return smp
        
    def __len__(self):
        return len(self.dataset)
    
    @property
    def shape(self):
        return np.array([len(self.dataset)])
        
    def __getitem__(self, key):
        if isinstance(key, slice) :
            return self[list(range(*key.indices(len(self))))]
        elif not hasattr(key, '__iter__'):
            x = self.dataset[key][0]
            if not self.return_pt:
                x = self.return_handle(x)
            return self._apply_trans(x)
        
        return self.return_handle([
            self._apply_trans(self.dataset[k][0])
            for k in key
        ])


####
# CIFAR10
####
def load_cifar10(path, to_one_ch=False, sep_trans=True):
    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    if to_one_ch:
        transform = transforms.Compose( [transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
    if sep_trans:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transform
        transform_test = transform

    return [torchvision.datasets.CIFAR10(root=str(path),
        train=True, download=True, transform=transform_train), 
            torchvision.datasets.CIFAR10(root=str(path),
        train=False, download=True, transform=transform_test)]

def cifar_unpack(dt, clss=None):
    xs = torch.stack([xx[0] for xx in dt])
    ys = np.array([xx[1] for xx in dt])
    if clss is not None:
        sel = np.zeros(ys.shape)
        for cr in clss:
            sel |= (sel == cr)
        xs = xs[sel]
        ys = ys[sel]
    return xs, ys

def get_cifar10(cifar10_data):
    cifar10_tr = list(cifar10_data[0])
    cifar10_va = list(cifar10_data[1])
    cifar10_xtr, cifar10_ytr = cifar_unpack(cifar10_tr)
    cifar10_xva, cifar10_yva = cifar_unpack(cifar10_va)
    return [cifar10_xtr, cifar10_ytr, cifar10_xva, cifar10_yva]

def sample_imbalance_data(data, sample_map, seed=None):
    """
    sample imbalanced dataset
    
    data = [Xtr, Ytr, Xte, Yte]
    sample_map: dict{class : n_tr, n_val}
    seed (optional): a random seed for numpy
    """
    if seed is not None:
        np.random.seed(hash(seed)%(2*32-1))

    clss = [c for c in sample_map]
    # sample sample_map[c][0] training samples for class c
    Xtr = torch.vstack([ data[0][data[1]==c][
            np.random.choice(np.sum(data[1]==c), sample_map[c][0])
        ] for c in clss])
    
    # sample sample_map[c][1] testing samples for class c
    Xte = torch.vstack([ data[2][data[3]==c][
            np.random.choice(np.sum(data[3]==c), sample_map[c][1])
        ] for c in clss])
        
    Ytr = torch.Tensor(np.hstack([[c] * sample_map[c][0] for c in clss]))
    
    Yte = torch.Tensor(np.hstack([[c] * sample_map[c][1] for c in clss]))
    
    return [Xtr, Ytr, Xte, Yte]


#### CelebA Data
def get_transform_celebA(train, target_resolution=(224,224), augment_data=False):
    import torchvision
    import torchvision.transforms as transforms
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if target_resolution is None:
        target_resolution = (orig_w, orig_h)

    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform

def load_celeba_data(path, split='train', target_type='attr',
                     download=False, return_tensor='pt', **kwargs):
    import torchvision
    import torchvision.transforms as transforms
    data = torchvision.datasets.CelebA(root=path,
        split=split, target_type=target_type,download=download,
        transform=get_transform_celebA(split=='train'))
    X = PseudoData(data, return_tensor=return_tensor, **kwargs)
    y = data.attr
    return [X, y]

def generate_celeba_groups(y, attr_list, label='Blond_Hair', attr='Male'):
    g1 = np.where(np.array(attr_list)==label)[0][0]
    g2 = np.where(np.array(attr_list)==attr)[0][0]
    y_re = torch.zeros(y.shape[0])
    y_re[(y[:,g1]==0) & (y[:,g2]==0)] = 1 # non-blond, female
    y_re[(y[:,g1]==1) & (y[:,g2]==0)] = 2 # blond, female
    y_re[(y[:,g1]==0) & (y[:,g2]==1)] = 3 # non-blond, male
    y_re[(y[:,g1]==1) & (y[:,g2]==1)] = 4 # blond, male
    assert (y_re != 0).all()
    return y_re - 1

def load_celeba_trval(path,download=False, return_tensor='pt', **kwargs):
    return load_celeba_data(path, split='train', target_type='attr', 
                            download=False, return_tensor=return_tensor, **kwargs) +\
        load_celeba_data(path, split='test', target_type='attr', download=False,
                        return_tensor=return_tensor, **kwargs)




def generate_celeba_sensitive(y, attr_list, label='Blond_Hair', attr='Male', label_name='blond', attr_name='gender'):
    logger.debug(f"generating dataset with label {label} and attribute {attr}")
    g1 = np.where(np.array(attr_list)==label)[0][0]# pd.Series, name='Blond')
    g2 = np.where(np.array(attr_list)==attr)[0][0]# pd.Series(, name='Male')
    label, A = np.zeros(y.shape[0]), np.zeros(y.shape[0], dtype='O')
    label[y[:, g1]==1] = 1
    label = pd.Series(data=label, name=label_name)
    A[y[:,g2]==0] = 'female'
    A[y[:,g2]==1] = 'male'
    A = pd.Series(data=A, name=attr_name)
    return label, A

def load_celeba_sensitive(path, group_kwargs={}, **kwargs):
    if not isinstance(path, Path):
        path = Path(path)
    
    CelebA_raw = load_celeba_trval(path, **kwargs)
    return [
        CelebA_raw[0],
        *generate_celeba_sensitive(CelebA_raw[1],CelebA_raw[0].attr_names, **group_kwargs),
        CelebA_raw[2],
        *generate_celeba_sensitive(CelebA_raw[3],CelebA_raw[2].attr_names, **group_kwargs),
    ]



# bank data
def load_bank_data(path,download=False):
    df = pd.read_csv(path)
    
    df_tr = df[df.train_val=='train'].reset_index(drop=True)
    df_val = df[df.train_val=='val'].reset_index(drop=True)
    

    
    y_tr = torch.Tensor(df_tr['y_yes'].values).long()
    y_val = torch.Tensor(df_val['y_yes'].values).long()
    
    X_tr = torch.Tensor(df_tr.drop(['y_yes','train_val'],axis=1).values)
    X_val = torch.Tensor(df_val.drop(['y_yes','train_val'],axis=1).values)
    return [X_tr, y_tr, X_val, y_val]

# msr data
def construct_data_size_map(y_train, A_train, y_test, A_test, label_name='label', attr_name='sex'):
    tr_df = pd.concat([y_train, A_train], axis=1)
    te_df = pd.concat([y_test, A_test], axis=1)
    logger.debug(str(tr_df.groupby([label_name, attr_name]).size().to_frame('n')))
    def size_map_handle(sample_type='train'):
        if sample_type == 'test':
            return te_df.groupby([label_name, attr_name]).size().to_frame('n')
        else:
            return tr_df.groupby([label_name, attr_name]).size().to_frame('n')
    return size_map_handle

def construct_celeba_size_map(y_train, y_test, attr_list, label='Blond_Hair', attr='Male', label_name='blond', attr_name='gender'):
    tr_df = pd.concat(generate_celeba_sensitive(
        y_train, attr_list,
        label=label, attr=attr, label_name=label_name, attr_name=attr_name
    ), axis=1)
    te_df = pd.concat(generate_celeba_sensitive(
        y_test, attr_list,
        label=label, attr=attr, label_name=label_name, attr_name=attr_name
    ), axis=1)
    logger.debug(str(tr_df.groupby([label_name, attr_name]).size().to_frame('n')))
    def size_map_handle(sample_type='train'):
        if sample_type == 'test':
            return te_df.groupby([label_name, attr_name]).size().to_frame('n')
        else:
            return tr_df.groupby([label_name, attr_name]).size().to_frame('n')
    return size_map_handle

def get_celeba_size_map(sample_type='train'):
    if sample_type == 'test':
        return pd.DataFrame([
            [0.0, 'male', 7535], [0.0, 'female', 9767], 
            [1.0, 'male', 180],[1.0, 'female', 2480]], 
          columns=['blond', 'gender', 'n']).set_index(['blond', 'gender'])
    else:
        return pd.DataFrame([
            [0.0, 'male', 66874], [0.0, 'female', 71629], 
            [1.0, 'male', 1387], [1.0, 'female', 22880]], 
          columns=['blond', 'gender', 'n']).set_index(['blond', 'gender'])
    
    
def construct_celeba_delta_map(d0m, d0f, d1m, d1f, label_name='blond'):
    return pd.DataFrame([[0.0, 'male', d0m],
              [0.0, 'female', d0f], 
              [1.0, 'male', d1m],
             [1.0, 'female', d1f]], 
        columns=[label_name, 'gender', 'delta']).set_index([label_name, 'gender'])
    
def construct_dataset_delta_map(d0m, d0f, d1m, d1f, 
                                label_name='label',
                               attr_name='sex'):
    return pd.DataFrame([[0.0, 'Male', d0m],
              [0.0, 'Female', d0f], 
              [1.0, 'Male', d1m],
             [1.0, 'Female', d1f]], 
        columns=[label_name, attr_name, 'delta']).set_index([label_name, attr_name])
    