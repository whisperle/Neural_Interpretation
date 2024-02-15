import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from nilearn.decoding import SpaceNetClassifier
from nilearn.decoding import SpaceNetRegressor 
import nibabel as nib
from scipy.stats import pearsonr
import argparse
import pickle
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--subj', type=int, default=1)
parser.add_argument('--n_PC', type=int, default=0,help='number range(50) for n-th Principal Component')
parser.add_argument('--output_dir', type=str, default='/scratch/cl6707/Projects/neuro_interp/output')
parser.add_argument('--feature', type=str, default='pca_feature',help='features to use for prediction')
args = parser.parse_args()
DEBUG = False

class NSD_PCA(Dataset):
    def __init__(self, subj=1, data_path = '/scratch/cl6707/Shared_Datasets/NSD_MindEye',mode='train'):
        assert mode in ['train','val'], 'mode must be either train or val'
        assert subj in [1,2,5,7], 'subj must be either 1,2,5,7'
        self.data_path = data_path
        self.mode = mode
        self.subj = subj
        meta_data = {}
        print(f'Loading {mode} data for subj {subj}...')
        with h5py.File(self.data_path+f'/{self.mode}_subj0{subj}_pca50.hdf5','r') as f:
            for key in f.keys():
                meta_data[key] = f[key][:]
        
        with h5py.File(f'/scratch/ne2213/projects/Nika-IVP/Features/subj_{subj}/Features{self.mode}_subj0{subj}_umap50.hdf5','r') as f:
            meta_data['umap_feature'] = f['umap_feature'][:]
            

        self.meta_data = meta_data
    def __len__(self):
        return self.meta_data['voxels'].shape[0]

    def __getitem__(self, idx):
        voxel = self.meta_data['voxels'][idx]
        image = self.meta_data['images'][idx]
        coco = self.meta_data['coco'][idx]
        trial = self.meta_data['trial'][idx]
        res_feature = self.meta_data['res_feature'][idx]
        pca_feature = self.meta_data['pca_feature'][idx]
        umap_feature_50 = self.meta_data['umap_feature'][idx]
        umap_feature_2 = self.meta_data['umap_feature_2'][idx]
        
        return_dict = {
            'voxel': voxel,
            'image': image,
            'coco': coco,
            'trial': trial,
            'res_feature': res_feature,
            'pca_feature': pca_feature,
            'umap_feature_50': umap_feature_50,
            'umap_feature_2': umap_feature_2
        }
        return return_dict
    
class NSD_PCA_Dataloader(DataLoader):
    def __init__(self, subj=1, data_path = '/scratch/cl6707/Shared_Datasets/NSD_MindEye',mode='train',batch_size=32,shuffle=True):
        self.dataset = NSD_PCA(subj=subj, data_path=data_path, mode=mode)
        super(NSD_PCA_Dataloader,self).__init__(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                generator=torch.Generator('cuda' if torch.cuda.is_available() else 'cpu'),
                                                drop_last=True if mode=='train' else False
                                                )

nsdgeneral_affine = nib.load(f'/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj0{args.subj}/func1pt8mm/roi/nsdgeneral.nii.gz').affine
nsdgeneral_roi_mask = nib.load(f'/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj0{args.subj}/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata()==1
anat_img = f'/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj0{args.subj}/func1pt8mm/T1_to_func1pt8mm.nii.gz'
def reconstruct_volume(vol_shape, binary_mask, data_vol, order='C'):
    
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    
    idx_mask = np.where(binary_mask)[0]
    
    view_vol[idx_mask] = data_vol
    return np.nan_to_num(view_vol.reshape(vol_shape, order=order))


if __name__ == '__main__':
    
    train_dataset =NSD_PCA_Dataloader(subj=args.subj,mode='train',batch_size=1)
    val_dataset = NSD_PCA_Dataloader(subj=args.subj,mode='val',batch_size=1)
    train_nii = []
    train_components = []
    trian_iter = iter(train_dataset)
    val_iter = iter(val_dataset)
    num_train = 100 if DEBUG else len(train_dataset)
    num_val = 100 if DEBUG else len(val_dataset)
    
    for i in tqdm(range(num_train)):
        batch = next (trian_iter)
        voxels = batch['voxel'].mean(axis=1)
        for voxel in voxels:
            voxel = reconstruct_volume(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(),voxel)
            train_nii.append(nib.Nifti1Image(voxel,nsdgeneral_affine))
        # train_nii.append(nib.Nifti1Image(reconstruct_volume_batch(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(),voxels),nsdgeneral_affine))
        train_components.append(batch[args.feature][:,args.n_PC])
    train_nii = nib.concat_images(train_nii)
    train_components = np.concatenate(train_components)

    val_nii = []
    val_components = []
    for i in tqdm(range(num_val)):
        batch = next (val_iter)
        voxels = batch['voxel'].mean(axis=1)
        for voxel in voxels:
            voxel = reconstruct_volume(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(),voxel)
            val_nii.append(nib.Nifti1Image(voxel,nsdgeneral_affine))
        # val_nii.append(nib.Nifti1Image(reconstruct_volume_batch(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(),voxels),nsdgeneral_affine))
        val_components.append(batch[args.feature][:,args.n_PC])
    val_nii = nib.concat_images(val_nii)
    val_components = np.concatenate(val_components)

    pca_Regressor = SpaceNetRegressor(                                              
                                        memory="nilearn_cache",
                                        penalty="graph-net",
                                        screening_percentile=5.0,
                                        memory_level=2,
                                        standardize="zscore_sample",
                                        n_jobs=2,
                                        ) 
    predict_components = {}
    CC_all = {}
    # for k in tqdm(range(50)):
    print(f'Fitting component {args.n_PC}...')
    pca_Regressor.fit(train_nii,train_components)
    predict_components = pca_Regressor.predict(val_nii)
    CC, pvalue= pearsonr(val_components,predict_components)
    # Save the coefficients from model
    coef_img = pca_Regressor.coef_img_
    coef_array = pca_Regressor.coef_
    # 
    results = {
        'CC':CC,
        'pvalue':pvalue,
        'coef_img':coef_img,
        'coef_array':coef_array
    }
    np.save(args.output_dir+f'/subj{args.subj}_{args.feature}_{args.n_PC}_results.npy',results)
    print('Done!')
