import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle
import os
import utils
from nilearn.decoding import SpaceNetRegressor
import nibabel as nib
from nilearn.decoding import SpaceNetClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_curve, auc, recall_score, precision_score
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize
import argparse 
import h5py
import pickle
import pandas as pd
import seaborn as sns

DEBUG = False

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.path.append('/scratch/ne2213/projects/tmp_packages')
sys.path.append('/scratch/ne2213/projects/tmp_packages/')
sys.path.append('/scratch/cl6707/Projects/neuro_interp/data/NSD/nsd')

# Convert to argparse
parser = argparse.ArgumentParser(description='SpaceNet Classification')
parser.add_argument('--datapath', type=str, default="/scratch/cl6707/Shared_Datasets/NSD_MindEye", help='path to NSD_MindEye')
parser.add_argument('--nsd_root', type=str, default="/scratch/cl6707/Projects/neuro_interp/data/NSD/", help='path to nsd')
parser.add_argument('--stim_root', type=str, default="/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata_stimuli/stimuli/nsd/", help='path to stimuli')
parser.add_argument('--beta_root', type=str, default="/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata_betas/ppdata/", help='path to betas')
parser.add_argument('--mask_root', type=str, default="/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/", help='path to masks')
parser.add_argument('--subj', type=int, default=1, help='subject number')
parser.add_argument('--load_mask', type=int, default=0, help='load mask')
parser.add_argument('--mask_id', type=int, default=0, help='mask id')
parser.add_argument('--output_dir', type=str, default="/scratch/cl6707/Projects/neuro_interp/Neural_Interpretation/notebooks/output/brain_decoding/", help='output directory')
parser.add_argument('--mask_path',type=str, default="/scratch/cl6707/Projects/neuro_interp/Neural_Interpretation/notebooks/output/brain_decoding/masks.h5", help='path to masks')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

seed=42
utils.seed_everything(seed=seed)

# voxel_roi_full  = load_mask_from_nii(args.mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz"%subj)

if args.subj == 1:
    num_voxels = 15724
elif args.subj == 2:
    num_voxels = 14278
elif args.subj == 3:
    num_voxels = 15226
elif args.subj == 4:
    num_voxels = 13153
elif args.subj == 5:
    num_voxels = 13039
elif args.subj == 6:
    num_voxels = 17907
elif args.subj == 7:
    num_voxels = 12682
elif args.subj == 8:
    num_voxels = 14386
    
    
print('Pulling NSD webdataset data...')
# Multi-GPU config #
# from accelerate import Accelerator
# accelerator = Accelerator(split_batches=False,mixed_precision='fp16')  
# print("PID of this process =",os.getpid())
# print = accelerator.print # only print if local_rank=0
# device = accelerator.device
# print("device:",device)
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices
# print(accelerator.state)
# local_rank = accelerator.state.local_process_index
# world_size = accelerator.state.num_processes
# distributed = not accelerator.state.distributed_type == 'NO'
# print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)



train_url = "{" + f"{args.datapath}/webdataset_avg_split/train/train_subj0{args.subj}_" + "{0..17}.tar," + f"{args.datapath}/webdataset_avg_split/val/val_subj0{args.subj}_0.tar" + "}"
val_url = f"{args.datapath}/webdataset_avg_split/test/test_subj0{args.subj}_" + "{0..1}.tar"
print(train_url,"\n",val_url)
meta_url = f"{args.datapath}/webdataset_avg_split/metadata_subj0{args.subj}.json"
num_train = 8559 + 300
num_val = 982
batch_size = 1
print('Prepping train and validation dataloaders...')

train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    batch_size,'images',
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=num_train,
    num_val=num_val,
    val_batch_size=1,
    cache_dir=args.datapath, #"/tmp/wds-cache",
    seed=seed,
    voxels_key='nsdgeneral.npy', # 'nsdgeneral.npy' (1d), 'wholebrain_3d.npy'(3d)
    to_tuple=["voxels", "images", "coco","trial"],
    local_rank=local_rank,
    # world_size=world_size,
)


nsd_mindroot = '/scratch/cl6707/Shared_Datasets/NSD_MindEye'

things = np.load(nsd_mindroot + '/subj%02d_things.npy'%args.subj,allow_pickle=True)
things_all = np.concatenate(things, axis=0)
unique_things = np.unique(things_all)
things_val_mapping = {k:i for i,k in enumerate(unique_things)}
val_things_mapping = {i:k for i,k in enumerate(unique_things)}


def plot_roc_curve(y_true, y_pred, unique_things, title,save_path = None):
    y_true = label_binarize(y_true, classes=np.arange(0, len(unique_things)))
    y_pred = label_binarize(y_pred, classes=np.arange(0, len(unique_things)))
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create a Plotly figure
    fig = go.Figure()

    # Add ROC curve traces for each class
    for i in range(n_classes):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines', name=f'{unique_things[i]} (AUC = {roc_auc[i]:.2f})'))

    # Add a diagonal line for a no-skill classifier
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No Skill', line=dict(dash='dash')))

    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title='Classes',
        # legend=dict(y=0.5, x=0.85),
        # margin=dict(l=40, r=40, t=40, b=40),
        width=800,height=800
    )
    if save_path is not None:
        fig.write_image(save_path)
    # Show the figure
    # fig.show()

def plot_confusion_matrix(y_true, y_pred, labels, title,save_path = None):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    # plt.show()   

def reconstruct_volume_corrected(vol_shape, binary_mask, data_vol, order='C'):
    
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    
    idx_mask = np.where(binary_mask)[0]
    
    view_vol[idx_mask] = data_vol
    return np.nan_to_num(view_vol.reshape(vol_shape, order=order))

nsdgeneral_affine = nib.load('/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz').affine
nsdgeneral =  nib.load('/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz')
nsdgeneral_roi_mask = (nsdgeneral.get_fdata()==1).astype(np.float32)
nsdgeneral = nib.Nifti1Image(nsdgeneral_roi_mask, nsdgeneral_affine)
anat_img = '/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/T1_to_func1pt8mm.nii.gz'



if __name__ == '__main__':
    print('LOADING DATA...')
    # LOAD ALL THE DATA
    X_train_all = []
    y_train_all = []
    X_val_all = []
    y_val_all = []
    # get all the data:

    train_iter = iter(train_dl)
    test_iter = iter(val_dl)
    if DEBUG:
        num_train = 100
        num_val = 100
    for i in tqdm(range(num_train)):
        voxels, _, _, trial = next(train_iter)
        voxels = voxels[0].cpu().numpy().mean(axis=0)
        # convert voxels to 3d volume
        voxels = reconstruct_volume_corrected(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(), voxels)
        nii_voxels = nib.Nifti1Image(voxels, nsdgeneral_affine)
        labels = things[trial]
        # for label in labels:
        #     X_train_all.append(nii_voxels)
        #     y_train_all.append(things_val_mapping[label])
        if len(labels)>=1 and len(labels)<=3:
            X_train_all.append(nii_voxels)
            y_train_all.append(things_val_mapping[labels[0]])

    for i in tqdm(range(num_val)):
        voxels, _, _, trial = next(test_iter)
        voxels = voxels[0].cpu().numpy().mean(axis=0)
        # convert voxels to 3d volume
        voxels = reconstruct_volume_corrected(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(), voxels)
        nii_voxels = nib.Nifti1Image(voxels, nsdgeneral_affine)
        labels = things[trial]
        # for label in labels:
        #     X_val_all.append(nii_voxels)
        #     y_val_all.append(things_val_mapping[label])
        if len(labels)>=1 and len(labels)<=3:
            X_val_all.append(nii_voxels)
            y_val_all.append(things_val_mapping[labels[0]])
            
    print('TRAINING MODEL...')
    # Loading Mask
    masks = {}
    if args.load_mask:
        print('Loading mask from:',args.mask_path)
        masks = {}
        with h5py.File(args.mask_path, 'r') as hf:
            for key in hf.keys():
                masks[key] = hf[key][:]
        mask = nib.Nifti1Image(masks[list(masks.keys())[args.mask_id]],nsdgeneral_affine)
        print('num voxels:',mask.get_fdata().sum())
    else:
        mask = nsdgeneral
        print('Using nsdgeneral mask')
        print('num voxels:',mask.get_fdata().sum())
        
    model = SpaceNetClassifier(
        mask=mask,
        memory = 'nilearn_cache',
        penalty='graph-net',
        screening_percentile=5.0,
        memory_level=2,
        standardize="zscore_sample",
     )
    
    print('Fitting model...')
    model.fit(X_train_all, y_train_all)
        
    # Save model
    print('Saving model...')
    model_name = f'SpaceNet_Classifier_subj{args.subj}_mask_{list(masks.keys())[args.mask_id]}_{args.mask_id}.pkl'

    pickle.dump(model, open(os.path.join(args.output_dir,model_name), 'wb'))
    
    print('EVALUATING MODEL...')
    print('Predicting...')
    y_pred_test = model.predict(X_val_all)
    y_pred_train = model.predict(X_train_all)
    
    print('Evaluating...')
    # save accuracy scores, f1 scores,  recall scores, precision scores, confusion matrix, roc curve
    accuracy_test = accuracy_score(y_val_all, y_pred_test)
    accuracy_train = accuracy_score(y_train_all, y_pred_train)
    print('Accuracy test:', accuracy_test)
    print('Accuracy train:', accuracy_train)
    f1_test = f1_score(y_val_all, y_pred_test, average='weighted')
    f1_train = f1_score(y_train_all, y_pred_train, average='weighted')
    print('F1 test:', f1_test)
    print('F1 train:', f1_train)
    recall_test = recall_score(y_val_all, y_pred_test, average='weighted')
    recall_train = recall_score(y_train_all, y_pred_train, average='weighted')
    print('Recall test:', recall_test)
    print('Recall train:', recall_train)
    precision_test = precision_score(y_val_all, y_pred_test, average='weighted')
    precision_train = precision_score(y_train_all, y_pred_train, average='weighted')
    print('Precision test:', precision_test)
    print('Precision train:', precision_train)
    df_metric = pd.DataFrame({'accuracy':[accuracy_test, accuracy_train], 'f1':[f1_test, f1_train], 'recall':[recall_test, recall_train], 'precision':[precision_test, precision_train]}, index=['test', 'train'])
    df_metric.to_csv(os.path.join(args.output_dir,f'metrics_{model_name}.csv'))
    
    plot_confusion_matrix(y_val_all, y_pred_test, unique_things, title=f'Confusion Matrix Test Accuracy: {accuracy_test:.2f}', save_path=os.path.join(args.output_dir,f'confusion_matrix_test_{model_name}.png'))
    plot_confusion_matrix(y_train_all, y_pred_train, unique_things, title=f'Confusion Matrix Train Accuracy: {accuracy_train:.2f}', save_path=os.path.join(args.output_dir,f'confusion_matrix_train_{model_name}.png'))
    plot_roc_curve(y_val_all, y_pred_test, unique_things, title=f'ROC Curve Test Accuracy: {accuracy_test:.2f}', save_path=os.path.join(args.output_dir,f'roc_curve_test_{model_name}.png'))
    plot_roc_curve(y_train_all, y_pred_train, unique_things, title=f'ROC Curve Train Accuracy: {accuracy_train:.2f}', save_path=os.path.join(args.output_dir,f'roc_curve_train_{model_name}.png'))
    
    print('Done!')
    
    
    
        
        


    