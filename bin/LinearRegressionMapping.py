import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse
# from src.file_utility import load_mask_from_nii, view_data
import nibabel as nib
from scipy.stats import pearsonr
import tensorflow_hub as hub

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

import utils
seed=42
utils.seed_everything(seed=seed)




data_path = "/scratch/cl6707/Shared_Datasets/NSD_MindEye"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye')
parser.add_argument('--nsd_root', type=str, default='/scratch/cl6707/Projects/neuro_interp/data/NSD/nsd')
parser.add_argument('--stim_root', type=str, default='/scratch/cl6707/Projects/neuro_interp/data/NSD/nsd/nsddata_stimuli/stimuli/nsd')
parser.add_argument('--beta_root', type=str, default='/scratch/cl6707/Projects/neuro_interp/data/NSD/nsd/nsddata_betas/ppdata')
parser.add_argument('--mask_root', type=str, default='/scratch/cl6707/Projects/neuro_interp/data/NSD/nsd/nsddata/ppdata')
parser.add_argument('--nsd_mindroot', type=str, default='/scratch/cl6707/Shared_Datasets/NSD_MindEye')
parser.add_argument('--subj', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_train', type=int, default=8559 + 300)
parser.add_argument('--num_val', type=int, default=982)
parser.add_argument('--val_batch_size', type=int, default=300)
parser.add_argument('--cache_dir', type=str, default='/tmp/wds-cache')
parser.add_argument('--voxels_key', type=str, default='nsdgeneral.npy')
parser.add_argument('--to_tuple', type=list, default=["voxels", "images", "coco","trial"])
parser.add_argument('--text_embed', type=str, default='guse')
parser.add_argument('--voxel_occlustion', type=bool, default=False)

args = parser.parse_args()
data_path = args.data_path
nsd_root = args.nsd_root
stim_root = args.stim_root
beta_root = args.beta_root
mask_root = args.mask_root
nsd_mindroot = args.nsd_mindroot
subj = args.subj
seed = args.seed
batch_size = args.batch_size
num_workers = args.num_workers
num_train = args.num_train
num_val = args.num_val
val_batch_size = args.val_batch_size
cache_dir = args.cache_dir
voxels_key = args.voxels_key
to_tuple = args.to_tuple



guse = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

subj =1
# voxel_roi_full  = load_mask_from_nii(mask_root + "subj%02d/func1pt8mm/roi/prf-visualrois.nii.gz"%subj)

if subj == 1:
    num_voxels = 15724
elif subj == 2:
    num_voxels = 14278
elif subj == 3:
    num_voxels = 15226
elif subj == 4:
    num_voxels = 13153
elif subj == 5:
    num_voxels = 13039
elif subj == 6:
    num_voxels = 17907
elif subj == 7:
    num_voxels = 12682
elif subj == 8:
    num_voxels = 14386
print('Pulling NSD webdataset data...')
# Multi-GPU config #
from accelerate import Accelerator
accelerator = Accelerator(split_batches=False,mixed_precision='fp16')  
print("PID of this process =",os.getpid())
print = accelerator.print # only print if local_rank=0
device = accelerator.device
print("device:",device)
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)



train_url = "{" + f"{data_path}/webdataset_avg_split/train/train_subj0{subj}_" + "{0..17}.tar," + f"{data_path}/webdataset_avg_split/val/val_subj0{subj}_0.tar" + "}"
val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
print(train_url,"\n",val_url)
meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982
batch_size = 512
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
    val_batch_size=300,
    cache_dir=data_path, #"/tmp/wds-cache",
    seed=seed,
    voxels_key='nsdgeneral.npy', # 'nsdgeneral.npy' (1d), 'wholebrain_3d.npy'(3d)
    to_tuple=["voxels", "images", "coco","trial"],
    local_rank=local_rank,
    world_size=world_size,
)
annotation_all = np.load(nsd_mindroot + '/subj%02d_annot.npy'%subj,allow_pickle=True)
nsdgeneral_affine = nib.load('/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz').affine
nsdgeneral_roi_mask = nib.load('/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata()==1
anat_img = '/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/T1_to_func1pt8mm.nii.gz'

if args.text_embed == 'guse':
    guse = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

elif args.text_embed == 'gpt':
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
    gpt = GPT2Model.from_pretrained('gpt2').to(device)
    gpt = gpt.eval()
else:
    raise NotImplementedError


def reconstruct_volume_corrected(vol_shape, binary_mask, data_vol, order='C'):
    
    view_vol = np.ones(np.prod(vol_shape), dtype=np.float32) * np.nan
    
    idx_mask = np.where(binary_mask)[0]
    
    view_vol[idx_mask] = data_vol
    return view_vol.reshape(vol_shape, order=order)

@torch.no_grad()
def get_guse(annotation,mode='max'):
    B = annotation.shape[0]
    annot_embed_all = np.zeros((B,512))
    if mode == 'max':
        for b in range(B):
            annot_embed = guse(annotation[b][0]).cpu().numpy()
            corr_mat = np.corrcoef(annot_embed)
            max_index = np.argmax(corr_mat.mean(axis=0),axis=0)
            annot_embed_all[b] = annot_embed[max_index]
    elif mode == 'mean':
        for b in range(B):
            annot_embed = guse(annotation[b][0]).cpu().numpy()
            annot_embed_all[b] = annot_embed.mean(axis=0)
    else:
        raise NotImplementedError
    
    return torch.tensor(annot_embed_all).to(device).float()

    
@torch.no_grad()
def get_gpt(annotation):
    B = annotation.shape[0]
    annot_embed_all = np.zeros((B,768))
    for b in range(B):
        encoded_input = tokenizer_gpt(annotation[b][0], padding=True, truncation=True,return_tensors='pt').to(device)
        output = gpt(**encoded_input)
        annot_embed = output.last_hidden_state.mean(dim=1).cpu().numpy()
        corr_mat = np.corrcoef(annot_embed)
        max_index = np.argmax(corr_mat.mean(axis=0),axis=0)
        annot_embed_all[b] = annot_embed[max_index]
    
    return torch.tensor(annot_embed_all).to(device).float()

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, alpha, l1_ratio, graph_structure):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.graph_structure = graph_structure

    def forward(self, x):
        return self.linear(x)

    def tv_l1_regularization(self):
        weights = self.linear.weight
        grad_x = torch.abs(weights[:, 1:] - weights[:, :-1])
        grad_y = torch.abs(weights[1:, :] - weights[:-1, :])
        tv_norm = torch.sum(grad_x) + torch.sum(grad_y)
        l1_norm = torch.sum(torch.abs(weights))
        return self.l1_ratio * tv_norm + (1 - self.l1_ratio) * l1_norm
        
model = LinearModel(num_voxels, 512,1,0.5, None)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loss_dict_train = {'tv_l1':[],
                'mse':[]
                }
tv_weight = 1e-3
pcc_list_train = []

epoch_num = 10
for epoch in tqdm(range(epoch_num)):
    for voxels, images, coco, trial in train_dl:
        pcc_all = []
        voxels = voxels.float().to(device)
        voxels = voxels.mean(axis=1)
        guse_embed = get_guse(annotation_all[trial])
        optimizer.zero_grad()
        outputs = model(voxels)
        tv_l1 = model.tv_l1_regularization()
        mse = criterion(outputs, guse_embed)
        loss_dict_train['tv_l1'].append(tv_l1.item()*tv_weight)
        loss_dict_train['mse'].append(mse.item())
        loss = mse + tv_weight*tv_l1
        loss.backward() 
        optimizer.step()
        # calculte PCC between outputs and guse_embed
        pcc = [pearsonr(outputs[i].detach().cpu().numpy(), guse_embed[i].detach().cpu().numpy(),)[0] for i in range(len(outputs))]
        pcc_all += pcc
    pcc_list_train.append(np.mean(pcc_all))
model = LinearModel(num_voxels, 512,1,0.5, None)
model.load_state_dict(torch.load('./output/subj%02d_linear_model.pt'%subj))
model.to(device)
loss_dict_val = {'tv_l1':[],
                'mse':[]
                }
pcc_list_val = []

model.eval()
for voxels, images, coco, trial in val_dl:
    voxels = voxels.float().to(device)
    voxels = voxels.mean(axis=1)
    guse_embed = get_guse(annotation_all[trial])
    outputs = model(voxels)
    # tv_l1 = model.tv_l1_regularization()
    # mse = criterion(outputs, guse_embed)
    # loss_dict_val['tv_l1'].append(tv_l1.item())
    # loss_dict_val['mse'].append(mse.item())
    # loss = mse + tv_weight*tv_l1
    # calculte PCC between outputs and guse_embed
    pcc = [pearsonr(outputs[i].detach().cpu().numpy(), guse_embed[i].detach().cpu().numpy(),)[0] for i in range(len(outputs))]
    pcc_list_val += pcc

print('Val PCC:',np.mean(pcc_list_val))
# save model
# torch.save(model.state_dict(), './output/subj%02d_linear_model.pt'%subj)
weights = model.linear.weight.detach().cpu().numpy()
weights = weights.mean(axis=0)
voxel_weight = weights
reconstructed_weight = reconstruct_volume_corrected(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(), voxel_weight)
reconstructed_weight = np.nan_to_num(reconstructed_weight)
ni_img = nib.Nifti1Image(reconstructed_weight,affine=nsdgeneral_affine)
from nilearn.plotting import plot_stat_map
plot_stat_map(ni_img,
                bg_img=anat_img,
                    cmap='hot')
epoch_num = 10
tv_weight = 1e-3
# weights for each voxel
weights = model.linear.weight.detach().cpu().numpy()
weights = weights.mean(axis=0)
voxel_weight = weights
# get top 10 voxels by absolute value
top_n = 5
top_voxel = np.argsort(np.abs(voxel_weight))[::-1][:top_n]
model_voxels = {voxel_pos:LinearModel(num_voxels, 512,1,0.5, None) for voxel_pos in top_voxel}
optimizer_voxels = {voxel_pos:torch.optim.Adam(model_voxels[voxel_pos].parameters(), lr=1e-3) for voxel_pos in top_voxel}
criterion = nn.MSELoss()
# for each voxel in the top10, mask out it and train the model again, see if the performance drops
pcc_voxel_dict = {}

for voxel_pos in tqdm(top_voxel):
    model_voxels[voxel_pos].to(device)
    for epoch in tqdm(range(epoch_num)):
        for voxels, images, coco, trial in train_dl:
            pcc_all = []
            voxels = voxels.float().to(device)
            voxels = voxels.mean(axis=1)
            guse_embed = get_guse(annotation_all[trial])
            optimizer_voxels[voxel_pos].zero_grad()
            outputs = model_voxels[voxel_pos](voxels)
            tv_l1 = model_voxels[voxel_pos].tv_l1_regularization()
            mse = criterion(outputs, guse_embed)
            # loss_dict_train['tv_l1'].append(tv_l1.item()*tv_weight)
            # loss_dict_train['mse'].append(mse.item())
            loss = mse + tv_weight*tv_l1
            loss.backward() 
            optimizer_voxels[voxel_pos].step()
            # calculte PCC between outputs and guse_embed
        #     pcc = [pearsonr(outputs[i].detach().cpu().numpy(), guse_embed[i].detach().cpu().numpy(),)[0] for i in range(len(outputs))]
        #     pcc_all += pcc
        # pcc_list_train.append(np.mean(pcc_all))
    
    # loss_dict_val = {'tv_l1':[],
    #                 'mse':[]
    #                 }
    pcc_list_val = []

    model_voxels[voxel_pos].eval()
    for voxels, images, coco, trial in val_dl:
        voxels = voxels.float().to(device)
        voxels = voxels.mean(axis=1)
        voxels[:,voxel_pos] = 0
        guse_embed = get_guse(annotation_all[trial])
        outputs = model_voxels[voxel_pos](voxels)
        tv_l1 = model_voxels[voxel_pos].tv_l1_regularization()
        mse = criterion(outputs, guse_embed)
        # loss_dict_val['tv_l1'].append(tv_l1.item())
        # loss_dict_val['mse'].append(mse.item())
        loss = mse + tv_weight*tv_l1
        # calculte PCC between outputs and guse_embed
        pcc = [pearsonr(outputs[i].detach().cpu().numpy(), guse_embed[i].detach().cpu().numpy(),)[0] for i in range(len(outputs))]
        pcc_list_val += pcc

    print('Val PCC:',np.mean(pcc_list_val))
    pcc_voxel_dict[voxel_pos] = np.mean(pcc_list_val)
    # save model
    torch.save(model_voxels[voxel_pos].state_dict(), './output/voxel_occlusion/subj%02d_linear_model_voxel%02d.pt'%(subj,voxel_pos))

if __name__ == "__main__":
    pass