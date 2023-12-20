import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle
import os
import utils
import nibabel as nib
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,roc_curve, auc, recall_score, precision_score
import plotly.graph_objects as go
from sklearn.preprocessing import label_binarize
import argparse 
import h5py
import pickle
import pandas as pd
import tensorflow_hub as hub
import torch.nn as nn
from scipy.stats import pearsonr
import wandb
from nilearn.plotting import plot_stat_map

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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
parser.add_argument('--tvl1_weight', type=float, default=1e-3, help='tv l1 weight')
parser.add_argument('--epoch_num', type=int, default=10, help='epoch number')
parser.add_argument('--text_embed', type=str, default='guse', help='text embedding method')
parser.add_argument('--output_dir', type=str, default='./output', help='output directory')
parser.add_argument('--mask_path', type=str, default='/scratch/cl6707/Projects/neuro_interp/Neural_Interpretation/notebooks/output/brain_decoding/masks.h5', help='mask path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)
seed=42
utils.seed_everything(seed=seed)

DEBUG = True
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
batch_size = 32
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
    val_batch_size=32,
    cache_dir=args.datapath, #"/tmp/wds-cache",
    seed=seed,
    voxels_key='nsdgeneral.npy', # 'nsdgeneral.npy' (1d), 'wholebrain_3d.npy'(3d)
    to_tuple=["voxels", "images", "coco","trial"],
    local_rank=local_rank,
    # world_size=world_size,
)


nsd_mindroot = '/scratch/cl6707/Shared_Datasets/NSD_MindEye'

# things = np.load(nsd_mindroot + '/subj%02d_things.npy'%args.subj,allow_pickle=True)
# things_all = np.concatenate(things, axis=0)
# unique_things = np.unique(things_all)
# things_val_mapping = {k:i for i,k in enumerate(unique_things)}
# val_things_mapping = {i:k for i,k in enumerate(unique_things)}

annotation_all = np.load(nsd_mindroot + '/subj%02d_annot.npy'%args.subj,allow_pickle=True)
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
    B = len(annotation)
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
        
if __name__ == '__main__':
    print('SAVING TO:',args.output_dir)
    wandb.init(project="neuro_interp", 
               mode = 'disabled' if DEBUG else 'online',
               name=f"subj{args.subj}_linear_model_mask_{args.load_mask}_id{args.mask_id}_tvl1weight{args.tvl1_weight}_guse",
               dir=args.output_dir,)
    # Loading voxel mask 
    mask = None
    affine = None
    if args.load_mask:
        print('Loading mask from:',args.mask_path)
        masks = {}
        with h5py.File(args.mask_path, 'r') as hf:
            for key in hf.keys():
                masks[key] = hf[key][:]
        mask = masks[list(masks.keys())[args.mask_id]]
        nsdgeneral1d = nsdgeneral_roi_mask.flatten()
        mask = torch.tensor(mask.flatten()[nsdgeneral1d!=0]).to(device).float()
        print('num voxels:',mask.sum())
    else:
        mask = nsdgeneral_roi_mask
        mask = torch.tensor(mask[mask!=0].flatten()).to(device).float()
        print('Using nsdgeneral mask')

    
    model = LinearModel(num_voxels, 512,1,0.5, None)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loss_dict_train = {'tv_l1':[],
                    'mse':[]
                    }
    
    pcc_list_train = []

    epoch_num = args.epoch_num
    for epoch in tqdm(range(epoch_num)):
        for voxels, images, coco, trial in train_dl:
            pcc_all = []
            voxels = voxels.float().to(device)
            voxels = voxels.mean(axis=1)
            voxels = voxels*mask
            guse_embed = get_guse(annotation_all[trial])
            optimizer.zero_grad()
            outputs = model(voxels)
            tv_l1 = model.tv_l1_regularization()
            mse = criterion(outputs, guse_embed)
            loss_dict_train['tv_l1'].append(tv_l1.item()*args.tvl1_weight)
            loss_dict_train['mse'].append(mse.item())
            loss = mse + args.tvl1_weight*tv_l1
            wandb.log({"train_loss": loss.item(),
                       'train_mse':mse.item(),
                          'train_tv_l1':tv_l1.item()*args.tvl1_weight,
                       })
            loss.backward() 
            optimizer.step()
            # calculte PCC between outputs and guse_embed
            pcc = [pearsonr(outputs[i].detach().cpu().numpy(), guse_embed[i].detach().cpu().numpy(),)[0] for i in range(len(outputs))]
            pcc_all += pcc
        pcc_list_train.append(np.mean(pcc_all))
        wandb.log({"train_pcc": np.mean(pcc_all)})
        print('Train PCC:',np.mean(pcc_all))

        loss_dict_val = {'tv_l1':[],
                        'mse':[]
                        }
        pcc_list_val = []

        model.eval()
        for voxels, images, coco, trial in val_dl:
            voxels = voxels.float().to(device)
            voxels = voxels.mean(axis=1)
            voxels = voxels*mask
            guse_embed = get_guse(annotation_all[trial])
            outputs = model(voxels)
            tv_l1 = model.tv_l1_regularization()
            mse = criterion(outputs, guse_embed)
            loss_dict_val['tv_l1'].append(tv_l1.item())
            loss_dict_val['mse'].append(mse.item())
            loss = mse + args.tvl1_weight*tv_l1
            wandb.log({"val_loss": loss.item(),
                    'val_mse':mse.item(),
                        'val_tv_l1':tv_l1.item()*args.tvl1_weight,
                    })
            # calculte PCC between outputs and guse_embed
            pcc = [pearsonr(outputs[i].detach().cpu().numpy(), guse_embed[i].detach().cpu().numpy(),)[0] for i in range(len(outputs))]
            pcc_list_val += pcc
        wandb.log({"val_pcc": np.mean(pcc_list_val)})
        print('Val PCC:',np.mean(pcc_list_val))
    torch.save(model.state_dict(), os.path.join(args.output_dir,'subj%02d_linear_model.pt'%args.subj))
    weights = model.linear.weight.detach().cpu().numpy()
    weights = weights.mean(axis=0)
    voxel_weight = weights
    reconstructed_weight = reconstruct_volume_corrected(nsdgeneral_roi_mask.shape, nsdgeneral_roi_mask.flatten(), voxel_weight)
    reconstructed_weight = np.nan_to_num(reconstructed_weight)
    ni_img = nib.Nifti1Image(reconstructed_weight,affine=nsdgeneral_affine)
    plot_stat_map(ni_img, bg_img=anat_img, threshold=0.0, display_mode='z', cut_coords=10, colorbar=True, vmax=0.1, output_file=os.path.join(args.output_dir,'/subj%02d_linear_model.png'%args.subj))
    
    wandb.finish()