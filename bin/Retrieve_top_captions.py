import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import nibabel as nib
import utils
import os
from tqdm.notebook import tqdm
import argparse


data_path = "/scratch/cl6707/Shared_Datasets/NSD_MindEye"
nsd_root = "/scratch/cl6707/Projects/neuro_interp/data/NSD/"
stim_root = nsd_root + "nsddata_stimuli/stimuli/nsd/"
beta_root = nsd_root + "nsddata_betas/ppdata/"
mask_root = nsd_root + "nsddata/ppdata/"
nsd_mindroot = '/scratch/cl6707/Shared_Datasets/NSD_MindEye'

guse = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

subj =1

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed=42
utils.seed_everything(seed=seed)

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
batch_size = 16
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
    cache_dir=data_path, #"/tmp/wds-cache",
    seed=seed,
    voxels_key='nsdgeneral.npy', # 'nsdgeneral.npy' (1d), 'wholebrain_3d.npy'(3d)
    to_tuple=["voxels", "images", "coco","trial"],
    local_rank=local_rank,
    world_size=world_size,
)

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', type=str, default= '/scratch/cl6707/Projects/neuro_interp/Neural_Interpretation/notebooks/output/subj01_linear_model.pt')
argparser.add_argument('--mask_id', type=int, default= 0)
argparser.add_argument('--load_mask', type=int, default= 0)
argparser.add_argument('--mask_path', type=str, default= '/scratch/cl6707/Projects/neuro_interp/Neural_Interpretation/notebooks/output/brain_decoding/masks.h5')
argparser.add_argument('--output_dir', type=str, default= '/scratch/cl6707/Projects/neuro_interp/Neural_Interpretation/notebooks/output/brain_decoding')
argparser.add_argument('--guse_norm', type=int, default= 0)
args = argparser.parse_args()


from scipy.stats import pearsonr
import h5py
annotation_all = np.load(nsd_mindroot + '/subj%02d_annot.npy'%subj,allow_pickle=True)
nsdgeneral_affine = nib.load('/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz').affine
nsdgeneral_roi_mask = nib.load('/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz').get_fdata()==1
nsdgeneral1d = nsdgeneral_roi_mask.flatten()
anat_img = '/scratch/cl6707/Projects/neuro_interp/data/NSD/nsddata/ppdata/subj01/func1pt8mm/T1_to_func1pt8mm.nii.gz'

masks = {}
with h5py.File('/scratch/cl6707/Projects/neuro_interp/Neural_Interpretation/notebooks/output/brain_decoding/masks.h5', 'r') as hf:
    for key in hf.keys():
        masks[key] = hf[key][:]
        
mask_id_num_voxels_mapping = {i:key for i,key in enumerate(list(masks.keys()))}

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, alpha, l1_ratio):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def forward(self, x):
        return self.linear(x)

    def tv_l1_regularization(self):
        weights = self.linear.weight
        grad_x = torch.abs(weights[:, 1:] - weights[:, :-1])
        grad_y = torch.abs(weights[1:, :] - weights[:-1, :])
        tv_norm = torch.sum(grad_x) + torch.sum(grad_y)
        l1_norm = torch.sum(torch.abs(weights))
        return self.l1_ratio * tv_norm + (1 - self.l1_ratio) * l1_norm
    
@torch.no_grad()
def get_guse(annotation,mode='max'):
    B = annotation.shape[0]
    annot_embed_all = np.zeros((B,512))
    if mode == 'max':
        for b in range(B):
            annot_embed = guse(annotation[b][0]).cpu().numpy()
            corr_mat = np.corrcoef(annot_embed)
            max_index = np.argmax(corr_mat.mean(axis=0),axis=0)
            if args.guse_norm:
                annot_embed_all[b] = annot_embed[max_index]/np.linalg.norm(annot_embed[max_index])
            else:
                annot_embed_all[b] = annot_embed[max_index]
    elif mode == 'mean':
        for b in range(B):
            annot_embed = guse(annotation[b][0]).cpu().numpy()
            annot_embed_all[b] = annot_embed.mean(axis=0)
    else:
        raise NotImplementedError
    
    return torch.tensor(annot_embed_all).to(device).float()

def get_guse_one(annotation,mode='max'):
    annot_embed = guse(annotation).cpu().numpy()
    if mode == 'max':
        corr_mat = np.corrcoef(annot_embed)
        max_index = np.argmax(corr_mat.mean(axis=0),axis=0)
        if args.guse_norm:
            annot_embed = annot_embed[max_index]/np.linalg.norm(annot_embed[max_index])
        else:
            annot_embed = annot_embed[max_index]
    elif mode == 'mean':
        annot_embed = annot_embed.mean(axis=0)
    else:
        raise NotImplementedError
    
    return torch.tensor(annot_embed).to(device).float()

def eval_regression(model, data_loader,mask):
    model.eval()
    pcc_list = []
    for voxels, images, coco, trial in data_loader:
        voxels = voxels.float().to(device)
        voxels = voxels.mean(axis=1)
        if mask is not None:
            voxels = voxels*mask
        guse_embed = get_guse(annotation_all[trial])
        outputs = model(voxels)
        # calculte PCC between outputs and guse_embed
        pcc = [pearsonr(outputs[i].detach().cpu().numpy(), guse_embed[i].detach().cpu().numpy(),)[0] for i in range(len(outputs))]
        pcc_list += pcc
    return np.mean(pcc_list)


Caption_embeddings = None
Captions_descriptions = None
with h5py.File('/scratch/cl6707/Shared_Datasets/ConceptualCaptions/caption_embeddings.h5', 'r') as hf:
    Caption_embeddings = hf['caption_embeddings'][:]
    Captions_descriptions = hf['captions'][:]

import random

# Randomly sample 500,000 indices
sample_indices = random.sample(range(len(Caption_embeddings)), 500000)

# Select the sampled data points
Caption_embeddings = Caption_embeddings[sample_indices]
Captions_descriptions = Captions_descriptions[sample_indices]
    
from scipy.spatial.distance import cosine
def find_closest_caption(sample_embed, captions, embeddings_pool):
    closest_distance = 1
    closest_caption = None
    
    distances = np.array([cosine(sample_embed, embedding) for embedding in embeddings_pool])
    closest_index = np.argmin(distances)
    
    closest_distance = distances[closest_index]
    closest_caption = captions[closest_index]
    
    return closest_caption, closest_distance
    

if __name__ == '__main__':
    base_model = LinearModel(num_voxels, 512, 0.1, 0.5)
    base_model.load_state_dict(torch.load(argparser.parse_args().model_path,map_location=device))
    base_model.to(device)
    base_model.eval()
    print('Base model loaded...')
    min_distance = 1.0
    val_iter = iter(val_dl)
    for _ in tqdm(range(num_val)):
        voxels, images, coco, trial = next(val_iter)
        voxels = voxels.float().to(device)
        voxels = voxels.mean(axis=1)
        if len(trial) == 1:
            guse_embed = get_guse_one(annotation_all[trial])
        else:
            guse_embed = get_guse(annotation_all[trial])
        outputs = base_model(voxels)
        # calculte PCC between outputs and guse_embed
        retrived_caption,distance = find_closest_caption(guse_embed.detach().cpu().numpy(),Captions_descriptions,Caption_embeddings)
        if distance < min_distance:
            min_distance = distance
            plt.imshow(images[0].permute(1,2,0))
            plt.savefig(os.path.join(args.output_dir,'subj%02d_trial%02d_distance%0.4f.png'%(subj,trial[0],distance)))
            # save the caption
            with open(os.path.join(args.output_dir,'subj%02d_trial%02d_distance%0.4f.txt'%(subj,trial[0],distance)),'w') as f:
                f.write(retrived_caption.decode('UTF8'))
            print(retrived_caption, distance)
            print(annotation_all[trial])
    