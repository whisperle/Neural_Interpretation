# Neural_Interpretation

This repo for fMRI image interpretation
* Intro to Haxby dataset: [Here](https://main-educational.github.io/brain_encoding_decoding/haxby_data.html)
* NSD data manual: [Here](https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual)

Framework: 

1. fMRI--> Model --> Feature Regression <-- Generated Features <-- Pretrained Model <-- Stimulus
* 1023 TODO:
    - Load MindEye's model [here](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main)
        - check its gradient contribution to image reconstruction (Done)
        - Generate gradient contibution map on fMRI image (Problem here: time consuming to generate for the whole embedding(3h for 1 img))
        - Fuse gradient contribution map from 4 subjects by using [2nd level analysis](https://nilearn.github.io/stable/auto_examples/05_glm_second_level/plot_thresholding.html#)
    - Implement (Deep Lasso/ MLP/ SpaceNet/ FREM) for our new framework
        - Model/ Loss/ Optimizer/ Training loop
        - Get corresponding brain region for each feature
    - Extracting features from stimulus with pretrained model(ViT/ResNet50/...)
        - Figure out with layer of the pretrained model to extract features
        - Apply PCA on extracted features and get 1st coefficient
        - Figure out what the 1st coefficient stands for (e.g. main component of the image)

* Done:
    - NSD dataset Dataloader (Done) {we could use webdataset from Mindeye directly}
        1. Load the 3D fMRI data (Used for fMRI decoder)
        2. Load paired stimuli image (Used for feature extraction)
        3. Load annotation/ categorie for images



<!-- * Todo:
    - Implement SpaceNet on Haxby
    - 3D Grad-CAM 
    - Try Deep Lasso
    - Try CNN
    - Pretrain deep model with contrastive learning by recon (Probably we can use fMRI+image CLIP/ MAE/ Swap) -->

