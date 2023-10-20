# Neural_Interpretation

This repo for fMRI image interpretation
* Intro to Haxby dataset: [Here](https://main-educational.github.io/brain_encoding_decoding/haxby_data.html)

Framework: 

fMRI--> Model --> Feature Regression <-- Generated Features <-- Pretrained Model <-- Stimulus
* 1016 TODO:
    - Extracting features from stimulus with pretrained model(ViT/ResNet50/...)
        - Figure out with layer of the pretrained model to extract features
        - Apply PCA on extracted features and get 1st coefficient
        - Figure out what the 1st coefficient stands for (e.g. direction of the chair)
    - NSD dataset Dataloader
        - Figure out how to load NSD dataset
        - Figure out how to load stimulus
        - Figure out how to load fMRI data and which to use
    - Implement (Deep Lasso/ MLP/ SpaceNet/ FREM) for our new framework
        - Dataloader/ Model/ Loss/ Optimizer/ Training loop
        - Get corresponding brain region for each feature





<!-- * Todo:
    - Implement SpaceNet on Haxby
    - 3D Grad-CAM 
    - Try Deep Lasso
    - Try CNN
    - Pretrain deep model with contrastive learning by recon (Probably we can use fMRI+image CLIP/ MAE/ Swap) -->

