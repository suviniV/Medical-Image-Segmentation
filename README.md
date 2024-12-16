## Medical Image Segmentation using CNNs and Diffusion Models
### Overview
This repository focuses on Medical Image Segmentation using two model architectures:

> CNN (Convolutional Neural Networks) and 
> Diffusion Models (MegSegDiff)
The models are trained and evaluated on the BraTS 2020 Dataset, which focuses on brain tumor segmentation.

### Introduction
This project explores brain tumor segmentation in medical imaging using two state-of-the-art architectures:

CNN: 3d CNN+ U-Net
Diffusion Models: MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer
The BraTS 2020 dataset is used to validate these methods, as it contains annotated MRI scans (T1, T2, T1ce, FLAIR) for brain tumor regions.

### Folder Structure
The repository is organized as follows:
├── CNN/                # CNN model for medical image segmentation
├── MegSegDiff/         # Diffusion-based segmentation models

### Dataset (https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015) 

The BraTS 2020 Dataset is used for training and evaluation.

Task: Brain Tumor Segmentation
Modalities: MRI scans (T1, T2, T1ce, FLAIR)
Classes:
Necrotic tumor core
Peritumoral edema
Enhancing tumor
Dataset access: BraTS 2020 Challenge

### Models
(Old Technology) CNN Models Link: https://github.com/BRML/CNNbasedMedicalSegmentation

(Latest Technology) Diffusion Models Link: https://github.com/SuperMedIntel/MedSegDiff
