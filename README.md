# FoundationModel
A foundation model developed for the Netherlands using the Pleiades and Superview NEO satellites for remote sensing data.

# README: Foundation Model for Remote Sensing

## Overview

This repository contains the implementation and evaluation of a novel Foundation Model (FM) designed for remote sensing tasks on the Netherlands. The model is trained using high-resolution satellite imagery from the Netherlands collected by Pleiades and Superview NEO which are currently licensed for free use for Dutch citizens. The Foundation Model is based on the swin-T model and is adapted to handle temporal data. The performance of the model is validated on standard remote sensing benchmark datasets, being RESISC-45, UC-Merced, Potsdam, and LEVIR-CD.

![Model Architecture](assets/model_architecture.png)

## Usage

### 1. Training the Model

To pre-train the Foundation Model on satellite imagery:

The `main_pretrain.py` can be adapted to load your specific data source. Specifying the model parameters properly using arguments would enable training the model using `python main_pretrain.py --args`. 

### 2. Fine-tuning for a specific downstream task:



## Performance Comparison

The Foundation Model (FM) has been rigorously evaluated against benchmark datasets. Below are the key performance metrics:

| Dataset       | Top-1 Accuracy | mIoU   | F1 Score |
| ------------- | -------------- | ------ | -------- |
| **RESISC-45** | 95.59%         | -      | -        |
| **UC-Merced** | 98.10%         | -      | -        |
| **Potsdam**   | -              | 72.87% | 87.48%   |
| **LEVER-CD**  | -              | -      | 87.94%   |

Compared to established SOTA models, the FM provides **competitive results** despite being trained on a smaller dataset focused on the Netherlands.

## Citation

If you use this code in your research, please cite this repository. 
The corresponding paper will be published soon and the repository will be updated accordingly.

