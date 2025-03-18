# README: Remote Sensing Foundation Model for the Netherlands

## Overview

This repository provides the implementation and evaluation of a novel Foundation Model tailored for remote sensing tasks in the Netherlands. The model is trained on high-resolution satellite imagery sourced from Pleiades and Superview NEO, which are freely accessible to Dutch citizens. A key innovation of this research is the integration of multiple state-of-the-art methodologies into a Swin-T backbone, optimizing both spatial and temporal feature learning. The model is designed to handle a variable number of temporal images, enabling it to capture seasonal variations at the same location. This temporal learning approach significantly improves performance, particularly when fine-tuning on single-image datasets. The performance of the model is validated on standard remote sensing benchmark datasets, being RESISC-45, UC-Merced, Potsdam, and LEVIR-CD. These result show the capabilites and wide applicabilities of this model's architecture having very little parameters. For these experiments, the model has been pre-trained on half of the Netherlands using six temporal timestamps, serving as a country-specific Foundation Model while still demonstrating strong performance on global benchmark datasets. This highlights the generalizability of the architecture, despite being trained on a geographically limited dataset.

![Model Architecture](assets/architecture.png)

## Usage

### 1. Training the Model

To pre-train the Foundation Model on satellite imagery:

The `main_pretrain.py` can be adapted to load your specific data source. Specifying the model parameters properly using arguments would enable training the model using `python main_pretrain.py --args`. 

### 2. Fine-tuning for a specific downstream task:

A pretrained model can be used for fine-tuning using the `downstream_task/main.py` file. Setting up the configuration files for the specific task and specifying the dataloader, decoder head, and validation metrics enables finetuning the model to a specific task.

## Performance Comparison

![Potsdam](assets/performance_potsdam.png)
![Resisc45](assets/performance_resisc.png)
![UC-Merced](assets/performance_merced.png)
![Levir-CD](assets/performance_levir.png)

## Citation

If you use this code in your research, please cite this repository. 
The corresponding paper will be published soon and the repository will be updated accordingly.
For contact, visit www.paulvermeeren.nl.
