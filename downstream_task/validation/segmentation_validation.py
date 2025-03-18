import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast

import numpy as np

import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex, Precision, Recall, F1Score, ConfusionMatrix, Accuracy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss import calculate_loss

def validate(args, model, data_loader, device):
    model.eval()
    
    # Initialize metrics
    if "vegetatie" in args.dataset:
        ignore_index = 0
        num_classes = args.num_classes + 1
        average = "macro"
    else:
        ignore_index = None
        num_classes = args.num_classes
        average = "weighted"
    
    iou_metric = JaccardIndex(num_classes=num_classes, task='multiclass', ignore_index=ignore_index).to(device)
    precision_metric = Precision(num_classes=num_classes, average=average, task='multiclass', ignore_index=ignore_index).to(device)
    recall_metric = Recall(num_classes=num_classes, average=average, task='multiclass', ignore_index=ignore_index).to(device)
    f1_metric = F1Score(num_classes=num_classes, average=average, task='multiclass', ignore_index=ignore_index).to(device)
    cm_metric = ConfusionMatrix(num_classes=num_classes, task='multiclass').to(device)
    accuracy_metric = Accuracy(num_classes=num_classes, task='multiclass', ignore_index=ignore_index).to(device) 
    
    per_class_precision = Precision(num_classes=num_classes, average=None, task='multiclass', ignore_index=ignore_index).to(device)
    per_class_recall = Recall(num_classes=num_classes, average=None, task='multiclass', ignore_index=ignore_index).to(device)
    per_class_f1 = F1Score(num_classes=num_classes, average=None, task='multiclass', ignore_index=ignore_index).to(device)
    per_class_accuracy = F1Score(num_classes=num_classes, average=None, task='multiclass', ignore_index=ignore_index).to(device)
    
    total_loss = torch.zeros(1).to(device)
    total_samples = torch.zeros(1).to(device)
    
    with torch.no_grad(), autocast('cuda'):
        for batch_idx, batch in enumerate(data_loader):
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = calculate_loss(args, outputs, masks) 
            
            preds = outputs.argmax(dim=2)
            masks = masks.squeeze(2)
            
            if "vegetatie" in args.dataset:
                preds = preds + 1
            
            # Update metrics
            iou_metric.update(preds, masks)
            precision_metric.update(preds, masks)
            recall_metric.update(preds, masks)
            f1_metric.update(preds, masks)
            cm_metric.update(preds, masks)
            accuracy_metric.update(preds, masks)  # Update accuracy
            per_class_precision.update(preds, masks)
            per_class_recall.update(preds, masks)
            per_class_f1.update(preds, masks)
            per_class_accuracy.update(preds, masks)
            
            total_loss += loss.item()
            total_samples += 1
    
    # Synchronize across GPUs
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    avg_loss = (total_loss / total_samples).item()
    
    # Compute metrics
    avg_iou = iou_metric.compute().cpu().item() * 100
    precision = precision_metric.compute().cpu().item() * 100
    recall = recall_metric.compute().cpu().item() * 100
    f1 = f1_metric.compute().cpu().item() * 100
    accuracy = accuracy_metric.compute().cpu().item() * 100  # Compute accuracy
    cm = cm_metric.compute().cpu().numpy()
    
    per_class_precision_values = per_class_precision.compute().cpu().numpy() * 100
    per_class_recall_values = per_class_recall.compute().cpu().numpy() * 100
    per_class_f1_values = per_class_f1.compute().cpu().numpy() * 100
    per_class_accuracy_values = per_class_accuracy.compute().cpu().numpy() * 100
    
    # Include all classes in IoU calculation
    class_iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)) * 100
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    per_class_metrics = {
        'iou': class_iou,
        'precision': per_class_precision_values,
        'recall': per_class_recall_values,
        'f1': per_class_f1_values,
        'accuracy': per_class_accuracy_values
    }
    
    torch.cuda.empty_cache()
    
    # Only print on the main process
    if dist.get_rank() == 0:
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation IoU: {avg_iou:.2f}%")
        print(f"Validation Precision: {precision:.2f}%")
        print(f"Validation Recall: {recall:.2f}%")
        print(f"Validation F1: {f1:.2f}%")
        print(f"Validation Accuracy: {accuracy:.2f}%")  # Print accuracy
        
        if "vegetatie" in args.dataset:
            for i in range(num_classes-1):
                print(f"\nClass {i}:")
                print(f"  IoU: {per_class_metrics['iou'][i+1]:.2f}%")
                print(f"  Precision: {per_class_metrics['precision'][i+1]:.2f}%")
                print(f"  Recall: {per_class_metrics['recall'][i+1]:.2f}%")
                print(f"  F1: {per_class_metrics['f1'][i+1]:.2f}%")
        else:
            for i in range(num_classes):
                print(f"\nClass {i}:")
                print(f"  IoU: {per_class_metrics['iou'][i]:.2f}%")
                print(f"  Precision: {per_class_metrics['precision'][i]:.2f}%")
                print(f"  Recall: {per_class_metrics['recall'][i]:.2f}%")
                print(f"  F1: {per_class_metrics['f1'][i]:.2f}%")
                print(f"  Accuracy: {per_class_metrics['accuracy'][i]:.2f}%")
            
        print("\nPer-Class Metrics (including all classes):")
        
    
    return avg_loss, avg_iou, precision, recall, f1, accuracy #, cm, cm_normalized, per_class_metrics