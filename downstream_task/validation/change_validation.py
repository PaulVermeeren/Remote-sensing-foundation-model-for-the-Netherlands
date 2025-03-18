import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast

import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex, Precision, Recall, F1Score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss import calculate_loss

def validate(args, model, data_loader, device):
    model.eval()
    
    num_classes = args.num_classes
    iou_metric = JaccardIndex(num_classes=num_classes, task='binary').to(device)
    precision_metric = Precision(num_classes=num_classes, average='binary', task='binary').to(device)
    recall_metric = Recall(num_classes=num_classes, average='binary', task='binary').to(device)
    f1_metric = F1Score(num_classes=num_classes, average='binary', task='binary').to(device)
    
    correct_predictions = torch.zeros(1).to(device)
    total_samples = torch.zeros(1).to(device)
    
    total_loss = torch.zeros(1).to(device)
    loss_count = torch.zeros(1).to(device)
    
    with torch.no_grad(), autocast('cuda'):
        for batch_idx, batch in enumerate(data_loader):
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = calculate_loss(args, outputs, masks) 

            preds = outputs.argmax(dim=1)  # Assuming outputs shape is [B, C, H, W]
            
            masks_squeeze = masks.squeeze(1).squeeze(1).long()
            iou_metric.update(preds, masks_squeeze)
            precision_metric.update(preds, masks_squeeze)
            recall_metric.update(preds, masks_squeeze)
            f1_metric.update(preds, masks_squeeze)

            # Flatten the predictions and masks for accuracy calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # Count correct predictions
            correct_predictions += (preds_flat == masks_flat).sum().item()
            total_samples += masks_flat.numel()
            
            total_loss += loss.item()
            loss_count += 1

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
    
    dist.all_reduce(correct_predictions, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    avg_loss = (total_loss / loss_count).item()
    
    # Calculate accuracy
    accuracy = ((correct_predictions / total_samples) * 100).item()
    
    avg_iou = iou_metric.compute().cpu().item() * 100
    precision = precision_metric.compute().cpu().item() * 100
    recall = recall_metric.compute().cpu().item() * 100
    f1 = f1_metric.compute().cpu().item() * 100

    # Only print on the main process
    if dist.get_rank() == 0:
        print(f"Validation Loss: {loss:.2f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Validation Avg_iou: {avg_iou:.2f}%")
        print(f"Validation Precision: {precision:.2f}%")
        print(f"Validation Recall: {recall:.2f}%")
        print(f"Validation F1: {f1:.2f}%")

    return avg_loss, accuracy, avg_iou, precision, recall, f1