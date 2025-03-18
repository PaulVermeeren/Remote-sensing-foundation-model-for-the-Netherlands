import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast
from torchmetrics import Precision, Recall, F1Score, Accuracy

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss import calculate_loss

def validate(args, model, data_loader, device):
    model.eval()
    
    num_classes = args.num_classes
    
    accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    precision_metric = Precision(task='multiclass', average='weighted', num_classes=num_classes).to(device)
    recall_metric = Recall(task='multiclass', average='weighted', num_classes=num_classes).to(device)
    f1_metric = F1Score(task='multiclass', average='weighted', num_classes=num_classes).to(device)
    
    total_loss = torch.zeros(1, device=device)
    loss_count = torch.zeros(1, device=device)
    
    with torch.no_grad(), autocast('cuda'):
        for batch_idx, batch in enumerate(data_loader):
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = calculate_loss(args, outputs, labels) 
            
            # Get predictions
            preds = outputs.argmax(dim=1)  # preds shape: (batch_size,)
            
            # Update metrics
            accuracy_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            
            # Accumulate loss
            total_loss += loss.detach()
            loss_count += 1

    # Synchronize losses across processes
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)

    # Calculate average loss
    avg_loss = (total_loss / loss_count).item()
    
    # Compute metrics
    avg_accuracy = accuracy_metric.compute().cpu().item() * 100
    precision = precision_metric.compute().cpu().item() * 100
    recall = recall_metric.compute().cpu().item() * 100
    f1 = f1_metric.compute().cpu().item() * 100

    # Only print/log results on rank 0
    if dist.get_rank() == 0:
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {avg_accuracy:.2f}%")
        print(f"Validation Precision (Macro Avg): {precision:.2f}%")
        print(f"Validation Recall (Macro Avg): {recall:.2f}%")
        print(f"Validation F1 Score (Macro Avg): {f1:.2f}%")
    
    return avg_loss, avg_accuracy, precision, recall, f1