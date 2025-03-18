import sys
import os
import torch
import torch.distributed as dist

from validation.classification_validation import validate as classification_validation
from validation.segmentation_validation import validate as segmentation_validation
from validation.change_validation import validate as change_validation
from validation.detection_validation import validate as detection_validation
from validation.height_validation import validate as height_validation

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import util.misc as misc
from util.misc import EarlyStopping

class ValidationHandler:
    def __init__(self, args, model, model_without_ddp, data_loader_val, optimizer, loss_scaler, log_writer, rank):
        self.args = args
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.data_loader_val = data_loader_val
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.log_writer = log_writer
        self.rank = rank

        # Default measures can be modified based on the dataset
        if self.args.task == "classification":
            self.measures = ['loss', 'accuracy', 'precision', 'recall', 'f1']
        elif self.args.task == "segmentation":
            self.measures = ['loss', 'iou', 'precision', 'recall', 'f1', 'accuracy']
        elif self.args.task == "change":
            self.measures = ['loss', 'accuracy', 'iou', 'precision', 'recall', 'f1']
        elif self.args.task == "detection":
            self.measures = ['loss', 'loss_iou', 'loss_cls', 'mAP']#, 'precision', 'recall']
        elif self.args.task == "height":
            self.measures = ['me', 'mae', 'rmse']

        # Initialize early stopping mechanisms for all the provided measures
        if self.args.task != "height":
            self.early_stopping_criteria = {measure: EarlyStopping(verbose=True, val_tag=measure, minim=(measure == "loss")) for measure in self.measures}
        else:
            self.early_stopping_criteria = {measure: EarlyStopping(verbose=True, val_tag=measure, minim=True) for measure in self.measures}
        

    def update(self, loss, epoch, device):
        """
        Call classification validation based on dataset and measures.
        """

        # Run validation
        if self.args.task == "classification":
            results = classification_validation(self.args, self.model, self.data_loader_val, device)
        elif self.args.task == "segmentation":
            results = segmentation_validation(self.args, self.model, self.data_loader_val, device)
        elif self.args.task == "change":
            results = change_validation(self.args, self.model, self.data_loader_val, device)
        elif self.args.task == "detection":
            results = detection_validation(self.args, self.model, self.data_loader_val, device)
        elif self.args.task == "height":
            results = height_validation(self.args, self.model, self.data_loader_val, device)

        metric_dict = dict(zip(self.measures, results))

        # Aggregate metrics across all ranks if in distributed mode
        if dist.is_initialized():
            for measure, value in metric_dict.items():
                value_tensor = torch.tensor([value], device=device)
                dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                metric_dict[measure] = value_tensor.item() / dist.get_world_size()  # Average over all ranks

        # Logging and early stopping handled by rank 0
        if self.rank == 0 and not self.args.disable_logging:
            # Update TensorBoard for all measures
            if self.log_writer is not None:
                for measure, value in metric_dict.items():
                    self.log_writer.add_scalar(f'val/{measure}', value, epoch)

            # Update early stopping for all measures
            for measure, early_stopper in self.early_stopping_criteria.items():
                if measure == 'me':
                    early_stopper(abs(metric_dict[measure]))
                else:
                    early_stopper(metric_dict[measure])
                
                if early_stopper.is_best:
                    misc.save_model(
                        args=self.args, model=self.model, model_without_ddp=self.model_without_ddp, 
                        optimizer=self.optimizer, loss_scaler=self.loss_scaler, epoch=epoch + 1, 
                        rand_id=self.args.rand_id, val_tag=measure)

        # Synchronize all processes after validation
        dist.barrier()

    def check_early_stop(self, device):
        """
        Determine if early stopping should be triggered by any of the measures.
        """
        stop_training = torch.tensor([0], device=device)
        
        # Only rank 0 checks for early stopping
        if self.rank == 0 and self.args.earlystop and any([stopper.early_stop for stopper in self.early_stopping_criteria.values()]):
            print("Early stopping caused by:")
            for measure, early_stopper in self.early_stopping_criteria.items():
                if early_stopper.early_stop:
                    print(f"- Validation {measure.capitalize()}")
            stop_training = torch.tensor([1], device=device)

        # Broadcast early stopping decision to all ranks
        dist.broadcast(stop_training, src=0)
        
        return stop_training.item() == 1