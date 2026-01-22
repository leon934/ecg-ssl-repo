import logging
import datetime
from pathlib import Path

import torch
import torch.distributed.nn.functional as dist_nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from accelerate import Accelerator

from utils import accuracy, save_config_file

torch.manual_seed(0)

class SimCLR(object):
    def __init__(self, model, optimizer, scheduler, accelerator: Accelerator, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.args = args

        self.writer = SummaryWriter()
        self.pretraining_criterion = torch.nn.CrossEntropyLoss()
        self.finetuning_criterion = torch.nn.MSELoss()

        self.device = self.accelerator.device

    def info_nce_loss(self, features):
        features = torch.distributed.nn.functional.all_gather(features)
        features = torch.cat(features, dim=0)

        total_size = features.shape[0]  
        # We assume 'features' input was [View1, View2] stacked locally
        # So the local chunk size is total_size / num_processes
        # But simpler: we know every even chunk is V1, odd chunk is V2.
        
        # Reshape to (Num_GPUs, 2_Views, Local_Batch, Dim)
        # Note: This assumes your local batch size is constant. 
        # A safer way without assuming constant local batch is simply sorting if you track indices, 
        # but for standard DDP this reshape works:
        
        n_views = 2
        # This recovers the global batch size (e.g., 48 if local is 12 and gpus=4)
        global_batch_size = features.shape[0] // n_views 
        
        # Reshape: [Global_Batch*2, Dim] -> [2, Global_Batch, Dim]
        # We have to be careful. The gather order is Process0, Process1, Process2...
        # Process0 has [v1_0..v1_n, v2_0..v2_n].
        # We want to pull all v1s together.
        
        # Reshape to [Num_Processes, 2, Local_Batch, Dim]
        num_processes = self.accelerator.num_processes
        local_batch = global_batch_size // num_processes
        
        features = features.view(num_processes, 2, local_batch, -1) 
        
        # Permute to [2, Num_Processes, Local_Batch, Dim] to group views
        features = features.permute(1, 0, 2, 3)
        
        # Flatten back to [2 * Global_Batch, Dim]
        # Now it looks like [All_View1s, All_View2s]
        features = features.reshape(2 * global_batch_size, -1)
        # --- FIX END ---

        # 2. Update the variable usage (Use calculated global_batch_size)
        batch_size = global_batch_size

        batch_size = features.shape[0] // 2

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader: torch.utils.data.DataLoader) -> None:
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs with {self.args.dataset_name} dataset.")

        date = datetime.datetime.now()

        for curr_epoch in range(self.args.epochs):
            for images, _ in train_loader:
                images = torch.cat(images, dim=0)

                with self.accelerator.autocast():
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)

                    loss = self.pretraining_criterion(logits, labels)

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    avg_loss = self.accelerator.gather(loss).mean()
                    
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar('loss', avg_loss, global_step=n_iter)
                        self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                        self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                
                n_iter += 1

            if curr_epoch >= 10:
                self.scheduler.step()

            # save model checkpoints
            checkpoint_name = "checkpoint_{:04d}.pth.tar".format(curr_epoch)
            curr_model_path = Path("models/pretraining/{date:%m-%d-%Y_%H:%M:%S}_checkpoints".format(date=date))
            curr_model_path.mkdir(parents=True, exist_ok=True)
            
            # accelerate wraps in distributeddataparallel class so we have to access the module first
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            
            checkpoint = {
                'epoch': self.args.epochs,
                'model': self.args.model,
                'state_dict': model_to_save.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            # applicable to hf architectures
            # todo: create model class for simclr to req. self.backbone, self.head, and self.config so we don't need to have this safe guard
            if hasattr(model_to_save, "config"):
                checkpoint["config"] = model_to_save.backbone.config.to_dict()

            if self.accelerator.is_main_process:
                save_checkpoint(checkpoint, is_best=False, filename=curr_model_path / checkpoint_name)
    
            logging.info(f"Model checkpoint {curr_epoch} and metadata has been saved at {curr_model_path}.")
            logging.debug(f"Epoch: {curr_epoch}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training finished.")

    def finetune(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader
    ):  
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Starting fine-tuning for downstream task: {self.args.y_name} prediction.")

        date = datetime.datetime.now()
        n_iter = 0

        for curr_epoch in range(self.args.epochs):
            self.model.train()
            for image, label in train_loader:
                with self.accelerator.autocast():
                    pred = self.model(image).squeeze(-1)
                    loss = self.finetuning_criterion(pred, label)

                self.accelerator.backward(loss)

                self.optimizer.zero_grad()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    avg_loss = self.accelerator.gather(loss).mean()

                    self.writer.add_scalar('loss', avg_loss, global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                
                n_iter += 1

            if curr_epoch >= 10:
                self.scheduler.step()

            # save model checkpoints
            checkpoint_name = "checkpoint_{:04d}.pth.tar".format(curr_epoch)
            curr_model_path = Path("models/finetuning/{date:%m-%d-%Y_%H:%M:%S}_checkpoints".format(date=date))
            curr_model_path.mkdir(parents=True, exist_ok=True)
            
            # accelerate wraps in distributeddataparallel class so we have to access the module first
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            
            checkpoint = {
                'epoch': self.args.epochs,
                'model': self.args.model,
                'state_dict': model_to_save.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            # applicable to hf architectures
            # todo: create model class for simclr to req. self.backbone, self.head, and self.config so we don't need to have this safe guard
            if hasattr(model_to_save, "config"):
                checkpoint["config"] = model_to_save.backbone.config.to_dict()

            accelerator.save(obj=checkpoint, f=curr_model_path / checkpoint_name)
            # save_checkpoint(checkpoint, is_best=False, filename=curr_model_path / checkpoint_name)
            logging.info(f"Model checkpoint {curr_epoch} and metadata has been saved at {curr_model_path}.")

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for image, label in val_loader:
                    pred = self.model(image).squeeze(-1)
                    val_loss += self.finetuning_criterion(pred, label)

            val_loss /= len(val_loader)

        self.model.train()
        test_loss = 0.0

        for image, label in test_loader:
            pred = self.model(image).squeeze(-1)
            test_loss += self.finetuning_criterion(pred, label)

        logging.info(f"Final testing loss: {test_loss}")
        logging.info("Fine-tuning finished.")