import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from accelerate import Accelerator

from utils import accuracy, save_config_file, save_checkpoint

class SimCLR(object):
    def __init__(
            self, 
            model, 
            optimizer,
            scheduler,
            accelerator: Accelerator,
            args
        ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.args = args

        self.writer = SummaryWriter()
        self.pretraining_criterion = torch.nn.CrossEntropyLoss()
        self.finetuning_criterion = torch.nn.MSELoss()

        self.device = self.accelerator.device

    def info_nce_loss(self, features: torch.Tensor):
        # used instead of accelerator.gather since all_gather has back-propagation
        features = torch.distributed.nn.functional.all_gather(features)
        features = torch.cat(features, dim=0)

        total_size = features.shape[0]
        n_views = 2
        global_batch_size = total_size // n_views
        
        # reshape to [num_procs, 2, local_batch_size, Dim]
        num_processes = self.accelerator.num_processes
        local_batch = global_batch_size // num_processes
        
        features = features.view(num_processes, n_views, local_batch, -1) 
        features = features.permute(1, 0, 2, 3)
        features = features.reshape(2 * global_batch_size, -1)

        batch_size = global_batch_size

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

    def train(self, train_loader):
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0

        if self.accelerator.is_main_process:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs with {self.args.dataset_name} dataset.")

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

                if n_iter % self.args.log_every_n_steps == 0 and self.accelerator.is_main_process:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                
                n_iter += 1

            if curr_epoch >= 10:
                self.scheduler.step()

            # save model checkpoints
            checkpoint_name = "checkpoint_{:04d}.pth.tar".format(curr_epoch)
            curr_model_path = Path("models/pretraining/{date:%m-%d-%Y_%H:%M:%S}_{model}checkpoints".format(date=self.args.date, model=self.args.model))
            curr_model_path.mkdir(parents=True, exist_ok=True)
            
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

            if (curr_epoch + 1) % self.args.save_every_n == 0 or curr_epoch >= self.args.epochs - self.args.save_last_n:
                self.accelerator.save(checkpoint, curr_model_path / checkpoint_name)

            if self.accelerator.is_main_process:
                logging.info(f"Model checkpoint {curr_epoch} and metadata has been saved at {curr_model_path}.")
                logging.debug(f"Epoch: {curr_epoch}\tLoss: {loss}")

        if self.accelerator.is_main_process:
            logging.info("Training finished.")

    def finetune(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader
    ):  
        save_config_file(self.writer.log_dir, self.args)

        logging.info(f"Starting fine-tuning for downstream task: {self.args.y_name} prediction.")

        n_iter = 0

        for curr_epoch in range(self.args.epochs):
            self.model.train()
            for image, label in train_loader:
                with self.accelerator.autocast():
                    pred = self.model(image).squeeze(-1)
                    loss = self.finetuning_criterion(pred, label)

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                # if n_iter % self.args.log_every_n_steps == 0:
                #     avg_loss = self.accelerator.gather(loss).mean()

                #     self.writer.add_scalar('loss', avg_loss, global_step=n_iter)
                #     self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                
                n_iter += 1

            # save model checkpoints
            checkpoint_name = "checkpoint_{:04d}.pth.tar".format(curr_epoch)
            curr_model_path = Path("models/finetuning/{date:%m-%d-%Y_%H:%M:%S}_checkpoints".format(date=self.args.date))
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

            if (curr_epoch + 1) % self.args.save_every_n == 0 or curr_epoch >= self.args.epochs - self.args.save_last_n:
                save_checkpoint(checkpoint, is_best=False, filename=curr_model_path / checkpoint_name)
            # save_checkpoint(checkpoint, is_best=False, filename=curr_model_path / checkpoint_name)
            logging.info(f"Model checkpoint {curr_epoch} and metadata has been saved at {curr_model_path}.")

            self.model.eval()
            val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for image, label in val_loader:
                    pred = self.model(image).squeeze(-1)
                    loss = self.finetuning_criterion(pred, label)

                    batch_size = image.size(0)
                    val_loss += val_loss.item() * batch_size
                    total_val_samples += batch_size

            val_loss /= total_val_samples
            logging.info(f"Validation loss for epoch {curr_epoch}: {val_loss}")

        test_loss = 0.0
        total_test_samples = 0

        for image, label in test_loader:
            pred = self.model(image).squeeze(-1)
            loss += self.finetuning_criterion(pred, label)

            batch_size = image.size(0)
            test_loss += val_loss.item() * batch_size
            total_test_samples += batch_size

        test_loss /= total_test_samples

        logging.info(f"Final testing loss: {test_loss}")
        logging.info("Fine-tuning finished.")