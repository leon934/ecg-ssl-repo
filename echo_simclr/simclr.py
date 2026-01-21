import logging
import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from accelerate import Accelerator

from utils import accuracy, save_checkpoint, save_config_file

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

        # self.device = self.accelerator.device

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for _ in range(2)], dim=0)
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
        logging.info(f"Using device: {self.args.device}.")
        logging.info(f"CUDA disabled: {self.args.disable_cuda}.")

        date = datetime.datetime.now()

        for curr_epoch in range(self.args.epochs):
            for images, _ in train_loader:
                images = torch.cat(images, dim=0).to(self.args.device)

                with self.accelerator.autocast():
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)

                    loss = self.pretraining_criterion(logits, labels)

                self.accelerator.backward(loss)

                self.optimizer.zero_grad()
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    avg_loss = self.accelerator.gather(loss).mean()

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
            
            checkpoint = {
                'epoch': self.args.epochs,
                'model': self.args.model,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            # applicable to hf architectures
            # todo: model class for simclr to req. self.backbone, self.head, and self.config so we don't need to have this safe guard
            if hasattr(self.model.backbone, "config"):
                checkpoint["config"] = self.model.backbone.config.to_dict()

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
            
            checkpoint = {
                'epoch': self.args.epochs,
                'model': self.args.model,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            # applicable to hf architectures
            # todo: create model class for simclr to req. self.backbone, self.head, and self.config so we don't need to have this safe guard
            if hasattr(self.model.backbone, "config"):
                checkpoint["config"] = self.model.backbone.config.to_dict()

            save_checkpoint(checkpoint, is_best=False, filename=curr_model_path / checkpoint_name)
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