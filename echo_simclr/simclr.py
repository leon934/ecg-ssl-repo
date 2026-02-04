import logging
from pathlib import Path

from accelerate import Accelerator
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchmetrics.regression import R2Score

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

        # accel. and is_main are for pretraining
        # accel. is none is for finetuning, since it isn't necessary to use accel. there
        if self.accelerator and self.accelerator.is_main_process or self.accelerator is None:
            # todo: lowk a bandaid fix
            mode = "pretraining" if self.accelerator else "finetuning"

            self.writer = SummaryWriter(log_dir="runs/{train_mode}/{date:%m-%d-%Y_%H:%M:%S}_{arch}".format(date=self.args.date, arch=self.args.arch, train_mode=mode))        

        if self.accelerator:
            self.device = self.accelerator.device
        else:
            self.device = "cuda"

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
        self.pretraining_criterion = torch.nn.CrossEntropyLoss()

        if self.accelerator.is_main_process:
            save_config_file(self.writer.log_dir, self.args)

        n_iter = 0

        if self.accelerator.is_main_process:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs with {self.args.dataset_name} dataset.")

        for curr_epoch in range(self.args.epochs):
            epoch_loss = 0

            for images, _ in train_loader:
                images = torch.cat(images, dim=0)

                with self.accelerator.autocast():
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)

                    loss = self.pretraining_criterion(logits, labels)
                    epoch_loss += loss.item()

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                if n_iter % self.args.log_every_n_steps == 0 and self.accelerator.is_main_process:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    self.writer.add_scalar('loss_n_steps', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                
                n_iter += 1
                self.scheduler.step()

            if self.accelerator.is_main_process:
                avg_loss = epoch_loss / len(train_loader)
                self.writer.add_scalar('loss_epoch', avg_loss, global_step=curr_epoch)

            # save model checkpoints
            checkpoint_name = "checkpoint_{:04d}.pth.tar".format(curr_epoch)
            curr_model_path = Path("models/pretraining/{date:%m-%d-%Y_%H:%M:%S}-{model}-checkpoints".format(date=self.args.date, model=self.args.arch))
            curr_model_path.mkdir(parents=True, exist_ok=True)
            
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model

            checkpoint = {
                'epoch': self.args.epochs,
                'model': self.args.arch,
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
                logging.debug(f"Epoch: {curr_epoch}\tLoss: {avg_loss}")

        if self.accelerator.is_main_process:
            logging.info("Training finished.")

    def finetune(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader
    ):  
        self.mae_criterion = torch.nn.L1Loss(reduction='none')
        self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.r2_criterion = R2Score(num_outputs=3, multioutput='raw_values').to(self.device)

        save_config_file(self.writer.log_dir, self.args)
        scaler = GradScaler(device=self.args.device, enabled=self.args.fp16_precision)

        logging.info(f"Starting fine-tuning for downstream task: {self.args.y_name} prediction.")

        n_iter = 0

        for curr_epoch in range(self.args.epochs):
            self.model.train()
            epoch_mse = torch.zeros((3,), device=self.device)

            for images, label in train_loader:
                images = images.to(self.device)
                label = label.to(self.device)

                with autocast(device_type=self.args.device, enabled=self.args.fp16_precision):
                    pred = self.model(images)
                    raw_loss = self.mse_criterion(pred, label)

                    self.r2_criterion.update(pred, label)

                    loss_scalar = raw_loss.mean()
                    loss_per_target = raw_loss.mean(dim=0).detach()

                    epoch_mse += loss_per_target * self.args.batch_size

                self.optimizer.zero_grad()
                scaler.scale(loss_scalar).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('ft/esv_mse_n_iter', loss_per_target[0], global_step=n_iter)
                    self.writer.add_scalar('ft/edv_mse_n_iter', loss_per_target[1], global_step=n_iter)
                    self.writer.add_scalar('ft/ef_mse_n_iter', loss_per_target[2], global_step=n_iter)
                    
                    self.writer.add_scalar('ft/learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                
                n_iter += 1

            avg_mse = epoch_mse / len(train_loader)
            self.writer.add_scalar('ft/esv_mse_epoch', avg_mse[0], global_step=n_iter)
            self.writer.add_scalar('ft/edv_mse_epoch', avg_mse[1], global_step=n_iter)
            self.writer.add_scalar('ft/ef_mse_epoch', avg_mse[2], global_step=n_iter)

            # save model checkpoints
            checkpoint_name = "checkpoint_{:04d}.pth.tar".format(curr_epoch)
            curr_model_path = Path("models/finetuning/{date:%m-%d-%Y_%H:%M:%S}-{arch}checkpoints".format(date=self.args.date, arch=self.args.arch))
            curr_model_path.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': self.args.epochs,
                'model': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            # applicable to hf architectures
            # todo: create model class for simclr to req. self.backbone, self.head, and self.config so we don't need to have this safe guard
            if hasattr(self.model, "config"):
                checkpoint["config"] = self.model.backbone.config.to_dict()

            if (curr_epoch + 1) % self.args.save_every_n == 0 or curr_epoch >= self.args.epochs - self.args.save_last_n:
                save_checkpoint(checkpoint, is_best=False, filename=curr_model_path / checkpoint_name)
            logging.info(f"Model checkpoint {curr_epoch} and metadata has been saved at {curr_model_path}.")

            # validation loss
            self.model.eval()
            val_mae = torch.zeros((3,), device=self.device)

            self.r2_criterion.reset()

            with torch.no_grad():
                for image, label in val_loader:
                    image = image.to(self.device)
                    label = label.to(self.device)

                    pred = self.model(image)

                    self.r2_criterion.update(pred, label)

                    raw_mae = self.mae_criterion(pred, label)
                    val_mae += raw_mae.sum(dim=0)

            val_mae /= len(val_loader)
            val_r2_scores = self.r2_criterion.compute()

            logging.info(f"Epoch {curr_epoch} Val MAE: ESV: {val_mae[0]:.2f}, EDV: {val_mae[1]:.2f}, EF: {val_mae[2]:.2f}")
            logging.info(f"Epoch {curr_epoch} Val R2 : ESV: {val_r2_scores[0]:.2f}, EDV: {val_r2_scores[1]:.2f}, EF: {val_r2_scores[2]:.2f}")

            self.writer.add_scalar('ft/val/ESV_MAE', val_mae[0], global_step=curr_epoch)
            self.writer.add_scalar('ft/val/EDV_MAE', val_mae[1], global_step=curr_epoch)
            self.writer.add_scalar('ft/val/EF_MAE',  val_mae[2], global_step=curr_epoch)
            
            self.writer.add_scalar('ft/val/ESV_R2', val_r2_scores[0], global_step=curr_epoch)
            self.writer.add_scalar('ft/val/EDV_R2', val_r2_scores[1], global_step=curr_epoch)
            self.writer.add_scalar('ft/val/EF_R2',  val_r2_scores[2], global_step=curr_epoch)

        logging.info("Evaluating MSE and MAE on test set")

        test_mae_accum = torch.zeros((3,), device=self.device)
        test_mse_accum = torch.zeros((3,), device=self.device)

        self.r2_criterion.reset()

        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(self.device)
                label = label.to(self.device)

                pred = self.model(image)

                self.r2_criterion.update(pred, label)

                loss += self.finetuning_criterion(pred, label)

                raw_mae = self.mae_criterion(pred, label)
                raw_mse = self.mse_criterion(pred, label)

                test_mae_accum += raw_mae.sum(dim=0)
                test_mse_accum += raw_mse.sum(dim=0)

        avg_test_mae = test_mae_accum / len(test_loader)
        avg_test_mse = test_mse_accum / len(test_loader)

        test_r2_scores = self.r2_criterion.compute()

        logging.info(f"Final Test MAE: ESV: {avg_test_mae[0]:.4f}, EDV: {avg_test_mae[1]:.4f}, EF: {avg_test_mae[2]:.4f}")
        logging.info(f"Final Test R2 : ESV: {test_r2_scores[0]:.4f}, EDV: {test_r2_scores[1]:.4f}, EF: {test_r2_scores[2]:.4f}")
        logging.info(f"Final Test MSE: ESV: {avg_test_mse[0]:.4f}, EDV: {avg_test_mse[1]:.4f}, EF: {avg_test_mse[2]:.4f}")

        self.writer.add_scalar('eval/test/ESV_MAE', avg_test_mae[0], global_step=self.args.epochs)
        self.writer.add_scalar('eval/test/EDV_MAE', avg_test_mae[1], global_step=self.args.epochs)
        self.writer.add_scalar('eval/EF_MAE',  avg_test_mae[2], global_step=self.args.epochs)
        
        self.writer.add_scalar('eval/test/ESV_R2', test_r2_scores[0], global_step=self.args.epochs)
        self.writer.add_scalar('eval/test/EDV_R2', test_r2_scores[1], global_step=self.args.epochs)
        self.writer.add_scalar('eval/test/EF_R2',  test_r2_scores[2], global_step=self.args.epochs)
        
        self.writer.add_scalar('eval/test/ESV_MSE', avg_test_mse[0], global_step=self.args.epochs)
        self.writer.add_scalar('eval/test/EDV_MSE', avg_test_mse[1], global_step=self.args.epochs)
        self.writer.add_scalar('eval/test/EF_MSE',  avg_test_mse[2], global_step=self.args.epochs)

        logging.info("Fine-tuning finished.")