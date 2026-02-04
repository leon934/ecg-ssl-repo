import math

import torch
from timm.optim import Lars

def get_optimizer(model, train_loader_len, args):
    total_steps = args.epochs * train_loader_len
    warmup_epochs = int(0.10 * total_steps)

    if args.optimizer == "lars":
        peak_lr = 0.3 * args.batch_size / 256 if args.batch_size >= 512 else 0.075 * math.sqrt(args.batch_size)

        optimizer = Lars(
            model.parameters(),
            lr=peak_lr,
            weight_decay=1e-6,
            momentum=0.9
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "nesterov":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.05 * args.batch_size / 256, 
            weight_decay=args.weight_decay,
            momentum=0.9,
            nesterov=True
        )

    def lr_lambda(current_step):
        if current_step < warmup_epochs:
            return float(current_step) / float(max(1, warmup_epochs))
        
        progress = float(current_step - warmup_epochs) / float(max(1, total_steps - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler