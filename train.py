import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.optim.lr_scheduler import StepLR

from src.util import load_model, save_model
from data.tumor_data import load_data
from src.metrics import *

def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("GPU not available, using MPS")
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    data_sources = ["data/dataset1", "data/dataset2"]
    train_data, val_data, test_data = load_data(dataset_paths=data_sources, shuffle=True, batch_size=batch_size, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=.8)
    loss = torch.nn.CrossEntropyLoss()

    global_step = 0
    metrics = {'train_acc': [], 'val_acc': []}

    # training loops
    for epoch in range(num_epoch):

        # clear metrics at beginning of epoch
        metrics['acc'] = []

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logits_pred = model(img)

            loss_val = loss(logits_pred, label)
        
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Calculate training metrics
            # Get predicted class indices
            predictions = torch.argmax(logits_pred, dim=1)  # Shape: [batch_size]
            accuracy = calculate_accuracy(predictions, label)  # Calculate accuracy
            metrics['train_acc'].append(accuracy)  # Store accuracy for this batch

            global_step += 1
        
        scheduler.step()

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                val_pred = model(img)
                loss_val = loss(val_pred, label)
                
                # Calculate validation accuracy
                val_metrics = compute_metrics(val_pred, label, num_classes=4)  # Assuming 4 classes
                metrics['val_acc'].append(val_metrics['accuracy'])  # Store validation accuracy


        # Print metrics for the first, last, and every 5th epoch for accuracy and loss
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            train_accuracy = np.mean(metrics['train_acc'])  # Average training accuracy for the epoch
            train_accuracy = np.mean(metrics['val_acc'])  # Average training accuracy for the epoch
            print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                  f"Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"Val Loss: {loss_val / len(val_data):.4f}")


    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
