import argparse
import numpy as np
import torch
import torch.nn.functional as F
from src.util import load_model
from data.tumor_data import load_testing_data
from src.metrics import compute_metrics

def test(
    model_name: str = "linear",
    exp_dir: str = "logs",
    batch_size: int = 64,
    seed: int = 2024,
    num_classes: int = 4,  # Specify the number of classes
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

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load the model
    model = load_model(model_name, with_weights=True, **kwargs)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Load testing data
    test_data_sources = ["csml_final/data/dataset1", "csml_final/data/dataset2"]
    test_data = load_testing_data(dataset_paths=test_data_sources, batch_size=batch_size, num_workers=0)

    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for img, label in test_data:
            img, label = img.to(device), label.to(device)

            # Forward pass
            logits_pred = model(img)

            # Store logits and labels for metric computation
            all_logits.append(logits_pred)
            all_labels.append(label)

    # Concatenate all logits and labels
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_logits, all_labels, num_classes)

    # Print results
    print(f"Test Results: "
          f"Accuracy: {metrics['accuracy']:.4f}, "
          f"Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, "
          f"F1 Score: {metrics['f1_score']:.4f}")

    # Print confusion matrix
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)

    # Pass all arguments to test
    test(**vars(parser.parse_args()))
