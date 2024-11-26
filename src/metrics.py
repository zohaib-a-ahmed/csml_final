import torch
import torch.nn.functional as F

def calculate_accuracy(predictions, labels):
    """Calculate accuracy."""
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def compute_confusion_matrix(predictions, labels, num_classes):
    """Compute the confusion matrix."""
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    return conf_matrix

def calculate_precision(conf_matrix):
    """Calculate precision for each class."""
    num_classes = conf_matrix.size(0)
    precision = torch.zeros(num_classes)
    for i in range(num_classes):
        tp = conf_matrix[i, i].item()  # True positives
        fp = conf_matrix[:, i].sum().item() - tp  # False positives
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
    return precision

def calculate_recall(conf_matrix):
    """Calculate recall for each class."""
    num_classes = conf_matrix.size(0)
    recall = torch.zeros(num_classes)
    for i in range(num_classes):
        tp = conf_matrix[i, i].item()  # True positives
        fn = conf_matrix[i, :].sum().item() - tp  # False negatives
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall

def calculate_f1_score(precision, recall):
    """Calculate F1 score for each class."""
    num_classes = precision.size(0)
    f1 = torch.zeros(num_classes)
    for i in range(num_classes):
        f1[i] = (2 * precision[i] * recall[i] / (precision[i] + recall[i])) if (precision[i] + recall[i]) > 0 else 0
    return f1

def compute_metrics(logits, labels, num_classes):
    """Compute all metrics and return them in a dictionary."""
    # Ensure logits and labels are of shape [batch_size, num_classes] and [batch_size]
    assert logits.shape == (labels.size(0), num_classes), "Logits must have shape [batch_size, num_classes]"
    assert labels.shape == (labels.size(0),), "Labels must have shape [batch_size]"

    # Convert logits to probabilities using softmax
    probabilities = F.softmax(logits, dim=1)

    # Get predicted class indices
    predictions = torch.argmax(probabilities, dim=1)  # Shape: [batch_size]

    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, labels)

    # Compute confusion matrix
    conf_matrix = compute_confusion_matrix(predictions, labels, num_classes)

    # Calculate precision and recall
    precision = calculate_precision(conf_matrix)
    recall = calculate_recall(conf_matrix)

    # Calculate F1 score
    f1 = calculate_f1_score(precision, recall)

    # Calculate macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    # Create a dictionary to hold the metrics
    metrics = {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


if __name__ == "__main__":
    # Example logits output from the model for a batch of 3 instances
    logits = torch.tensor([[2.0, 0.5, 1.0, 0.1],   # Instance 1
                        [0.1, 1.5, 0.3, 0.4],   # Instance 2
                        [0.3, 0.2, 1.2, 0.5]])  # Instance 3

    # Corresponding labels for the instances (class indices)
    labels = torch.tensor([3, 1, 2])  # Instance 1 -> class 3, Instance 2 -> class 1, Instance 3 -> class 2

    # Number of classes
    num_classes = 4

    # Compute metrics
    metrics = compute_metrics(logits, labels, num_classes)
    print(metrics)

