import torch
import matplotlib.pyplot as plt

def binary_classifier_area_growth(initial_areas: torch.Tensor, target_areas: torch.Tensor, threshold, show_hist=False) -> torch.Tensor:
    """
    Create binary classifier on if tumor growed based on area comparison. Take difference
    in areas between initial and target images. Any delta area greater than a given threshold will be classified as tumor growth. All input images did have tumor growth

    Provide some data on the distribution of area differences like standard deviation and mean. visualize with histogram.
     
    Args:
        initial_areas (torch.Tensor): Tensor of shape (N,) representing areas from initial images.
        target_areas (torch.Tensor): Tensor of shape (N,) representing areas from target images.
    """

    # Calculate area differences
    area_differences = target_areas - initial_areas

    # Compute mean and standard deviation of area differences
    mean_diff = area_differences.mean()
    std_diff = area_differences.std()
    print(f"Mean area difference: {mean_diff.item()}, Standard Deviation: {std_diff.item()}")   

    # Visualize distribution of area differences
    plt.hist(area_differences.numpy(), bins=30, alpha=0.7, color='blue')
    plt.axvline(mean_diff.item(), color='red', linestyle='dashed', label='Mean')
    plt.axvline((mean_diff - std_diff).item(), color='green', linestyle='dashed', label='Mean - 1 Std Dev')
    plt.title('Distribution of Area Differences')
    plt.xlabel('Area Difference')
    plt.ylabel('Frequency')
    plt.legend()
    if show_hist:
        plt.show()  

    # # Create binary classifier based on threshold (mean - 1 std dev) possible alternative
    # threshold = mean_diff - std_diff
    binary_classification = (area_differences > threshold).float()

    # Print classification results
    num_growth = binary_classification.sum().item()
    num_no_growth = (binary_classification == 0).sum().item()
    print(f"\nClassification Results (threshold: {threshold}):")
    print(f"Classified as growth: {num_growth}/{len(binary_classification)}")
    print(f"Classified as no growth: {num_no_growth}/{len(binary_classification)}")

    return binary_classification


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute the Dice Score between predicted and target tensors (each are masks of 1's and 0's).

    Args:
        pred (torch.Tensor): Predicted tensor of shape (N, C, H, W) or (N, H, W).
            where N is batch size, C is number of classes, H and W are height and width.
        target (torch.Tensor): Ground truth tensor of the same shape as pred.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice Score.
    """

    # Flatten the tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    # Calculate number of true positives
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return dice.item()

def f1_score(binaryPreds: torch.Tensor, binaryTargets: torch.Tensor) -> float:
    """
    Compute the F1 Score between binary predictions and binary targets.

    Args:
        binaryPreds (torch.Tensor): Binary predictions tensor of shape (N) for N samples.
        binaryTargets (torch.Tensor): Binary ground truth tensor of the same shape as binaryPreds.

    Returns:
        float: F1 Score.
    """

    # Calculate true positives, false positives, and false negatives
    true_positives = (binaryPreds * binaryTargets).sum().item()
    false_positives = (binaryPreds * (1 - binaryTargets)).sum().item()
    false_negatives = ((1 - binaryPreds) * binaryTargets).sum().item()

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return f1

