from typing import Tuple, Optional
import numpy as np


def detect_outliers_and_centered_points(
    points: np.ndarray,
    num_samples: Optional[int] = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers and most centered points in a set of 2D points.

    Parameters:
        points (np.ndarray): Array of 2D points in the shape (n, 2).
        num_samples (Optional[int]): Number of most centered points to return.
            Default is 2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
            1. Array of outlier point indices.
            2. Array of most centered point indices.
    """
    # Compute mean and standard deviation for each dimension
    mean_x, std_x = np.mean(points[:, 0]), np.std(points[:, 0])
    mean_y, std_y = np.mean(points[:, 1]), np.std(points[:, 1])

    # Calculate Z-scores for each dimension
    z_scores_x = (points[:, 0] - mean_x) / std_x
    z_scores_y = (points[:, 1] - mean_y) / std_y

    # Compute overall Z-score (euclidean distance)
    z_scores = np.sqrt(z_scores_x**2 + z_scores_y**2)

    # Identify outliers
    sorted_indices = np.argsort(z_scores)
    outlier_idxs = sorted_indices[-num_samples:]

    # Calculate distance of each point from the centroid
    centroid_x, centroid_y = np.mean(points[:, 0]), np.mean(points[:, 1])
    distances = np.sqrt((points[:, 0] - centroid_x)**2 +
                        (points[:, 1] - centroid_y)**2)

    # Find indices of points furthest from the centroid
    sorted_indices = np.argsort(distances)

    # Return the specified number of most centered points
    most_centered_idxs = sorted_indices[:num_samples]

    return outlier_idxs, most_centered_idxs
