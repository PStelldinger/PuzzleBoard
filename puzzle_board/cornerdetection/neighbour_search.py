import numpy as np


def find_neighbors_according_to_angle(orientation_diff_angle, idx_neighbors):
    """
    Looks for orientation of eigenvectors from neighbors.
    :param orientation_diff_angle:
    :param idx_neighbors:
    :return: copied idx_neighbors with all not eligible neighbors, due to orientation diff too huge, set to -1
    """
    idx_neighbors_with_mask = np.copy(idx_neighbors)
    for row, col in np.ndindex((idx_neighbors_with_mask.shape[0], idx_neighbors_with_mask.shape[1])):
        if col >= orientation_diff_angle.shape[1]:
            continue
        if orientation_diff_angle[row, col] == 0:
            idx_neighbors_with_mask[row, col + 1] = -1

    return idx_neighbors_with_mask
