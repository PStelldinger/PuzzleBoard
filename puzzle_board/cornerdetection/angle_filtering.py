import numpy as np
from sklearn.neighbors import NearestNeighbors


def angle_filter(dot: np.ndarray, sub_dot: np.ndarray):
    """
    Cross Mask filter?
    :param dot:
    :param sub_dot:
    :return:
    """
    nr = dot.shape[0]
    old_nr = 0
    while nr != old_nr:
        if nr < 3:
            dot = np.array([])
            sub_dot = np.array([])
        else:
            model_knn = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(dot)
            distances, indices = model_knn.kneighbors(dot)
            v1 = dot - dot[indices[:, 1]]
            v2 = dot - dot[indices[:, 2]]
            angle = np.sum(v1 * v2, axis=1) / (distances[:, 1] * distances[:, 2])
            thresh = 0.4 * np.maximum(distances[:, 1] / distances[:, 2], distances[:, 2] / distances[:, 1])
            mask = angle < thresh
            dot = dot[mask]
            sub_dot = sub_dot[mask]

        old_nr = nr  # update old_nr
        nr = dot.shape[0]  # update nr
    return nr
