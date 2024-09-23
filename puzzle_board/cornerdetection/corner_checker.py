import cv2
import numpy as np


class CornerChecker:

    def __init__(self):
        self.outer_filter = [
            np.array([[-1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, -1]]),

            np.array([[0, -1, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0]]),

            np.array([[0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, -1, 0, 0]])
        ]
        self.inner_filter = [
            np.array([[-1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1]]),

            np.array([[0, -1, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0]]),

            np.array([[0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0]]),

            np.array([[0, 0, 0, -1, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]]),

            np.array([[0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0]]),

            np.array([[0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]]),

            np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]),

            np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
        ]
        self.p = 1

    def filter_corners(self, image: np.ndarray) -> np.ndarray:
        results1 = [cv2.filter2D(image, -1, f) for f in self.outer_filter]
        outer_filters_concatenated = np.stack(results1, axis=-1)

        # Taking max along the 3rd axis and multiplying by p/2
        max_result1 = self.p * np.max(np.abs(outer_filters_concatenated), axis=-1) / 2
        results2 = [cv2.filter2D(image, -1, f) for f in self.inner_filter]
        inner_filters_concatenated = np.stack(results2, axis=-1)
        max_result2 = np.max(np.abs(inner_filters_concatenated), axis=-1)  # Taking max along the 3rd axis

        final_result = np.maximum(image * 0, max_result1 - max_result2)
        #print(np.min(final_result))
        #print(np.max(final_result))
        return final_result
