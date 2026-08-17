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

        self.outer_filters_array = None
        self.inner_filters_array = None

    def filter_corners(self, image: np.ndarray) -> np.ndarray:

        # reserve memory for the filtered images
        if self.outer_filters_array is None or self.outer_filters_array.shape[:2] != image.shape[:2]:
            self.outer_filters_array = np.zeros((len(self.outer_filter), *image.shape[:2]))
        if self.inner_filters_array is None or self.inner_filters_array.shape[:2] != image.shape[:2]:
            self.inner_filters_array = np.zeros((len(self.inner_filter), *image.shape[:2]))

        for i, f in enumerate(self.outer_filter):
            cv2.filter2D(image, -1, f, dst=self.outer_filters_array[i])

        for i, f in enumerate(self.inner_filter):
            cv2.filter2D(image, -1, f, dst=self.inner_filters_array[i])

        # Taking max along axis=0 and multiplying by p/2
        max_result1 = self.p * np.max(np.abs(self.outer_filters_array), axis=0) / 2
        max_result2 = np.max(np.abs(self.inner_filters_array), axis=0)

        final_result = np.maximum(image * 0, max_result1 - max_result2)
        # print(np.min(final_result))
        # print(np.max(final_result))
        return final_result

