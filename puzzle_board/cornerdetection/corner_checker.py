import cv2
import numpy as np


class CornerChecker:

    def __init__(self, dtype=np.float32):

        self.dtype = dtype

        self.outer_filter = [
            np.array([[-1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, -1]], dtype=dtype),

            np.array([[0, -1, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [0, 0, 0, -1, 0]], dtype=dtype),

            np.array([[0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, -1, 0, 0]], dtype=dtype)
        ]
        self.inner_filter = [
            np.array([[-1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1]], dtype=dtype),

            np.array([[0, -1, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0]], dtype=dtype),

            np.array([[0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0]], dtype=dtype),

            np.array([[0, 0, 0, -1, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]], dtype=dtype),

            np.array([[0, 0, 0, 0, -1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0]], dtype=dtype),

            np.array([[0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0]], dtype=dtype),

            np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]], dtype=dtype),

            np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [-1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]], dtype=dtype)
        ]
        self.p = 1

        # placeholder for memory allocation
        self.outer_filters_array = None
        self.inner_filters_array = None

    def allocate_memory(self, image_size):
        self.outer_filters_array = np.empty((len(self.outer_filter), *image_size), dtype=self.dtype)
        self.inner_filters_array = np.empty((len(self.inner_filter), *image_size), dtype=self.dtype)

    def filter_corners(self, image: np.ndarray) -> np.ndarray:

        # reserve memory for the filtered images
        if self.outer_filters_array is None or self.outer_filters_array.shape[:2] != image.shape[:2]:
            self.allocate_memory(image.shape[:2])

        # filter the image with outer and inner filters
        for i, f in enumerate(self.outer_filter):
            cv2.filter2D(image, -1, f, dst=self.outer_filters_array[i])
        for i, f in enumerate(self.inner_filter):
            cv2.filter2D(image, -1, f, dst=self.inner_filters_array[i])

        # Taking max along axis=0 and multiplying by p/2
        max_result1 = self.p * np.max(np.abs(self.outer_filters_array), axis=0) / 2
        max_result2 = np.max(np.abs(self.inner_filters_array), axis=0)

        final_result = np.maximum(0, max_result1 - max_result2)
        # print(np.min(final_result))
        # print(np.max(final_result))
        return final_result

