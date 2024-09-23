from typing import Any
import cv2
import numpy as np
from numpy import ndarray, dtype, bool_
from skimage.feature import peak_local_max


def image_regional_max_as_binary_matrix(max_s_one_matrix):
    # Convert the data type to float32 before processing
    max_s_one_matrix = max_s_one_matrix.astype(np.float32)
    #print(f'shape of mS={max_s_one_matrix.shape}')
    # Meeting on October 20th: Outcome -> usage of scikits "peak_local_max"
    maximum_positions = peak_local_max(max_s_one_matrix, min_distance=5)
    maximum_positions = np.array(maximum_positions)
    #print(f'shape of maximum_positions = {maximum_positions.shape}')
    max_s_one_matrix = max_s_one_matrix * 0
    for pos in maximum_positions:
    #    print(f'pos: {pos}')
        max_s_one_matrix[pos[0], pos[1]] = 1

    #print(f'max_s_one_matrix after finding maxima ->\n{max_s_one_matrix}')
    return max_s_one_matrix


def _apply_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, kernel)


def _create_derivative(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _apply_filter(in_image, kernel)


class HessianDetector:

    def __init__(self, image: np.ndarray, epsilon=0.03):
        # already smoothed, so we can ignore using the actual sobel-operator but instead using the
        # discrete derivatives
        self.y_derivative_filter = np.array([[-1], [0], [1]])
        self.x_derivative_filter = np.array([[-1, 0, 1]])

        self.diagonal_kernel_one = np.array([[0, 0, 1],
                                             [0, 0, 0],
                                             [-1, 0, 0]])
        self.diagonal_kernel_two = np.array([[1, 0, 0],
                                             [0, 0, 0],
                                             [0, 0, -1]])
        self.epsilon = epsilon
        self.f_y = _create_derivative(image, self.y_derivative_filter)
        self.f_yy = _create_derivative(self.f_y, self.y_derivative_filter)
        self.f_x = _create_derivative(image, self.x_derivative_filter)
        self.f_xx = _create_derivative(self.f_x, self.x_derivative_filter)
        self.f_xy = _create_derivative(self.f_y, self.x_derivative_filter)
        self.f_rl = _apply_filter(image, self.diagonal_kernel_one)
        self.f_rl_rl = _apply_filter(self.f_rl,
                                          self.diagonal_kernel_one)

        self.f_lr = _apply_filter(image, self.diagonal_kernel_two)
        self.f_lr_lr = _apply_filter(self.f_lr,
                                          self.diagonal_kernel_two)

        self.f_rl_lr = _apply_filter(self.f_rl,
                                          self.diagonal_kernel_two)

    def _detect_corners_xy(self, image: np.ndarray, k=1) -> np.ndarray:
        # S=gbb.*gdd-gbd.^2+(gbb+gdd).^2;
        # S from paper Liu et al.: S = det(H) = r_xx * r_yy - (r_xy)^2
        dt = self.f_yy * self.f_xx - (self.f_xy ** 2)
        tr = self.f_yy + self.f_xx
        s = dt + k * (tr**2)

        max_s_one = -s / np.max(-s)
#        qv = 2*dt / (2*dt -tr**2-0.00000000001)
#        qv[qv<0.9]=0.
#        print(np.max(qv))
#        max_s_one = max_s_one * qv

        #print(f'max_s_one in detect_corners {max_s_one}')
        #print(f'max_s_one in detect_corners maximum value= {np.max(max_s_one)}')
        # eliminates every value in the "max_s_one" that is below the threshold (epsilon = 0.03)
        max_s_one[max_s_one < self.epsilon] = 0.0
        #print(f'max_s_one in detect_corners after normalization {max_s_one}')
        return max_s_one

    def _detect_corners_diagonal(self, image: np.ndarray, k=1) -> np.ndarray:
        # S = gaa.*gcc-gac.^2+(gaa+gcc).^2;
        s = ((self.f_lr_lr
              * self.f_rl_rl)
             - self.f_rl_lr ** 2
             + (k * ((self.f_rl_rl + self.f_lr_lr) ** 2)))

        max_s_two = -s / np.max(-s)
        max_s_two[max_s_two < self.epsilon] = 0.0
        return max_s_two

    def detect_corners(self, image: np.ndarray, k=1):
        detected_corners_xy = self._detect_corners_xy(image, k)
        detected_corners_diagonal = self._detect_corners_diagonal(image, k)
        profile = detected_corners_xy + detected_corners_diagonal
        mS = detected_corners_xy * detected_corners_diagonal
        return profile, mS
