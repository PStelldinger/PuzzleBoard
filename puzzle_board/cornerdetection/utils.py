import cv2
import numpy as np


def get_eigenvectors_at_maxima(image_with_first_eigenvectors, image_with_second_eigenvectors, dot):
    eigenvectors_at_max = np.array(
        [(image_with_first_eigenvectors[dot[idx, 0], dot[idx, 1]],
          image_with_second_eigenvectors[dot[idx, 0], dot[idx, 1]]) for idx in range(len(dot))])
    eigenvectors_at_max = eigenvectors_at_max[:] / np.linalg.norm(eigenvectors_at_max, axis=1).reshape(-1, 1)
    return eigenvectors_at_max


def get_sub_pix_brightness(video_frame_gray, p0):
    # Convert the points to integer and get the decimal part
    p = np.floor(p0).astype(int)
    dp = p0 - p

    # Ensure indices are within bounds
    p[:, 0] = np.clip(p[:, 0], 0, video_frame_gray.shape[1] - 2)
    p[:, 1] = np.clip(p[:, 1], 0, video_frame_gray.shape[0] - 2)

    # Extract the pixel values from the grayscale image
    g00 = video_frame_gray[p[:, 1], p[:, 0]]
    g01 = video_frame_gray[p[:, 1], p[:, 0] + 1]
    g10 = video_frame_gray[p[:, 1] + 1, p[:, 0]]
    g11 = video_frame_gray[p[:, 1] + 1, p[:, 0] + 1]

    # Perform bilinear interpolation
    dgx0 = g00 + (g01 - g00) * dp[:, 0]
    dgx1 = g10 + (g11 - g10) * dp[:, 0]
    ret = dgx0 + (dgx1 - dgx0) * dp[:, 1]

    return ret


def get_sub_pix_brightness_single_point(video_frame_gray, p0):
    # Convert the points to integer and get the decimal part
    p = np.floor(p0).astype(int)
    dp = p0 - p

    # Extract the pixel values from the grayscale image
    g00 = video_frame_gray[p[1], p[0]]
    g01 = video_frame_gray[p[1], p[0] + 1]
    g10 = video_frame_gray[p[1] + 1, p[0]]
    g11 = video_frame_gray[p[1] + 1, p[0] + 1]

    # Perform bilinear interpolation
    dgx0 = g00 + (g01 - g00) * dp[0]
    dgx1 = g10 + (g11 - g10) * dp[0]
    ret = dgx0 + (dgx1 - dgx0) * dp[1]

    return ret


def create_average_filtered_image(input_image, filter_size):
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size ** 2)
    return cv2.filter2D(input_image, -1, kernel)
