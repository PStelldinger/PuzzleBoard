import numpy as np
import cv2


def get_subpixel_positions(profile: np.ndarray, mS: np.ndarray, dot: np.ndarray) -> np.ndarray:
    ys, xs = np.meshgrid(range(0, np.shape(mS)[1]), range(0, np.shape(mS)[0]))
    #print(f'xs={xs}, shape={xs.shape}')
    #print(f'ys={ys}, shape={ys.shape}')

    # Make all negative values in profile to zero (assuming profile is already defined)
    profile = np.where(profile > 0, profile, 0)
    profile = np.float32(profile)

    ones_kernel = np.ones([3, 3], dtype=np.float32)
    sub_denominator = cv2.filter2D(profile, -1, ones_kernel) + 0.00000000000001

    sub_x = np.divide(cv2.filter2D(xs * profile, -1, ones_kernel), sub_denominator)
    sub_y = np.divide(cv2.filter2D(ys * profile, -1, ones_kernel), sub_denominator)

    # Convert 2D subscripts in dot to linear indices
    indices = np.ravel_multi_index((dot[:, 0] - 1, dot[:, 1] - 1),
                                   mS.shape)  # subtracting 1 because Python uses 0-based indexing

    # Fetch the values from subx and suby using the indices
    sub_dot = np.column_stack((sub_x.ravel()[indices], sub_y.ravel()[indices]))

    return sub_dot
