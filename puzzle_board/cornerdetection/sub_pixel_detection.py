import numpy as np
import cv2


def get_subpixel_positions(profile: np.ndarray, mS: np.ndarray, dot: np.ndarray) -> np.ndarray:

    dtype = profile.dtype

    # TODO: BUG? xs/ys are the the other way arround
    ys, xs = np.meshgrid(range(0, np.shape(mS)[1]), range(0, np.shape(mS)[0]))
    ys, xs = ys.astype(dtype), xs.astype(dtype)
    #print(f'xs={xs}, shape={xs.shape}')
    #print(f'ys={ys}, shape={ys.shape}')

    # Make all negative values in profile to zero (assuming profile is already defined)
    profile = np.where(profile > 0, profile, 0)

    ones_kernel = np.ones([3, 3], dtype=dtype)
    sub_denominator = cv2.filter2D(profile, -1, ones_kernel) + 0.00000000000001

    sub_x = np.divide(cv2.filter2D(xs * profile, -1, ones_kernel), sub_denominator)
    sub_y = np.divide(cv2.filter2D(ys * profile, -1, ones_kernel), sub_denominator)

    # TODO: BUG? dot is already in 0-based indexin convention!?
    # Convert 2D subscripts in dot to linear indices
    indices = np.ravel_multi_index((dot[:, 0] - 1, dot[:, 1] - 1), mS.shape)  # subtracting 1 because Python uses 0-based indexing

    # Fetch the values from subx and suby using the indices
    sub_dot = np.column_stack((sub_x.ravel()[indices], sub_y.ravel()[indices]))

    return sub_dot


def get_subpixel_positions_new(profile: np.ndarray, dot: np.ndarray) -> np.ndarray:

    dtype = profile.dtype

    # gather a 3x3 square around each dot
    ys, xs = np.meshgrid(range(-1, 2), range(-1, 2), indexing='ij')
    dot_ys = dot[..., 0:1] + ys.flatten()
    dot_xs = dot[..., 1:2] + xs.flatten()
    # clip indices to avoid out of image sampling
    dot_ys = np.clip(dot_ys, 0, profile.shape[0]-1)
    dot_xs = np.clip(dot_xs, 0, profile.shape[1]-1)
    # take dot profile values
    profile_batches = profile[dot_ys, dot_xs]

    # take the profile weighted mean dot
    sub_denominator = np.sum(profile_batches, axis=-1)
    sub_y = np.sum(dot_ys * profile_batches, axis=-1) / sub_denominator
    sub_x = np.sum(dot_xs * profile_batches, axis=-1) / sub_denominator
    sub_dot = np.stack([sub_y, sub_x], axis=-1)

    return sub_dot

