import numpy as np

from cornerdetection.utils import get_sub_pix_brightness_single_point


def get_distance(coords, other_coords):
    point1 = np.array((coords[0], coords[1]))
    point2 = np.array((other_coords[0], other_coords[1]))
    euclidean_distance = np.linalg.norm(point1 - point2)
    return euclidean_distance


def get_weight_based_on_first_sixth_and_last_sixth(coords, other_coords, image_grayscale):
    # this threshold must be set in correspondance of the detected image. set it to 0.3 for the super fisheye image
    threshold = 0.3
    point1 = np.array((coords[1], coords[0]))
    point2 = np.array((other_coords[1], other_coords[0]))
    line = np.linspace(point1, point2, num=15, endpoint=True)
    first_sixth = line[:5]
    last_sixth = line[-5:]
    grayscale_values_line = []
    grayscale_values_first_sixth = []
    grayscale_values_last_sixth = []
    for point in first_sixth:
        grayscale_values_first_sixth.append(get_sub_pix_brightness_single_point(image_grayscale, point))
    for point in last_sixth:
        grayscale_values_last_sixth.append(get_sub_pix_brightness_single_point(image_grayscale, point))
    for point in line:
        grayscale_values_line.append(get_sub_pix_brightness_single_point(image_grayscale, point))

    first_sixth_values = np.asarray(grayscale_values_first_sixth)
    last_sixth_values = np.asarray(grayscale_values_last_sixth)
    line_values = np.asarray(grayscale_values_line)
    first_sixth_values.min()
    first_sixth_values.max()
    last_sixth_values.min()
    last_sixth_values.max()
    first_sixth_avg = np.mean(first_sixth_values)
    last_sixth_avg = np.mean(last_sixth_values)
    line_min = line_values.min()
    line_max = line_values.max()
    line_avg = np.mean(line_values)

    # returns true, if the edge is faulty
    #
    # plt.imshow(image_grayscale, cmap='gray')
    # plt.scatter(coords[1], coords[0])
    # plt.scatter(other_coords[1], other_coords[0])
    # threshold = ((line_max + line_min) / 7)
    threshold = ((line_avg) / 2)
    diff = abs(first_sixth_avg - last_sixth_avg)
    # if diff > threshold:
    #     plt.plot([coords[1], other_coords[1]], [coords[0], other_coords[0]], color="red")
    # else:
    #     plt.plot([coords[1], other_coords[1]], [coords[0], other_coords[0]], color="green")
    # plt.title(f"min:{line_min}, max:{line_max}\n, t:{threshold}, true:{diff}")
    # plt.show()
    return diff > threshold



def calc_edge_weight(node_coords, other_node_coords, image):
    # for now: just distance as a measure:
    distance = get_distance(node_coords, other_node_coords)
    if get_weight_based_on_first_sixth_and_last_sixth(node_coords, other_node_coords, image):
        distance = distance * 3
    return distance
