import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

import neighbour_search
import plotting.hessian_plotting
import sub_pixel_detection
import utils
from corner_checker import CornerChecker
from cornerdetection.preprocessor import read_image_as_normalized_grayscale
from grid.kruskal.old.kruskal_alg import KruskalAlgorithm
from grid.kruskal.optimized.decoder import decode_horizontal_lines, decode_vertical_lines
from grid.kruskal.optimized.optimized_kruskal import OptimizedKruskalAlgorithm
from hessian_detector import HessianDetector, image_regional_max_as_binary_matrix

experimental = False
debug = False

# y-values => row
# x-values => col

if __name__ == "__main__":
    image_grayscale = read_image_as_normalized_grayscale(
#        "c:\\Users\\Stelldinger\\Documents\\MATLAB\\puzzleboardImage10.png")
        "C:\\Users\\Stelldinger\\Documents\\Projekte\\PuzzleBoardDetection\\CalibrationPatternImages\\Target_1a.png")
    # apply gaussian blur to the grayscale image
    image_grayscale = image_grayscale[1200:1800,1200:1400]
    start_gauss = time.time()
    # image_ary = image_grayscale
    # image_grayscale = image = np.array([[1.0, 1.0,1.0,0, 0, 0],
    #                               [1.0, 1.0, 1.0, 0, 0, 0],
    #                               [1.0, 1.0, 1.0, 0, 0, 0],
    #                               [0, 0, 0, 1.0, 1.0, 1.0],
    #                               [0, 0, 0, 1.0, 1.0, 1.0],
    #                               [0, 0, 0, 1.0, 1.0, 1.0],
    #                               ])
    image_ary = cv2.GaussianBlur(image_grayscale, (5, 5), 1)
    end_gauss = time.time()
    print(f"Runtime gaussian blur in ms: {(end_gauss-start_gauss) * 1000}")
    average_filtered_image = utils.create_average_filtered_image(image_grayscale, 5)
    plt.imshow(average_filtered_image, cmap="gray")
    plt.show()
    start = time.time()
    h_d = HessianDetector(image_ary)
    if debug:
        plotting.hessian_plotting.plot_gradients(h_d, image_ary)

    if debug:
        sobel_derivative = cv2.Sobel(image_grayscale, cv2.CV_64F, 1, 0, ksize=5)
        plt.imshow(sobel_derivative, cmap="gray")
        plt.title("sobel first derivative")
        plt.show()
    corner_checker = CornerChecker()

    profile, mS = h_d.detect_corners(image_ary, k=0.05)
    if debug:
        print(f"{mS}")
        plt.imshow(image_grayscale, cmap="grey")
        plt.imshow(mS, cmap="hot", alpha=0.5)
        plt.title("Hessian Detector Response")
        plt.colorbar(label="Response intensity")
        plt.show()
    mS_corner = corner_checker.filter_corners(image_ary)
    mS_corner = np.float32(mS_corner)

    mS_corner_binary = np.where(mS_corner > 0, 1, 0)

    mS = mS * mS_corner_binary
    mS = mS / np.nanmax(mS + 0.00000001)
    mS[mS < 0.03] = 0.0
    mS = image_regional_max_as_binary_matrix(mS) * mS
    if debug:
        print(f"{mS}")
        plt.imshow(image_grayscale, cmap="grey")
        plt.imshow(mS, cmap="hot", alpha=mS)
        plt.show()

    dot_row, dot_col = np.where(mS > 0)
    if debug:
        plt.imshow(image_grayscale, cmap="grey")
        plt.scatter(dot_col, dot_row)
        plt.show()
    # Creating dot array and filtering it
    # Every row in "dot" are the coordinates of the dot
    dot = np.column_stack((dot_row, dot_col))
    if debug:
        print(f"Found corners before refinement: {dot.shape[0]}")
    dot = dot[dot[:, 0] > 3]
    dot = dot[dot[:, 1] > 3]
    dot = dot[dot[:, 0] <= image_grayscale.shape[0] - 3]
    dot = dot[dot[:, 1] <= image_grayscale.shape[1] - 3]

    sub_dot = sub_pixel_detection.get_subpixel_positions(profile, mS, dot)
    end = time.time()
    print(f"corner detection to sub_pixel accuracy runtime in ms: {(end - start) * 1000}")
    # number_rows = angle_filter(dot, sub_dot)

    dot_row, dot_col = np.split(sub_dot, 2, axis=1)

    if debug:
        plt.imshow(image_grayscale, cmap='gray', vmin=0)
        plt.scatter(dot_col, dot_row, c='r', s=1)
        plt.show()

    # calculation for eigenvectors:  [(rxx - ryy - sqrtf((rxx - ryy) * (rxx - ryy) + 4 * rxy * rxy)), 2 * rxy]
    first_eigenvector_x = h_d.f_xx - h_d.f_yy + np.sqrt(
        (h_d.f_xx - h_d.f_yy) * (h_d.f_xx - h_d.f_yy) + 4 * h_d.f_xy ** 2)
    first_eigenvector_y = 2 * h_d.f_xy

    second_eigenvector_x = h_d.f_xx - h_d.f_yy - np.sqrt(
        (h_d.f_xx - h_d.f_yy) * (h_d.f_xx - h_d.f_yy) + 4 * h_d.f_xy ** 2)
    second_eigenvector_y = 2 * h_d.f_xy

    first_eigenvector_at_max = utils.get_eigenvectors_at_maxima(first_eigenvector_x, first_eigenvector_y, dot)
    second_eigenvector_at_max = utils.get_eigenvectors_at_maxima(second_eigenvector_x, second_eigenvector_y, dot)

    if debug:
        plotting.hessian_plotting.plot_eigenvectors(h_d, dot, image_ary)

    # ============ Neighbor block =============

    # Orientation-Berechnung
    # orientation = np.arctan2(np.real((h_d.f_rl_rl - h_d.f_lr_lr - np.sqrt(h_d.f_rl_rl ** 2 - 2 * h_d.f_rl_rl *
    #                                                                      h_d.f_lr_lr + h_d.f_lr_lr ** 2
    #                                                                      + 4 * h_d.f_rl_lr ** 2))), 2 * h_d.f_rl_lr)

    orientation = np.arctan2(first_eigenvector_x, first_eigenvector_y)

    print(f"orientation_max: {np.unravel_index(orientation.argmax(), orientation.shape)}")
    print(f"orientation_min: {np.unravel_index(orientation.argmin(), orientation.shape)}")

    NUMBER_WANTED_NEIGHBORS = 9
    if sub_dot.shape[0] >= NUMBER_WANTED_NEIGHBORS:
        # Find the 8 nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=NUMBER_WANTED_NEIGHBORS, algorithm='auto').fit(sub_dot)
        distances, idx_neighbors = nbrs.kneighbors(sub_dot)

        orientation_diff_angle = []
        for i in range(1, NUMBER_WANTED_NEIGHBORS):
            # Calculate orientation difference
            orientation_diff = np.abs(np.sin((orientation[dot[:, 0], dot[:, 1]] -
                                                     orientation[
                                                         dot[idx_neighbors[:, i], 0], dot[idx_neighbors[:, i], 1]])))
            orientation_diff_angle.append(orientation_diff)

        orientation_diff_angle = np.column_stack(orientation_diff_angle)
        neighb_mask = orientation_diff_angle > 0.707  #0.707 => 45°
        idx_neighbors_with_mask = neighbour_search.find_neighbors_according_to_angle(neighb_mask, idx_neighbors)

        # Find the nearest neighbor with appropriate orientation
        nearest_neighbor = np.sum(np.cumprod(1 - neighb_mask, axis=1), axis=1) + 1
        nearest_neighbor2 = np.where(neighb_mask[:, 1] == 0)
        nearest_neighbor = np.mod(nearest_neighbor, NUMBER_WANTED_NEIGHBORS) + 1  # Adjusting for 0-based indexing

        # =====================  MST TEST ===============================
        dot_to_neighbors_map = {}
        for i, sub_pixel_coords in enumerate(sub_dot):
            neighbor_coords = []
            for neighbor in idx_neighbors_with_mask[i]:
                # threshold = image_ary[int(((sub_pixel_coords[0] + sub_dot[neighbor][0]) / 2))][int((sub_pixel_coords[1] + sub_dot[neighbor][1]) / 2)]
                if neighbor != -1:  # and not edge_calculation.get_weight_based_on_first_sixth_and_last_sixth(sub_pixel_coords, sub_dot[neighbor], image_ary):
                    neighbor_coords.append(tuple(sub_dot[neighbor]))
            neighbor_coords = tuple(neighbor_coords)
            dot_to_neighbors_map[(sub_pixel_coords[0], sub_pixel_coords[1],)] = neighbor_coords
        if debug:
            colors = plt.cm.get_cmap('tab20c', len(dot_to_neighbors_map))
            plt.title("Neighbors to every corner")
            plt.imshow(image_grayscale, cmap="gray")
            neighbor_num = NUMBER_WANTED_NEIGHBORS - 1

            # Iterate over each corner, using enumerate to get a unique index for color
            for idx, (corner, neighbors) in enumerate(dot_to_neighbors_map.items()):
                # Plot each corner point in a unique color
                plt.scatter(corner[1], corner[0], color=colors(idx))

                # Plot lines to each neighbor in the same color and dotted style
                for n in neighbors:
                    plt.plot([corner[1], n[1]], [corner[0], n[0]], color=colors(idx), linestyle='--')

            plt.show()
            start = time.time()
        old_kruskal_alg = KruskalAlgorithm(tuple(map(tuple, sub_dot)), dot_to_neighbors_map, image_ary)
        old_kruskal_alg.calc_kruskal()
        tree = old_kruskal_alg.union_find.get_largest_tree()
        end = time.time()
        print(f"old_algorithm runtime in ms: {(end-start) * 1000}")

        # tree.get_root().set_coordinates_bfs(0, 0)
        # plt.imshow(image_grayscale, cmap="gray")
        # plt.scatter(0, 0)
        # tree.get_root().plot_tree()
        # plt.show()

        start2 = time.time()
        kruskal_alg = OptimizedKruskalAlgorithm(tuple(map(tuple, sub_dot)), dot_to_neighbors_map, image_ary)
        kruskal_alg.calc_kruskal()
        tree = kruskal_alg.union_find.get_largest_tree()
        grid = tree.to_2d_node_grid()
        print(f"GRID: \n {grid}")
        horizontal_lines = decode_horizontal_lines(grid, image_grayscale)
        print(f"horizontal lines:\n {horizontal_lines}")
        vertical_lines = decode_vertical_lines(grid, image_grayscale)
        print(f"vertical lines:\n {vertical_lines}")
        end2 = time.time()
        print(f"optimized algorithm runtime in ms: {(end2 - start2) * 1000}")

        fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 1, 1)
        # ax1.set_title('Outlier Elimination')
        ax2.set_title('Largest Tree')
        #ax1.axis('off')
        ax2.axis('off')

        # colors = ['cyan', 'green', 'blue', 'red', 'magenta', 'yellow', 'black', 'white']
        # for i, tree in enumerate(kruskal_alg.union_find.get_disjoint_trees()):
        #     print(f"size: {tree.get_size()}")
        #     ax1.imshow(image_grayscale, cmap="gray")
        #     tree.plot_mesh(ax=ax1, color=colors[i % len(colors)])

        ax2.imshow(image_ary, cmap="gray")
        kruskal_alg.union_find.get_largest_tree().plot_mesh(ax=ax2)
        plt.tight_layout()
        plt.show()
        # plt.savefig(os.path.join(
        #     "/Users/justus/Documents/uni/BA/bachelor-thesis/thesis/jbiermann_thesis/images/evaluation",
        #     "outlier-detection.png"), dpi=400)

        # tree.get_root().set_coordinates_bfs(0, 0)
        # plt.imshow(image_grayscale, cmap="gray")
        # tree.plot_mesh()
        # plt.show()
        print(f"bounds: {tree.compressed_predecessor.get_bounds()}")

        # mst_to_grid_mapper = MSTtoGridMapper(image_grayscale)
        # mst_to_grid_mapper.add_nodes_from_mst(tree)
        # mst_to_grid_mapper.add_nodes_from_mst_and_plot(tree)
        # plt.show()
