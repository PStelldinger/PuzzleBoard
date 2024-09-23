import numpy as np


def decode_horizontal_lines(grid, image):
    horizontal_edges = np.full((grid.shape[0], grid.shape[1] - 1), -1, dtype=int)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row][col] is not None:
                node = grid[row][col]
                if col + 1 < grid.shape[1]:
                    east_node = grid[row][col + 1]
                    horizontal_edges[row, col] = get_edge_value_to_node(node, east_node, image)
    return horizontal_edges


def decode_vertical_lines(grid, image):
    vertical_edges = np.full((grid.shape[0] - 1, grid.shape[1]), -1, dtype=int)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row][col] is not None:
                node = grid[row][col]
                if row + 1 < grid.shape[0]:
                    south_node = grid[row + 1][col]
                    vertical_edges[row, col] = get_edge_value_to_node(node, south_node, image)
    return vertical_edges


def get_edge_value_to_node(node, other_node, image):
    if other_node is None or node is None:
        return -1
    else:
        img_x_pos = int((node.abs_x + other_node.abs_x) / 2)
        img_y_pos = int((node.abs_y + other_node.abs_y) / 2)
        if image[img_y_pos, img_x_pos] > 0.5:
            return 1
        else:
            return 0
