"""
An optimized tree node for faster grid reconstruction.

Optimizaitons:
    + usage of distance vectors
    + flattening the tree
    + rotation along the path to root
"""
from __future__ import annotations

import statistics
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from grid.edge_creator import are_nodes_collinear
from grid.kruskal.optimized import vector_rotation
from grid.kruskal.optimized.axis import Axis
from grid.kruskal.optimized.optimized_direction import OptimizedDirection


class OptimizedTreeNode:

    def __init__(self, abs_x, abs_y, image):
        self.predecessor = self
        self.abs_x = abs_x
        self.abs_y = abs_y
        self.vector_to_predecessor = np.array([0, 0])
        self.rotation = 0
        # Direction => Node
        self.neighbors = dict()
        self.all_children = set()
        self.image = image
        self.compressed_predecessor = self
        self.edge_weights = []
        self.min_relative_x = 0
        self.min_relative_y = 0
        self.max_relative_x = 0
        self.max_relative_y = 0

    def get_root(self) -> OptimizedTreeNode:
        if self.compressed_predecessor is self:
            return self
        return self.compressed_predecessor.get_root()

    def get_size(self) -> int:
        return len(self.all_children)

    def compare_edges(self, edge_weight: float, other_edge_weight: float, factor: float) -> bool:
        return edge_weight * (1 / factor) < other_edge_weight < edge_weight * factor

    def is_other_tree_avg_weight_eligible(self, other_node: OptimizedTreeNode) -> bool:
        other_uf_root = other_node.get_root()
        if len(other_uf_root.edge_weights) == 0 or len(self.get_root().edge_weights) == 0:
            return True
        avg_weight = statistics.mean(self.get_root().edge_weights)
        other_avg_weight = statistics.mean(other_uf_root.edge_weights)
        return other_avg_weight * 0.3 < avg_weight < other_avg_weight * 2.9

    def get_rotation_to_root(self) -> int:
        rotation = 0
        current = self
        while current is not current.predecessor:
            current = current.predecessor
            rotation += current.rotation
        return rotation

    def get_rotated_vector_to_predecessor(self) -> np.ndarray:
        return vector_rotation.rotate_point(self.vector_to_predecessor, self.get_rotation_to_root())

    def get_vector_to_root(self, rotated=True) -> np.ndarray:
        current = self
        vector_to_root = np.array([0, 0])
        while current is not current.predecessor:
            if rotated and current.rotation != 0:
                vector_to_root = vector_rotation.rotate_point(vector_to_root, current.rotation)
            vector_to_root += current.vector_to_predecessor
            current = current.predecessor
        vector_to_root += current.vector_to_predecessor
        return vector_to_root

    def get_neighbors_with_directions_and_edge_weights(self) -> set[
        tuple[OptimizedTreeNode, OptimizedDirection, float]]:
        neighbors = set()
        directions_to_check = [OptimizedDirection.X, OptimizedDirection.Y, OptimizedDirection.NEGATIVE_X,
                               OptimizedDirection.NEGATIVE_Y]
        for direction in directions_to_check:
            found_neighbor = self.neighbors.get(direction)
            if found_neighbor is not None:
                neighbors.add((found_neighbor[0], direction, found_neighbor[1]))
        return neighbors

    def add_child(self, child_node_to_add: OptimizedTreeNode, edge_weight: float) -> bool:
        n_d_e = self.get_neighbors_with_directions_and_edge_weights()
        for neighbor, direction, neighbor_edge_weight in n_d_e:
            if not self.compare_edges(neighbor_edge_weight, edge_weight, 2.2):
                return False
        if not self.is_other_tree_avg_weight_eligible(
                child_node_to_add):  # not self.is_edge_difference_eligible(edge_weight) or
            return False
        root_one = self.get_root()
        other_root = child_node_to_add.get_root()
        wanted_direction = self.wanted_direction(child_node_to_add)
        other_wanted_direction = child_node_to_add.wanted_direction(self)
        if wanted_direction == OptimizedDirection.NO_DIRECTION or other_wanted_direction == OptimizedDirection.NO_DIRECTION:
            return False
        original_direction = other_wanted_direction
        rotations = 0
        while wanted_direction != other_wanted_direction.opposite():
            other_wanted_direction = other_wanted_direction.rotate(1)
            rotations += 1
        direction_vector = np.array(wanted_direction.value)
        other_child_to_other_root_vector = vector_rotation.rotate_point(child_node_to_add.get_vector_to_root(),
                                                                        rotations * 90)
        child_to_root = self.get_vector_to_root()
        self.neighbors[wanted_direction] = (child_node_to_add, edge_weight,)
        child_node_to_add.neighbors[original_direction] = (self, edge_weight,)

        other_root_to_root_vector = - other_child_to_other_root_vector - direction_vector + child_to_root

        root_one.set_new_bounds(other_root.max_relative_x - other_root_to_root_vector[0],
                                other_root.max_relative_y - other_root_to_root_vector[1],
                                other_root.min_relative_x - other_root_to_root_vector[0],
                                other_root.min_relative_y - other_root_to_root_vector[1])

        other_root.vector_to_predecessor += other_root_to_root_vector
        other_root.rotation += rotations * 90
        other_root.predecessor = root_one

        root_one.all_children.add(other_root)

        for weight in other_root.edge_weights:
            root_one.edge_weights.append(weight)
        root_one.edge_weights.append(edge_weight)

        for child in other_root.all_children:
            root_one.all_children.add(child)

        # print(f"weight: {edge_weight}")
        # plt.imshow(self.image, cmap="grey")
        # self.plot_mesh()
        # plt.scatter(self.abs_x, self.abs_y, color="green")
        # plt.scatter(child_node_to_add.abs_x, child_node_to_add.abs_y, color="blue")
        # plt.show()
        return True

    def wanted_direction(self, child_node_to_add: OptimizedTreeNode) -> OptimizedDirection:
        neighbors_with_directions = self.get_neighbors_with_directions_and_edge_weights()
        rotations = self.get_rotation_to_root() + self.rotation
        blocked_axes = set()
        for neighbor, direction, edge_weight in neighbors_with_directions:
            direction = direction.rotate(int(rotations / 90))
            blocked_axes.add(direction.get_axis())
            if are_nodes_collinear(neighbor, self, child_node_to_add):
                return direction.opposite()
        return self.get_direction_from_absolute_image_coordinates(child_node_to_add, blocked_axes)

    def get_direction_from_absolute_image_coordinates(self, other: OptimizedTreeNode,
                                                      blocked_axes: set[Axis]) -> OptimizedDirection:
        # x_vector = np.array([other.abs_x - self.abs_x, 0])
        # direction_vector = np.array([other.abs_x - self.abs_x, other.abs_y - self.abs_y])
        # orientation_vec_ab = np.arctan2(x_vector[1], x_vector[0])
        # orientation_direction_vector = np.arctan2(direction_vector[1], direction_vector[0])


        x_diff = other.abs_x - self.abs_x
        y_diff = other.abs_y - self.abs_y
        if Axis.X_AXIS in blocked_axes and Axis.Y_AXIS in blocked_axes:
            return OptimizedDirection.NO_DIRECTION

        # if (np.abs(np.sin(orientation_direction_vector)) < 0.09 and Axis.X_AXIS not in blocked_axes) or Axis.Y_AXIS in blocked_axes:
        if (abs(x_diff) > abs(y_diff) and Axis.X_AXIS not in blocked_axes) or Axis.Y_AXIS in blocked_axes:
            #     if abs(orientation_direction_vector) > 1.57:
            if x_diff < 0:
                return OptimizedDirection.NEGATIVE_X
            # elif abs(orientation_direction_vector) <= 1.57:
            elif x_diff > 0:
                return OptimizedDirection.X
        elif OptimizedDirection.Y and Axis.Y_AXIS not in blocked_axes:
            if y_diff > 0:
                return OptimizedDirection.Y
            else:
                return OptimizedDirection.NEGATIVE_Y
        raise RuntimeError("Somethings wrong with the direction system")

    def set_new_bounds(self, added_tree_max_x, added_tree_max_y, added_tree_min_x, added_tree_min_y):
        if added_tree_max_x > self.max_relative_x:
            self.max_relative_x = added_tree_max_x
        if added_tree_max_y > self.max_relative_y:
            self.max_relative_y = added_tree_max_y
        if added_tree_min_x < self.min_relative_x:
            self.min_relative_x = added_tree_min_x
        if added_tree_min_y < self.min_relative_y:
            self.min_relative_y = added_tree_min_y

    def get_bounds(self):
        return self.max_relative_x, self.max_relative_y, self.min_relative_x, self.min_relative_y

    def plot_mesh(self, ax=plt, color="red"):
        root = self.get_root()
        ax.scatter(root.abs_x, root.abs_y, marker='o', color=color, s=1)
        ax.text(root.abs_x, root.abs_y, "ROOT",
                color=color, size=6)
        for node in root.all_children:
            # print(f"node to root {node.get_vector_to_root()}")
            # for n, direction in node.get_neighbors_with_directions():
            #     print(f"neighbor {direction}")
            #     plt.plot([node.abs_x, n.abs_x], [node.abs_y,n.abs_y], color="red")
            #     plt.text((node.abs_x + n.abs_x) / 2, (node.abs_y + n.abs_y) / 2, f"{direction.value}", color="red")
            ax.scatter(node.abs_x, node.abs_y, marker='o', color=color, s=1)
            pos = -node.get_vector_to_root()
            ax.text(node.abs_x, node.abs_y, f"({pos[0]},{pos[1]})",
                    color=color, size=6)
        # for node in root.all_children:
        #     plt.arrow(node.abs_x, node.abs_y, (node.union_find_root.abs_x - node.abs_x) * 0.9,
        #               (node.union_find_root.abs_y - node.abs_y) * 0.9,
        #               head_width=10, head_length=10, ec='green', fc='green')

    def to_2d_node_grid(self) -> np.ndarray:
        root = self.get_root()
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0
        for c in root.all_children:
            c_pos = -c.get_vector_to_root()
            if c_pos[0] < min_x:
                min_x = c_pos[0]
            if c_pos[0] > max_x:
                max_x = c_pos[0]
            if c_pos[1] < min_y:
                min_y = c_pos[1]
            if c_pos[1] > max_y:
                max_y = c_pos[1]
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        # fill the grid with initial none's
        grid = np.full((y_diff + 1, x_diff + 1), None)
        candidates = copy(root.all_children)
        candidates.add(root)
        for c in candidates:
            initial_pos = -c.get_vector_to_root()
            shifted_pos = np.array([initial_pos[0] - min_x, initial_pos[1] - min_y])
            grid[shifted_pos[1]][shifted_pos[0]] = c
        print(grid[0, 0])
        return grid

    def __eq__(self, other):
        if not isinstance(other, OptimizedTreeNode):
            return False
        return (self.abs_x == other.abs_x
                and self.abs_y == other.abs_y)

    def __hash__(self):
        return hash((self.abs_x, self.abs_y))

    def __str__(self):
        return f"[{-self.get_vector_to_root()[0]}, {-self.get_vector_to_root()[1]}]"

    def __repr__(self):
        return f"[{-self.get_vector_to_root()[0]}, {-self.get_vector_to_root()[1]}]"
