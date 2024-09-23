from collections import deque
from queue import Queue

from grid.direction import Direction
from grid.edge_creator import check_if_parent_child_and_next_child_on_same_axis, are_nodes_collinear
import matplotlib.pyplot as plt


class TreeNode:

    def __init__(self, abs_x, abs_y, image, relative_x=0, relative_y=0):
        self.abs_x = abs_x
        self.abs_y = abs_y
        self.relative_x = relative_x
        self.relative_y = relative_y
        self.relative_coords = (self.relative_x, self.relative_y)
        self.children = set()
        self.parent = None
        self.direction_from_parent = None
        self.tree_size = 1
        self.image = image
        self.root = self

    def get_root(self):
        if self.root.parent is None:
            return self.root
        if self.parent is None:
            return self
        else:
            self.root = self.parent.get_root()
            return self.root

    def does_subtree_contain(self, node):
        for node_in_tree in self.get_nodes():
            if node_in_tree == node:
                return True
        return False

    def get_nodes(self):
        visited_nodes = set()
        queue = deque()
        queue.append(self)
        while len(queue) > 0:
            node = queue.popleft()
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child in node.children:
                    queue.append(child)
                yield node

    def set_direction_from_parent(self, new_direction):
        self.direction_from_parent = new_direction

    def add_child_in_direction(self, child, direction) -> None:
        self.children.add(child)
        child.set_direction_from_parent(direction)
        root = self.get_root()
        other_root = child.get_root()
        root.tree_size += other_root.tree_size
        child.parent = self
        coords_to_add = self.get_coordinates_from_direction(direction)
        child.set_coordinates_bfs(self.relative_coords[0] + coords_to_add[0],
                                  self.relative_coords[1] + coords_to_add[1])

    def get_coordinates_from_direction(self, direction):
        coordinate_map = {
            Direction.X: (1, 0),
            Direction.Y: (0, -1),
            Direction.NEGATIVE_X: (-1, 0),
            Direction.NEGATIVE_Y: (0, 1)
        }
        return coordinate_map[direction]

    def set_coordinates_bfs(self, relative_x, relative_y):
        queue = Queue()
        visited_nodes = set()
        self.relative_coords = (relative_x, relative_y)
        queue.put(self)
        visited_nodes.add(self)
        while not queue.empty():
            node = queue.get()
            for child_node in node.children:
                if child_node is not None:
                    direction = child_node.direction_from_parent
                    if direction == Direction.X:
                        child_node.relative_coords = (node.relative_coords[0] + 1, node.relative_coords[1])
                        queue.put(child_node)
                    if direction == Direction.Y:
                        child_node.relative_coords = (node.relative_coords[0], node.relative_coords[1] - 1)
                        queue.put(child_node)
                    if direction == Direction.NEGATIVE_Y:
                        child_node.relative_coords = (node.relative_coords[0], node.relative_coords[1] + 1)
                        queue.put(child_node)
                    if direction == Direction.NEGATIVE_X:
                        child_node.relative_coords = (node.relative_coords[0] - 1, node.relative_coords[1])
                        queue.put(child_node)
            visited_nodes.add(node)

    def get_wanted_direction(self, other_node) -> Direction:
        x_diff = self.abs_x - other_node.abs_x
        y_diff = self.abs_y - other_node.abs_y
        blocked_directions = set()
        if self.parent:
            if are_nodes_collinear(self.parent, self, other_node):
                return self.direction_from_parent
            else:
                blocked_directions.add(self.direction_from_parent)
        for child in self.children:
            if are_nodes_collinear(child, self, other_node):
                return child.direction_from_parent.opposite()
            else:
                blocked_directions.add(child.direction_from_parent.opposite())
                blocked_directions.add(child.direction_from_parent)
        next_direction = self.get_direction_from_diffs(x_diff, y_diff)
        if next_direction in blocked_directions:
            if next_direction.is_x_axis():
                next_direction = self.get_y_axis_direction(y_diff)
            else:
                next_direction = self.get_x_axis_direction(x_diff)
        return self.resolve_conflict_and_set_direction(other_node, next_direction)

    def get_direction_if_parent_exists(self, other_node, x_diff, y_diff) -> Direction:
        if self.check_alignment(self.parent, other_node):
            return self.direction_from_parent
        else:
            if self.direction_from_parent.is_x_axis():
                return self.get_y_axis_direction(y_diff)
            else:
                return self.get_x_axis_direction(x_diff)

    def get_direction_from_diffs(self, x_diff, y_diff) -> Direction:
        if abs(x_diff) > abs(y_diff):
            return self.get_x_axis_direction(x_diff)
        else:
            return self.get_y_axis_direction(y_diff)

    def get_x_axis_direction(self, x_diff) -> Direction:
        if x_diff > 0:
            return Direction.NEGATIVE_X
        else:
            return Direction.X

    def get_y_axis_direction(self, y_diff) -> Direction:
        if y_diff < 0:
            return Direction.NEGATIVE_Y
        else:
            return Direction.Y

    def check_alignment(self, parent, other_node) -> bool:
        return are_nodes_collinear(parent, self, other_node)

    def check_alignment_with_any_neighbor(self, other_node) -> bool:
        for child in self.children:
            if (child != other_node
                    and are_nodes_collinear(self, child, other_node)
                    and child.direction_from_parent.opposite() == other_node.direction_from_parent):
                return True
        return False

    def add_sub_tree(self, other_node) -> None:
        if self.get_root().does_subtree_contain(other_node):
            return
        direction = self.get_wanted_direction(other_node)
        other_node.flip_child_parent_nodes_in_branch()
        if other_node.parent is not None or len(other_node.children) > 0:
            other_node_wanted_direction = other_node.get_wanted_direction(self)
            while direction != other_node_wanted_direction.opposite():
                # print(f"wanted: {direction}, other {other_node_wanted_direction}")
                # plt.imshow(self.image, cmap="gray")
                # self.get_root().plot_tree()
                # other_node.get_root().plot_tree("green")
                # plt.text(other_node.abs_x, other_node.abs_y, f"before_rotation", color="red")
                # plt.show()

                other_node.get_root().rotate_tree()
                other_node_wanted_direction = other_node_wanted_direction.rotate()

                # plt.imshow(self.image, cmap="gray")
                # self.get_root().plot_tree()
                # other_node.get_root().plot_tree()
                # plt.text(other_node.abs_x, other_node.abs_y, f"after_rotation", color="red")
                # plt.show()
            self.add_child_in_direction(other_node, direction)
            # plt.imshow(self.image, cmap="gray")
            # self.get_root().plot_tree()
            # plt.text(other_node.abs_x, other_node.abs_y, f"added in {direction} with parent", color="red")
            # plt.show()
        else:
            self.add_child_in_direction(other_node, direction)
            # plt.imshow(self.image, cmap="gray")
            # self.get_root().plot_tree()
            # plt.text(other_node.abs_x, other_node.abs_y, f"added in {direction}", color="blue")
            # plt.show()

    def flip_child_parent_nodes_in_branch(self):
        current_node = self
        node_stack = []
        while current_node is not None:
            node_stack.append(current_node)
            current_node = current_node.parent
        parent_node = node_stack.pop()
        while len(node_stack) > 0:
            child_node = node_stack.pop()
            parent_node.parent = child_node
            parent_node.set_direction_from_parent(child_node.direction_from_parent.opposite())
            parent_node.children.remove(child_node)
            child_node.children.add(parent_node)
            child_node.parent = None
            parent_node = child_node

    def rotate_node(self, n=1):
        # self.node_map = {key.rotate(n): value for key, value in self.node_map.items()}
        # self.relative_coords = rotate_point(self.relative_coords, n)
        if self.direction_from_parent is not None:
            self.set_direction_from_parent(self.direction_from_parent.rotate(n))

    def rotate_tree(self, n=1):
        visited_nodes = set()
        queue = [self]

        while queue:
            node = queue.pop(0)

            if node is not None and node not in visited_nodes:
                visited_nodes.add(node)
                node.rotate_node(n)

                # Add only unvisited child nodes to the queue
                for child in node.children:
                    if child and child not in visited_nodes:
                        queue.append(child)

    def resolve_conflict_and_set_direction(self, other_node, direction):
        x_diff = self.abs_x - other_node.abs_x
        y_diff = self.abs_y - other_node.abs_y
        is_x_axis = direction.is_x_axis()
        updated_direction = direction
        if self.has_child_in_direction(direction):
            conflicting_child = self.get_child_in_direction(direction)
            if conflicting_child:
                if self.check_alignment_with_any_neighbor(conflicting_child):
                    updated_direction = self.get_alternate_direction(x_diff, y_diff, is_x_axis)
                else:
                    self.reorient_conflicting_child(conflicting_child, x_diff, y_diff, is_x_axis)
        return updated_direction

    def get_alternate_direction(self, x_diff, y_diff, is_x_axis):
        if is_x_axis:
            return Direction.Y if y_diff > 0 else Direction.NEGATIVE_Y
        else:
            return Direction.X if x_diff < 0 else Direction.NEGATIVE_X

    def reorient_conflicting_child(self, conflicting_child, x_diff, y_diff, is_x_axis) -> Direction:
        old_direction = conflicting_child.direction_from_parent
        new_direction = conflicting_child.get_alternate_direction(x_diff, y_diff, is_x_axis)
        # conflicting_child.set_direction_from_parent(new_direction)
        rotations_needed = old_direction.rotations_needed_for(new_direction)
        conflicting_child.get_root().rotate_tree(rotations_needed)

    def has_no_child_in_direction(self, direction):
        for child in self.children:
            if child.direction_from_parent == direction:
                return False
        return True

    def get_child_in_direction(self, direction):
        for child in self.children:
            if child.direction_from_parent == direction:
                return child

    def has_child_in_direction(self, direction):
        for child in self.children:
            if child.direction_from_parent == direction:
                return True
        return False

    def plot(self):
        plt.scatter(self.abs_x, self.abs_y)
        root_string = "R: " if self.parent is None else ""
        plt.text(self.abs_x, self.abs_y, f"{root_string}[x:{self.relative_coords[0]}|y:{self.relative_coords[1]}]",
                 color="red")

    def plot_tree(self, color="red"):
        plt.scatter(self.abs_x, self.abs_y)
        root_string = "R: " if self.parent is None else ""
        plt.text(self.abs_x, self.abs_y, f"{root_string}[x:{self.relative_coords[0]}|y:{self.relative_coords[1]}]",
                 color=color)

        for child in self.children:
            if child:
                # Draw an edge from the current node to the child node
                plt.plot([self.abs_x, child.abs_x], [self.abs_y, child.abs_y], color=color)
                plt.text((self.abs_x + child.abs_x) / 2, (self.abs_y + child.abs_y) / 2,
                         f"{child.direction_from_parent}", color=color)
                # Recursively plot the child node and its subtree
                child.plot_tree(color)

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return (self.abs_x == other.abs_x
                and self.abs_y == other.abs_y)

    def __hash__(self):
        return hash((self.abs_x, self.abs_y))

    def debug_plot(self, x_pos, y_pos, text, color):
        self.get_root().plot_tree()
        plt.imshow(self.image, cmap="gray")
        plt.text(x_pos, y_pos, text, color=color)
        plt.show()
