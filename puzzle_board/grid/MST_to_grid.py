from grid.kruskal.old.tree_node import TreeNode
from grid.grid_edge import GridEdge


class MSTtoGridMapper:
    def __init__(self, image):
        self.min_x = float('inf')
        self.min_y = float('inf')
        self.max_x = float('-inf')
        self.max_y = float('-inf')
        self.coords = dict()
        self.edges = set()
        self.image = image

    def add_nodes_from_mst_and_plot(self, node: TreeNode):
        self.add_nodes_from_mst(node)
        for edge in self.edges:
            edge.plot(self.image)

    def add_nodes_from_mst(self, node: TreeNode):
        for node in node.get_root().get_nodes():
            self.add_node(node)

    def add_node(self, node: TreeNode):
        node_x = node.relative_coords[0]
        node_y = node.relative_coords[1]
        self.check_and_set_bounds(node_x, node_y)
        neighbors = self.get_neighbors((node_x, node_y))
        for neighbor in neighbors:
            node_abs_x = node.abs_x
            node_abs_y = node.abs_y
            neighbor_abs_x = neighbor.abs_x
            neighbor_abs_y = neighbor.abs_y
            self.edges.add(GridEdge((node_abs_x, node_abs_y), (neighbor_abs_x, neighbor_abs_y), self.image))
        self.coords[(node_x, node_y)] = node

    def check_and_set_bounds(self, x, y):
        if x < self.min_x:
            self.min_x = x
        elif x > self.max_x:
            self.max_x = x
        if y < self.min_y:
            self.min_y = y
        elif y > self.max_y:
            self.max_y = y

    def get_neighbors(self, x_y_tuple):
        neighbors = []
        for coords in ((x_y_tuple[0] - 1, x_y_tuple[1]), (x_y_tuple[0] + 1, x_y_tuple[1]), (x_y_tuple[0], x_y_tuple[1] + 1), (x_y_tuple[0], x_y_tuple[1] - 1)):
            try:
                neighbor = self.coords[coords]
                neighbors.append(neighbor)
            except KeyError:
               print("No neighbor")
        return neighbors

    def get_edges(self):
        return self.edges
