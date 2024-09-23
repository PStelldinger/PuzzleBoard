from queue import PriorityQueue

from grid.edge import Edge
from grid.edge_calculation import get_distance
from grid.edge_calculation import get_weight_based_on_first_sixth_and_last_sixth
from grid.kruskal.old.union_find import UnionFind


class KruskalAlgorithm:

    def __init__(self, dot, dot_neighbors_map, image):
        # dot and neighbors holds coords in (y,x) form
        self.union_find = UnionFind(dot, image)
        self.edge_queue = PriorityQueue()
        self.image = image
        for coords in dot:
            self.union_find.make_tree(coords[1], coords[0])
            current_neighbors = dot_neighbors_map[coords]
            for neighbor in current_neighbors:
                self.union_find.make_tree(neighbor[1], neighbor[0])
            self.init_edges(coords, current_neighbors)

    def calc_kruskal(self):
        while not self.edge_queue.empty():
            edge = self.edge_queue.get()
            self.union_find.union(edge.node_one, edge.node_two)

    def init_edges(self, corner, neighbors):
        corner_node = self.union_find.coords_to_node_map[(corner[1], corner[0])]
        for neighbor in neighbors:
            if neighbor != corner:
                neighbor_node = self.union_find.coords_to_node_map[(neighbor[1], neighbor[0])]
                weight = self.calc_edge_weight(corner, neighbor)
                edge = Edge(weight, corner_node, neighbor_node)
                self.edge_queue.put(edge)

    def calc_edge_weight(self, node_coords, other_node_coords):
        # for now: just distance as a measure:
        distance = get_distance(node_coords, other_node_coords)
        if get_weight_based_on_first_sixth_and_last_sixth(node_coords, other_node_coords, self.image):
            distance = distance * 100
        return distance
