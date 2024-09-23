from queue import PriorityQueue

from grid import edge_calculation
from grid.edge import Edge
from grid.kruskal.optimized.optimized_union_find import OptimizedUnionFind


class OptimizedKruskalAlgorithm:

    def __init__(self, dot, dot_neighbors_map, image):
        # dot and neighbors holds coords in (y,x) form
        self.union_find = OptimizedUnionFind(dot, image)
        self.edge_queue = PriorityQueue()
        self.image = image
        self.added_edges = []
        for coords in dot:
            self.union_find.make_tree(coords[1], coords[0])
            current_neighbors = dot_neighbors_map[coords]
            for neighbor in current_neighbors:
                self.union_find.make_tree(neighbor[1], neighbor[0])
            self.init_edges(coords, current_neighbors)

    def calc_kruskal(self):
        while not self.edge_queue.empty():
            edge = self.edge_queue.get()
            self.union_find.union(edge.node_one, edge.node_two, edge.weight)
            # if self._check_if_edge_weight_is_satisfactory(edge):
            #     self.union_find.union(edge.node_one, edge.node_two)
            #     self.added_edges.append(edge)


    def init_edges(self, corner, neighbors):
        corner_node = self.union_find.coords_to_node_map[(corner[1], corner[0])]
        for neighbor in neighbors:
            if neighbor != corner:
                neighbor_node = self.union_find.coords_to_node_map[(neighbor[1], neighbor[0])]
                weight = edge_calculation.calc_edge_weight(corner, neighbor, self.image)
                edge = Edge(weight, corner_node, neighbor_node)
                self.edge_queue.put(edge)

    def _check_if_edge_weight_is_satisfactory(self, edge):
        if len(self.added_edges) == 0:
            return True
        else:
            total_weight = 0
            for edge in self.added_edges:
              total_weight += edge.weight
            avg_weight = total_weight / len(self.added_edges)
            return avg_weight * 3 > edge.weight > avg_weight / 3
