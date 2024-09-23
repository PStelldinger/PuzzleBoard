from grid.kruskal.optimized.optimized_tree_node import OptimizedTreeNode


class OptimizedUnionFind:

    def __init__(self, dot, image):
        self.node_to_tree_map = {}
        self.coords_to_node_map = {}
        self.image = image
        self.last_unified_tree = None
        self.disjoint_trees = set()
        # for abs_y, abs_x in dot:
        #     self.make_tree(abs_x, abs_y)
        print("done")

    def union(self, u: OptimizedTreeNode, v: OptimizedTreeNode, edge_weight: float):
        if (u is None
                or v is None
                or u == v
                or len(self.disjoint_trees) <= 1):
            return
        u_uf_root = self.find(u)
        v_uf_root = self.find(v)
        if u_uf_root == v_uf_root:
            return
        if u_uf_root.get_size() >= v_uf_root.get_size():
            if u.add_child(v, edge_weight):
                self.disjoint_trees.remove(v_uf_root)
            self.last_unified_tree = u
        else:
            if v.add_child(u, edge_weight):
                self.disjoint_trees.remove(u_uf_root)
            self.last_unified_tree = v

    def find(self, u: OptimizedTreeNode) -> OptimizedTreeNode:
        current = u
        second_pointer = u
        while current is not current.predecessor:
            current = current.predecessor
        while second_pointer is not second_pointer.predecessor:
            compression_node = second_pointer
            compression_node.compressed_predecessor = current
            second_pointer = second_pointer.predecessor
        return current

    def make_tree(self, abs_x, abs_y):
        if not (abs_x, abs_y) in self.coords_to_node_map:
            root = OptimizedTreeNode(abs_x, abs_y, self.image)
            self.coords_to_node_map[(abs_x, abs_y)] = root
            self.disjoint_trees.add(root)

    def get_largest_tree(self):
        max_size = 0
        largest_tree = self.last_unified_tree
        for tree in self.disjoint_trees:
            if tree.get_size() > max_size:
                max_size = tree.get_size()
                largest_tree = tree
        return largest_tree

    def get_disjoint_trees(self):
        return self.disjoint_trees
