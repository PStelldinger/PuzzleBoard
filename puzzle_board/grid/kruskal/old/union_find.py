from grid.kruskal.old.tree_node import TreeNode


class UnionFind:

    def __init__(self, dot, image):
        self.node_to_tree_map = {}
        self.coords_to_node_map = {}
        self.trees = set()
        self.image = image
        self.last_unified_tree = None
        for abs_y, abs_x in dot:
            self.make_tree(abs_x, abs_y)
        print("done")

    def union(self, u: TreeNode, v: TreeNode):
        if (u is None
                or v is None
                or u == v):
            return
        u_size = u.get_root().tree_size
        v_size = v.get_root().tree_size
        if u_size > v_size:
            u.add_sub_tree(v)
        else:
            v.add_sub_tree(u)
        self.last_unified_tree = u

    def find_tree(self, u):
        if self.node_to_tree_map[u] is not None:
            return self.node_to_tree_map[u]

    def make_tree(self, abs_x, abs_y):
        if not (abs_x, abs_y) in self.coords_to_node_map:
            root = TreeNode(abs_x, abs_y, self.image)
            self.trees.add(root)
            self.coords_to_node_map[(abs_x, abs_y)] = root
            self.node_to_tree_map[root] = root

    def get_largest_tree(self):
        return self.last_unified_tree

    def find_tree_from_coords(self, coords):
        """
        :param coords: tuple in (x, y) coordinate form
        :return:
        """
        if self.coords_to_node_map.__contains__(coords):
            return self.coords_to_node_map[coords]
        else:
            raise Exception("Node with given coordinates not in dict-structure")
