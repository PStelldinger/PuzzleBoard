import unittest

from grid.direction import Direction
from grid.edge_creator import is_symmetric_opposite
from grid.kruskal.old.tree_node import TreeNode  # Ensure TreeNode is accessible

class TestTreeNode(unittest.TestCase):

    def setUp(self):
        # Setup a simple tree structure for testing
        self.root = TreeNode(0, 0)
        self.child_one = TreeNode(1, 0)
        self.child_two = TreeNode(0, 1)
        self.child_one_one = TreeNode(2, 0)
        self.root.add_child_in_direction(self.child_one, Direction.X)
        self.root.add_child_in_direction(self.child_two, Direction.Y)
        self.child_one.add_child_in_direction(self.child_one_one, Direction.X)

    def test_get_root(self):
        self.assertEqual(self.root, self.root.get_root())
        self.assertEqual(self.root, self.child_one.get_root())
        self.assertEqual(self.root, self.child_two.get_root())
        self.assertEqual(self.root, self.child_one_one.get_root())

    def test_get_wanted_direction(self):
        self.assertEqual(Direction.X, self.root.get_wanted_direction(self.child_one))
        self.assertEqual(Direction.NEGATIVE_Y, self.root.get_wanted_direction(self.child_two))
        self.assertEqual(Direction.NEGATIVE_X, self.root.get_wanted_direction(TreeNode(-1, 0)))
        self.assertEqual(Direction.Y, self.root.get_wanted_direction(TreeNode(0, -1)))
        self.assertEqual(self.child_one.direction_from_parent, self.child_one.get_wanted_direction(self.child_one_one))
        #self.assertEqual(self.child_one.direction_from_parent, self.child_one.get_wanted_direction(TreeNode(-1, 0)))

    def test_add_subtree(self):
        pass

    def test_does_subtree_contain(self):
        self.assertTrue(self.root.does_subtree_contain(self.root))
        self.assertFalse(self.child_one.does_subtree_contain(self.root))
        self.assertTrue(self.child_one.does_subtree_contain(self.child_one_one))

    def test_get_tree_size(self):
        pass
        #self.assertEqual(3, self.root.subtree_size)

    def test_alignment(self):
        pass
    def test_get_nodes_stream(self):
        nodes = set()
        expected_nodes = set()
        expected_nodes.add(self.root)
        expected_nodes.add(self.child_one)
        expected_nodes.add(self.child_two)
        expected_nodes.add(self.child_one_one)
        for node in self.root.get_nodes():
            nodes.add(node)
        self.assertEqual(expected_nodes, nodes)
        nodes = set()
        expected_nodes.remove(self.child_two)
        expected_nodes.remove(self.root)
        for node in self.child_one.get_nodes():
            nodes.add(node)
        self.assertEqual(expected_nodes, nodes)

    def test_symmetry(self):
        self.assertTrue(is_symmetric_opposite(TreeNode(-1, 0), TreeNode(0, 0), TreeNode(1, 0), 0.0))
        self.assertFalse(is_symmetric_opposite(TreeNode(-1, 0), TreeNode(0, 0), TreeNode(-1, 0), 0.0))
        self.assertFalse(is_symmetric_opposite(TreeNode(-4, 0), TreeNode(0, 0), TreeNode(1, 0), 1.0))

if __name__ == '__main__':
    unittest.main()
