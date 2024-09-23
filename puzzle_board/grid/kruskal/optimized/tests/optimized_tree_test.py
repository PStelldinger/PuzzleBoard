from grid.kruskal.optimized.optimized_tree_node import OptimizedTreeNode
import unittest
import numpy as np

from grid.kruskal.optimized.vector_rotation import rotate_point


class TestOptimizedTreeNode(unittest.TestCase):

    def test_single_node(self):
        node = OptimizedTreeNode(0, 0)
        node.rotation = 45
        self.assertEqual(node.get_rotation_to_root(),  0, "Failed single node test")

    def test_linear_tree(self):
        root = OptimizedTreeNode(0, 0)
        child = OptimizedTreeNode(1, 1)
        child.predecessor = root
        root.rotation = 30
        child.rotation = 15
        self.assertEqual(child.get_rotation_to_root(), 30, "Failed linear tree test")

    def test_no_rotation_to_root(self):
        root = OptimizedTreeNode(0, 0)
        child = OptimizedTreeNode(1, 1)
        child.predecessor = root
        # No rotation set
        self.assertEqual(child.get_rotation_to_root(), 0, "Failed no rotation test")

    def test_complex_tree(self):
        root = OptimizedTreeNode(0, 0)
        node1 = OptimizedTreeNode(1, 1)
        node2 = OptimizedTreeNode(2, 2)
        node1.predecessor = root
        node2.predecessor = node1
        root.rotation = 20
        node1.rotation = 15
        node2.rotation = 5
        self.assertEqual(node2.get_rotation_to_root(), 35, "Failed complex tree test")

    def test_vector_to_predecessor_no_rotation(self):
        root = OptimizedTreeNode(0, 0)
        child = OptimizedTreeNode(1, 1)
        child.vector_to_predecessor = np.array([-1, -1])
        child.predecessor = root
        child.rotation = 0  # No rotation
        expected_vector = np.array([-1,
                                    -1])  # Since there is no rotation, the vector should directly point to the child's position relative to the root
        np.testing.assert_array_equal(child.get_rotated_vector_to_predecessor(), expected_vector)

    def test_vector_to_predecessor_with_rotation(self):
        root = OptimizedTreeNode(0, 0)
        child = OptimizedTreeNode(1, 1)
        child.vector_to_predecessor = np.array([-1, -1])
        child.predecessor = root
        root.rotation = 90  # Apply rotation
        expected_vector = rotate_point(np.array([-1, -1]), 90)  # The expected vector after applying 90 degree rotation
        np.testing.assert_array_almost_equal(child.get_rotated_vector_to_predecessor(), expected_vector, decimal=5)

    def test_vector_to_root_single_predecessor(self):
        root = OptimizedTreeNode(0, 0)
        child = OptimizedTreeNode(1, 1)
        child.predecessor = root
        child.vector_to_predecessor = np.array([1, 1])
        expected_vector = np.array([1, 1])  # Direct vector to the root as there's only one predecessor
        np.testing.assert_array_equal(child.get_vector_to_root(), expected_vector)

    def test_vector_to_root_multiple_predecessors(self):
        root = OptimizedTreeNode(0, 0)
        child1 = OptimizedTreeNode(1, 1)
        child2 = OptimizedTreeNode(2, 2)
        child1.predecessor = root
        child2.predecessor = child1
        child1.vector_to_predecessor = np.array([1, 1])
        child2.vector_to_predecessor = np.array([2, 2])
        child1.rotation = 0
        child2.rotation = 0
        # Here, you would calculate the expected vector taking into account the rotations and positions.
        # This is a placeholder for the expected logic.
        expected_vector = np.array([3, 3])  # Simplified expected vector for illustration
        actual = child2.get_vector_to_root()
        np.testing.assert_array_equal(actual, expected_vector)

    def test_vector_to_root_multiple_predecessors_with_rotation(self):
        root = OptimizedTreeNode(0, 0)
        child1 = OptimizedTreeNode(1, 1)
        child2 = OptimizedTreeNode(2, 2)
        child1.predecessor = root
        child2.predecessor = child1
        child1.vector_to_predecessor = np.array([1, 1])
        child2.vector_to_predecessor = np.array([2, 2])
        root.rotation = 180
        child1.rotation = 0
        child2.rotation = 0
        # Here, you would calculate the expected vector taking into account the rotations and positions.
        # This is a placeholder for the expected logic.
        expected_vector = np.array([-3, -3])  # Simplified expected vector for illustration
        actual = child2.get_vector_to_root()
        np.testing.assert_array_equal(actual, expected_vector)
    def test_vector_to_self(self):
        node = OptimizedTreeNode(0, 0)
        expected_vector = np.array([0, 0])  # Vector to itself should be zero
        np.testing.assert_array_equal(node.get_rotated_vector_to_predecessor(), expected_vector)

    def test_add_child(self):
        root1 = OptimizedTreeNode(100, 0)
        root2 = OptimizedTreeNode(200, -100)
        child_of_root1 = OptimizedTreeNode(200, 0)
        child_of_root2 = OptimizedTreeNode(300, 0)
        root1.add_child(child_of_root1)
        root2.add_child(child_of_root2)
        child_of_root1.add_child(child_of_root2)

        expected_vector = np.array([-3, 0])
        np.testing.assert_array_equal(root2.get_vector_to_root(), expected_vector)


if __name__ == "__main__":
    unittest.main()
