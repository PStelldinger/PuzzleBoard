import math

import numpy as np


def pruefe_ob_vektor_punkt_nahe_schneidet(vektor, anfangspunkt, pruefpunkt, toleranz):
    """
    Überprüft, ob ein gegebener Vektor einen bestimmten Punkt innerhalb einer Toleranz schneidet.

    Args:
    vektor (tuple): Der Vektor in der Form (vx, vy).
    anfangspunkt (tuple): Der Anfangspunkt des Vektors (x0, y0).
    pruefpunkt (tuple): Der zu überprüfende Punkt (px, py).
    toleranz (float): Die Toleranz für die Nähe des Punktes zur Linie.

    Returns:
    bool: True, wenn der Punkt nahe genug an der Linie liegt, sonst False.
    """
    vx, vy = vektor
    x0, y0 = anfangspunkt
    px, py = pruefpunkt

    # Formel für den Abstand eines Punktes von einer Linie
    # Linie: v_y * (x - x0) - v_x * (y - y0) = 0
    A, B, C = vy, -vx, vx * y0 - vy * x0
    distanz = abs(A * px + B * py + C) / math.sqrt(A ** 2 + B ** 2)
    print(distanz)
    # Überprüfen, ob die Distanz innerhalb der Toleranz liegt
    return distanz <= toleranz


# Beispiel: Überprüfen mit Toleranz
toleranz = 0.5  # Toleranzbereich von 0.5 Einheiten
print(pruefe_ob_vektor_punkt_nahe_schneidet((3, 2), (0, 0), (3, 3), 0.84))


def check_if_parent_child_and_next_child_on_same_axis(parent, child, next_child, threshold):
    vx = child.abs_x - next_child.abs_x
    vy = child.abs_y - next_child.abs_y
    x0 = child.abs_x
    y0 = child.abs_y
    px = parent.abs_x
    py = parent.abs_y

    a = vy
    b = -vx
    c = vx * y0 - vy * x0

    distance = abs(a * px + b * py + c) / math.sqrt(a ** 2 + b ** 2)
    print(distance)

    # Überprüfen, ob die Distanz innerhalb der Toleranz liegt
    return distance <= threshold


# Check if pointC is the symmetric point of pointA or pointB with respect to the middle point
def is_symmetric_opposite(node, mid_node, check_node, threshold):
    # Calculate the mid-point of AC or BC (depending on which one is supposed to be symmetric)
    midpoint = ((node.abs_x + check_node.abs_x) / 2, (node.abs_y + check_node.abs_y) / 2)
    # Check if B is the midpoint within a threshold
    return abs(midpoint[0] - mid_node.abs_x) <= threshold and abs(midpoint[1] - mid_node.abs_y) <= threshold


def are_nodes_collinear(node1, node2, node3, tolerance=10_000):
    """
    Checks if three nodes are collinear within a given tolerance.

    Args:
    - node1, node2, node3: Dictionaries with 'abs_x' and 'abs_y' keys representing node positions.
    - tolerance: The tolerance within which the area of the triangle formed by the nodes is considered zero.

    Returns:
    - True if the nodes are collinear within the given tolerance, False otherwise.
    """

    # Extract the coordinates of the nodes
    x1, y1 = node1.abs_x, node1.abs_y
    x2, y2 = node2.abs_x, node2.abs_y
    x3, y3 = node3.abs_x, node3.abs_y

    vec_a = np.array([x1, y1])
    vec_b = np.array([x2, y2])
    vec_c = np.array([x3, y3])

    vec_ab = vec_b - vec_a
    vec_bc = vec_c - vec_b

    orientation_vec_ab = np.arctan2(vec_ab[1], vec_ab[0])
    orientation_vec_bc = np.arctan2(vec_bc[1], vec_bc[0])

    angle_diff = np.abs(np.sin((orientation_vec_ab - orientation_vec_bc)))
    return angle_diff <= 0.342  # 0.342 approx 20 degrees; approx. 10 degrees 0.174
    # Calculate the area of the triangle formed by the three nodes
    # The formula for the area of a triangle given three vertices is:
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    # area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    # Check if the area is within the tolerance
    # return area < tolerance


def are_nodes_neighbors(node1, node2, node3):
    # Extract the coordinates of the nodes
    x1, y1 = node1.abs_x, node1.abs_y
    x2, y2 = node2.abs_x, node2.abs_y
    x3, y3 = node3.abs_x, node3.abs_y

    vec_a = np.array([x1, y1])
    vec_b = np.array([x2, y2])
    vec_c = np.array([x3, y3])

    vec_ab = vec_b - vec_a
    vec_bc = vec_c - vec_b

    orientation_vec_ab = np.arctan2(vec_ab[1], vec_ab[0])
    orientation_vec_bc = np.arctan2(vec_bc[1], vec_bc[0])
    print(f"abs_diff: {np.abs(orientation_vec_ab - orientation_vec_bc)}")
    angle_diff = np.sin(np.abs(orientation_vec_ab - orientation_vec_bc))
    print(f"angle_diff: {angle_diff}")
    return angle_diff >= 0.766 or angle_diff <= 0.64