from enum import Enum

from grid.kruskal.optimized.axis import Axis


class OptimizedDirection(Enum):
    X = (1, 0)
    NEGATIVE_Y = (0, -1)
    NEGATIVE_X = (-1, 0)
    Y = (0, 1)
    NO_DIRECTION = (0, 0)

    def opposite(self):
        direction_map = {
            OptimizedDirection.X: OptimizedDirection.NEGATIVE_X,
            OptimizedDirection.NEGATIVE_X: OptimizedDirection.X,
            OptimizedDirection.Y: OptimizedDirection.NEGATIVE_Y,
            OptimizedDirection.NEGATIVE_Y: OptimizedDirection.Y
        }
        return direction_map[self]

    def rotate(self, n=1):
        rotation_map = {
           OptimizedDirection.X: OptimizedDirection.Y,
           OptimizedDirection.Y: OptimizedDirection.NEGATIVE_X,
           OptimizedDirection.NEGATIVE_X: OptimizedDirection.NEGATIVE_Y,
           OptimizedDirection.NEGATIVE_Y: OptimizedDirection.X
        }
        current_direction = self
        for i in range(n):
            current_direction = rotation_map[current_direction]
        return current_direction

    def get_axis(self) -> Axis:
        if self == OptimizedDirection.X or self == OptimizedDirection.NEGATIVE_X:
            return Axis.X_AXIS
        else:
            return Axis.Y_AXIS
