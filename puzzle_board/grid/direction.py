from enum import Enum


class Direction(Enum):
    X = 0
    NEGATIVE_Y = 1
    NEGATIVE_X = 2
    Y = 3

    def rotate(self, n=1):
        rotation_map = {
            Direction.X: Direction.Y,
            Direction.Y: Direction.NEGATIVE_X,
            Direction.NEGATIVE_X: Direction.NEGATIVE_Y,
            Direction.NEGATIVE_Y: Direction.X,
        }
        current_direction = self
        for i in range(n):
            current_direction = rotation_map[current_direction]
        return current_direction

    def rotate_right(self, n=1):
        rotation_map = {
            Direction.X: Direction.NEGATIVE_Y,
            Direction.Y: Direction.X,
            Direction.NEGATIVE_X: Direction.Y,
            Direction.NEGATIVE_Y: Direction.NEGATIVE_X
        }
        current_direction = self
        for i in range(n):
            current_direction = rotation_map[current_direction]
        return current_direction

    def rotations_needed_for(self, direction):
        if self == direction:
            return 0
        rotations = 0
        while rotations <= 4:
            rotations += 1
            current = self.rotate(rotations)
            if current == direction:
                return rotations

    def opposite(self):
        return Direction((self.value + 2) % 4)

    def is_x_axis(self):
        return self == Direction.X or self == Direction.NEGATIVE_X

    def is_y_axis(self):
        return self == Direction.Y or self == Direction.NEGATIVE_Y

    def __str__(self):
        str_map = {
            Direction.X: "X",
            Direction.NEGATIVE_X: "NX",
            Direction.Y: "Y",
            Direction.NEGATIVE_Y: "NY"
        }
        return str_map[self]
