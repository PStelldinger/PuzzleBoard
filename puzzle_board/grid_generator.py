import numpy as np

class Grid:

    rotation_matrices = np.array([[[1,0],[0,1]],[[0,1],[-1,0]],[[-1,0],[0,-1]],[[0,-1],[1,0]]])
    first_root = None

    def __init__(self, nr):
        self.id = nr
        # parent node:
        self.parent = self
        # number of subtree elements with this node as root:
        self.size = 1
        # orientation 0..3 (0:left, 1:top, 2:right, 3:bottom):
        self.orientation = 0
        # distance to the Grid root (Self reference if node is its root):
        self.offset = np.array([[0],[0]])
        # array of 4 dimensions of subtree [left,top,right,bottom] (0 if just one point):
        self.dimensions = np.array([0, 0, 0, 0])
        # array of 4 neighbors by id [left,top,right,bottom] (Self reference if not known):
        self.neighbors = np.array([self, self, self, self])
        self.nr_neighbors = 0
        # list of roots:
        if(Grid.first_root is None):
            Grid.first_root = self
            self.next_root = self
            self.prev_root = self
        else:
            self.prev_root = Grid.first_root
            self.next_root = Grid.first_root.next_root
            Grid.first_root.next_root.prev_root = self
            Grid.first_root.next_root = self
        self.next = self
        self.prev = self
            
    def reset():
        Grid.first_root = None    

#    def setNeighbors(self, left, top, right, bottom):
#        self.neighbors=[left, top, right, bottom]
#        self.nr_neighbors = 4
        
    def set_left(self, nb):
        # if self.has_neighbor(nb.id):
        #     print("Hier ist was falsch!")
        #     return
        if self.neighbors[0] == self:
            self.nr_neighbors = self.nr_neighbors + 1
        self.neighbors[0] = nb

    def set_top(self, nb):
        # if self.has_neighbor(nb.id):
        #     print("Hier ist was falsch!")
        #     return
        if self.neighbors[1] == self:
            self.nr_neighbors = self.nr_neighbors + 1
        self.neighbors[1] = nb

    def set_right(self, nb):
        # if self.has_neighbor(nb.id):
        #     print("Hier ist was falsch!")
        #     return
        if self.neighbors[2] == self:
            self.nr_neighbors = self.nr_neighbors + 1
        self.neighbors[2] = nb

    def set_bottom(self, nb):
        # if self.has_neighbor(nb.id):
        #     print("Hier ist was falsch!")
        #     return
        if self.neighbors[3] == self:
            self.nr_neighbors = self.nr_neighbors + 1
        self.neighbors[3] = nb

    def get_left(self):
        return self.neighbors[0]

    def get_top(self):
        return self.neighbors[1]

    def get_right(self):
        return self.neighbors[2]

    def get_bottom(self):
        return self.neighbors[3]
    
    def has_neighbor(self, nb):
        return self.neighbors[0].id == nb or self.neighbors[1].id == nb or self.neighbors[2].id == nb or self.neighbors[3].id == nb

    def rotate(self, rotation):
        self.orientation = (self.orientation + rotation) % 4
        self.dimensions = np.roll(self.dimensions, -rotation)
        self.neighbors = np.roll(self.neighbors.copy(), -rotation)
        self.offset = np.matmul(self.rotation_matrices[(rotation)%4], self.offset)

    def _get_root_and_update(self):
        parent = self.parent
        if parent == self:
            return self, 0, 0
        root, rotation, offset = parent._get_root_and_update()
        self.parent = root
        if rotation > 0:
            self.rotate(rotation)
        self.offset = self.offset + offset
        return root, self.orientation, self.offset

    def get_root(self):
        root, rotation, offset = self._get_root_and_update()
        return root

    def union(self, B, direction_to_B, rot_A): # direction_to_B is a vector e.g. [[0],[1]] or [[-1,0]]; rot_A is the necessary rotation of self such that it fits to B.
        
        A = self
        
        x = A.get_root()     # get the Grid root of A (self)
        y = B.get_root()     # get the Grid root of B

        if x == y:  # already in same Grid
            return
        
        y.size = y.size + x.size
        x.offset = x.offset - A.offset + direction_to_B
        x.rotate((-rot_A)%4)
        x.offset = x.offset + B.offset
        y.dimensions = np.maximum(y.dimensions, x.dimensions + [x.offset[0][0], x.offset[1][0], -x.offset[0][0], -x.offset[1][0]])
        x.parent = y
        x.prev.next = y.next
        y.next.prev = x.prev
        y.next = x
        x.prev = y
        x.prev_root.next_root = x.next_root
        x.next_root.prev_root = x.prev_root
        if(x == Grid.first_root):
            Grid.first_root = x.next_root
        
        
    def connect(self, B):
        A = self
        
        x = A.get_root()     # get the Grid root of A (self)
        y = B.get_root()     # get the Grid root of B

        if x == y:  # already in same Grid
            return False
        
        if x.size > y.size: # We assume, that y is the larger Grid. Otherwise swap
            A,B = B,A
            x,y = y,x
        
        # if  (A.neighbors[0] == B and A.neighbors[1] == B) or \
        #     (A.neighbors[0] == B and A.neighbors[2] == B) or \
        #     (A.neighbors[0] == B and A.neighbors[3] == B) or \
        #     (A.neighbors[1] == B and A.neighbors[2] == B) or \
        #     (A.neighbors[1] == B and A.neighbors[3] == B) or \
        #     (A.neighbors[2] == B and A.neighbors[3] == B):
        #         print("Das darf nicht sein!")
        # if  (B.neighbors[0] == A and B.neighbors[1] == A) or \
        #     (B.neighbors[0] == A and B.neighbors[2] == A) or \
        #     (B.neighbors[0] == A and B.neighbors[3] == A) or \
        #     (B.neighbors[1] == A and B.neighbors[2] == A) or \
        #     (B.neighbors[1] == A and B.neighbors[3] == A) or \
        #     (B.neighbors[2] == A and B.neighbors[3] == A):
        #         print("Das darf auch nicht sein!")
            
            
        if A.neighbors[0] == B:
            direction_to_B = np.array([[-1],[0]])
            rot_A = 0
        elif A.neighbors[1] == B:
            direction_to_B = np.array([[0],[-1]])
            rot_A = 3
        elif A.neighbors[2] == B:
            direction_to_B = np.array([[1],[0]])
            rot_A = 2
        elif A.neighbors[3] == B:
            direction_to_B = np.array([[0],[1]])
            rot_A = 1
        else:
            return False
            
        
        if B.neighbors[0] == A:
            rot_A = (rot_A + 2) % 4
        elif B.neighbors[1] == A:
            rot_A = (rot_A + 3) % 4
        elif B.neighbors[2] == A:
            rot_A = (rot_A + 0) % 4
        elif B.neighbors[3] == A:
            rot_A = (rot_A + 1) % 4
        else:
            return False # This has to be replaced by a check, which rotation fits best (based on real coordinates and orientations)
        
        self.union(B, direction_to_B, rot_A)
        return True
