class Edge:

    def __init__(self, weight, node_one, node_two):
        self.weight = weight
        self.node_one = node_one
        self.node_two = node_two

    def __lt__(self, other):
        return self.weight < other.weight

    def __hash__(self):
        return hash((self.weight, self.node_one, self.node_two))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.weight == other.weight
                and ((self.node_one == other.node_one
                      and self.node_two == other.node_two)
                     or (self.node_one == other.node_two
                         and self.node_two == other.node_one)))
