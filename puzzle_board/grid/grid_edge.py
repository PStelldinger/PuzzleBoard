import matplotlib.pyplot as plt


class GridEdge(object):

    def __init__(self, coords_one, coords_two, image):
        self.coords_one = coords_one
        self.coords_two = coords_two
        self.decoded_value = self.decode_value(image)

    def __hash__(self):
        return hash(hash(self.coords_one) + hash(self.coords_two))

    def __eq__(self, other):
        if not isinstance(other, GridEdge):
            return False
        return ((self.coords_one == other.coords_one
                and self.coords_two == other.coords_two)
                or (self.coords_one == other.coords_two
                    and self.coords_two == other.coords_one))

    def decode_value(self, image):
        # von der mitte aus 1/24 nach coords_one und 1/24 nach coords_two und davon average wert?
        # vorerst: nur der Pixel in der Mitte. Muss eventuell noch an Bild max und min grauwerte angepasst werden!
        img_x_pos = int((self.coords_one[0] + self.coords_two[0]) / 2)
        img_y_pos = int((self.coords_one[1] + self.coords_two[1]) / 2)
        if image[img_y_pos, img_x_pos] > 0.5:
            return 1
        else:
            return 0

    def plot(self, bg_image, plot_lines=True):
        plt.imshow(bg_image, cmap="gray")
        if plot_lines:
            plt.plot([self.coords_one[0], self.coords_two[0]], [self.coords_one[1], self.coords_two[1]], color="blue")
        plt.text((self.coords_one[0] + self.coords_two[0])/ 2, (self.coords_one[1] + self.coords_two[1])/2, str(self.decoded_value), c="green" if self.decoded_value == 0 else "red")
