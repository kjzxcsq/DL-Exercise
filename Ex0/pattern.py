import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        num_tiles = self.resolution // (2 * self.tile_size)
        tile_pattern = np.concatenate((np.zeros(self.tile_size), np.ones(self.tile_size)))
        row_pattern = np.tile(tile_pattern, (self.tile_size, num_tiles))
        board_pattern = np.tile(np.vstack((row_pattern, 1 - row_pattern)), (num_tiles, 1))
        self.output = board_pattern
        return self.output.copy()

    def show(self):
        if self.output is None:
            print("Please call draw() first.")
        else:
            plt.imshow(self.output, cmap='gray')
            plt.show()

# checker = Checker(8, 2)
# checker.draw()
# checker.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x = np.arange(0, self.resolution)
        y = np.arange(0, self.resolution)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt((X - self.position[0]) ** 2 + (Y - self.position[1]) ** 2)
        self.output = np.where(dist <= self.radius, 1, 0)
        return self.output.copy()

    def show(self):
        if self.output is None:
            print("Please call draw() first.")
        else:
            plt.imshow(self.output, cmap='gray')
            plt.show()

# circle = Circle(8, 3, (4, 2))
# circle.draw()
# circle.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros([self.resolution, self.resolution, 3])
        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(-1, 1)
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)
        return self.output.copy()

    def show(self):
        if self.output is None:
            print("Please call draw() first.")
        else:
            plt.imshow(self.output)
            plt.show()

# spectrum = Spectrum(256)
# spectrum.draw()
# spectrum.show()