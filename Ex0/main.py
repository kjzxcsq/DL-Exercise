from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Checkerboard example
    checker = Checker(250, 25)
    checker.draw()
    checker.show()

    # Circle example
    circle = Circle(1024, 200, (512, 256))
    circle.draw()
    circle.show()

    # Spectrum example
    spectrum = Spectrum(255)
    spectrum.draw()
    spectrum.show()

    # ImageGenerator example
    generator = ImageGenerator('exercise0_material/src_to_implement/exercise_data',
                                'exercise0_material/src_to_implement/Labels.json',
                                batch_size=50, image_size=[32, 32, 3], rotation=True, mirroring=True, shuffle=True)
    generator.show()
    generator.show()

if __name__ == '__main__':
    main()
