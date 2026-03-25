import unittest
from task11 import SquareModel
import numpy as np

TOLERANCE = 0.5


class TestPerceptron(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        x = np.arange(-25, 25.5, 0.5)
        y = x**2
        dataset = np.column_stack((x, y))
        cls.model = SquareModel()
        cls.model.train(dataset)

    def test_when_input_2_then_output_4(self):
        predicted = self.model(2)
        expected = 4

        self.assertAlmostEqual(predicted, expected, delta=TOLERANCE)

    def test_when_input_0_then_output_0(self):
        predicted = self.model(0)
        expected = 0

        self.assertAlmostEqual(predicted, expected, delta=TOLERANCE)

    def test_when_input_1_then_output_1(self):
        predicted = self.model(1)
        expected = 1

        self.assertAlmostEqual(predicted, expected, delta=TOLERANCE)

    def test_when_input_7_then_output_49(self):
        predicted = self.model(7)
        expected = 49

        self.assertAlmostEqual(predicted, expected, delta=TOLERANCE)

    def test_when_input_23_then_output_529(self):
        predicted = self.model(23)
        expected = 529

        self.assertAlmostEqual(predicted, expected, delta=TOLERANCE)

    def test_when_input_negative_then_output_positive_square(self):
        predicted = self.model(-12)
        expected = 144

        self.assertAlmostEqual(predicted, expected, delta=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
