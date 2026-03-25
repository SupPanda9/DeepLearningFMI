import unittest
import torch
from dl_lib.nn import Sigmoid, Tanh, ReLU, LeakyReLU, Sequential


class TestSigmoid(unittest.TestCase):

    def test_when_zero_input_provided_then_returns_half(self):
        # Arrange
        sigmoid = Sigmoid()
        input_tensor = torch.tensor([0.0])
        expected = torch.tensor([0.5])

        # Act
        result = sigmoid(input_tensor)

        # Assert
        self.assertTrue(torch.allclose(result, expected), f"Expected 0.5, got {result}")

    def test_when_large_positive_and_negative_inputs_provided_then_saturates(self):
        # Arrange
        sigmoid = Sigmoid()
        input_tensor = torch.tensor([100.0, -100.0])

        # Act
        result = sigmoid(input_tensor)

        # Assert
        self.assertTrue(
            torch.isclose(result[0], torch.tensor(1.0)), "Should saturate to 1.0"
        )
        self.assertTrue(
            torch.isclose(result[1], torch.tensor(0.0)), "Should saturate to 0.0"
        )

    def test_when_multidimensional_tensor_provided_then_returns_correct_shape_and_range(
        self,
    ):
        # Arrange
        sigmoid = Sigmoid()
        input_tensor = torch.randn(5, 5)

        # Act
        result = sigmoid(input_tensor)

        # Assert
        self.assertEqual(result.shape, (5, 5))
        self.assertTrue(
            torch.all(result >= 0) and torch.all(result <= 1),
            "Output out of bounds [0, 1]",
        )


class TestTanh(unittest.TestCase):
    def test_when_zero_input_provided_then_returns_zero(self):
        # Arrange
        tanh = Tanh()
        input_tensor = torch.tensor([0.0])
        expected = torch.tensor([0.0])

        # Act
        result = tanh(input_tensor)

        # Assert
        self.assertTrue(torch.allclose(result, expected))

    def test_when_large_positive_and_negative_inputs_provided_then_saturates(self):
        # Arrange
        tanh = Tanh()
        input_tensor = torch.tensor([100.0, -100.0])

        # Act
        result = tanh(input_tensor)

        # Assert
        self.assertTrue(
            torch.isclose(result[0], torch.tensor(1.0)), "Should saturate to 1.0"
        )
        self.assertTrue(
            torch.isclose(result[1], torch.tensor(-1.0)), "Should saturate to -1.0"
        )

    def test_when_specific_values_provided_then_returns_correct_mapping(self):
        # Arrange
        tanh = Tanh()

        input_tensor = torch.tensor([1.0])
        expected = torch.tensor([0.76159415595])

        # Act
        result = tanh(input_tensor)

        # Assert
        self.assertTrue(torch.allclose(result, expected))


class TestReLU(unittest.TestCase):

    def test_when_positive_input_provided_then_returns_same_value(self):
        # Arrange
        relu = ReLU()
        input_tensor = torch.tensor([1.0, 5.5, 10.2])
        expected = torch.tensor([1.0, 5.5, 10.2])

        # Act
        result = relu(input_tensor)

        # Assert
        self.assertTrue(torch.equal(result, expected))

    def test_when_negative_input_provided_then_returns_zero(self):
        # Arrange
        relu = ReLU()
        input_tensor = torch.tensor([-1.0, -5.5, -100.0])
        expected = torch.zeros(3)

        # Act
        result = relu(input_tensor)

        # Assert
        self.assertTrue(torch.equal(result, expected))

    def test_when_mixed_input_provided_then_thresholds_correctly(self):
        # Arrange
        relu = ReLU()
        input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        expected = torch.tensor([0.0, 0.0, 1.0])

        # Act
        result = relu(input_tensor)

        # Assert
        self.assertTrue(torch.equal(result, expected))


class TestLeakyReLU(unittest.TestCase):

    def test_when_positive_input_provided_then_returns_same_value(self):
        # Arrange
        lrelu = LeakyReLU(negative_slope=0.1)
        input_tensor = torch.tensor([1.0, 5.0])
        expected = torch.tensor([1.0, 5.0])

        # Act
        result = lrelu(input_tensor)

        # Assert
        self.assertTrue(torch.equal(result, expected))

    def test_when_negative_input_provided_then_returns_scaled_value(self):
        # Arrange
        slope = 0.1
        lrelu = LeakyReLU(negative_slope=slope)
        input_tensor = torch.tensor([-1.0, -10.0])
        expected = torch.tensor([-0.1, -1])

        # Act
        result = lrelu(input_tensor)

        # Assert
        self.assertTrue(torch.allclose(result, expected))

    def test_when_default_slope_used_then_scales_correctly(self):
        # Arrange
        lrelu = LeakyReLU()
        input_tensor = torch.tensor([-2.0])
        expected = torch.tensor([-0.02])

        # Act
        result = lrelu(input_tensor)

        # Assert
        self.assertTrue(torch.allclose(result, expected))


class TestSequential(unittest.TestCase):

    def test_when_multiple_modules_provided_then_processes_in_order(self):
        # Arrange
        model = Sequential(ReLU(), Sigmoid())
        input_tensor = torch.tensor([-5.0])
        expected = torch.tensor([0.5])

        # Act
        result = model(input_tensor)

        # Assert
        self.assertTrue(torch.allclose(result, expected))

    def test_when_module_appended_then_it_is_added_to_end(self):
        # Arrange
        model = Sequential(ReLU())
        model.append(Tanh())
        input_tensor = torch.tensor([1.0])
        expected = torch.tanh(torch.tensor([1.0]))

        # Act
        result = model(input_tensor)

        # Assert
        self.assertTrue(torch.allclose(result, expected))

    def test_when_insert_used_then_module_placed_at_correct_index(self):
        # Arrange
        model = Sequential(ReLU(), Sigmoid())
        model.insert(1, Tanh())

        # Act
        input_tensor = torch.tensor([-1.0])
        result = model(input_tensor)

        # Assert
        self.assertEqual(result.item(), 0.5)

    def test_when_extend_used_then_modules_merged(self):
        # Arrange
        model1 = Sequential(ReLU())
        model2 = Sequential(Tanh())
        model1.extend(model2)

        # Assert internal list length
        self.assertEqual(len(model1._modules), 2)


if __name__ == "__main__":
    unittest.main()
