import unittest
from task01 import simulate_random_walk


class TestSimulateRandomWalk(unittest.TestCase):

    def test_when_seed_is_123_then_random_float_matches_expected_value(self):
        random_float, _, _, _, _ = simulate_random_walk()
        self.assertEqual(random_float, 0.6823518632481435)

    def test_when_seed_is_123_then_first_dice_roll_matches_expected_value(self):
        _, random_integer1, _, _, _ = simulate_random_walk()
        self.assertEqual(random_integer1, 4)

    def test_when_seed_is_123_then_second_dice_roll_matches_expected_value(self):
        _, _, random_integer2, _, _ = simulate_random_walk()
        self.assertEqual(random_integer2, 1)

    def test_when_dice_is_6_then_additional_roll_updates_step_correctly(self):
        _, _, _, dice, step = simulate_random_walk()
        self.assertEqual(dice, 6)
        self.assertEqual(step, 52)


if __name__ == "__main__":
    unittest.main()
