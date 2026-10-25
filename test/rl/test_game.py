"""Regression test for Gym reset results containing observation metadata."""
import unittest
from unittest.mock import patch

import numpy as np

from labml_nn.rl.game import Game


class TestGame(unittest.TestCase):
    def test_reset_with_info(self):
        with patch('labml_nn.rl.game.gym.make') as make:
            make.return_value.reset.return_value = (
                np.full((84, 84, 3), 42, dtype=np.uint8), {})
            make.return_value.unwrapped.ale.lives.return_value = 5
            obs = Game(7).reset()

        np.testing.assert_array_equal(obs, np.full((4, 84, 84), 42))


if __name__ == '__main__':
    unittest.main()
