import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from privacy_integration.legacy_constants import EPSILON_MAX
from privacy_integration.training.dp_trainer import DPTrainingWrapper


def make_model(feature_dim: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(feature_dim, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )


def make_loader(n: int = 200, feature_dim: int = 32, batch_size: int = 32) -> DataLoader:
    X = torch.randn(n, feature_dim)
    y = torch.randint(0, 2, (n, 1)).float()
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


class TestDPTrainingWrapper(unittest.TestCase):

    def _make_wrapper(self, target_epsilon: float = 3.0, epochs: int = 3) -> DPTrainingWrapper:
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = make_loader()
        return DPTrainingWrapper(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=target_epsilon,
            epochs=epochs,
            device="cpu",
        )

    def test_initialises_without_error(self):
        wrapper = self._make_wrapper()
        self.assertIsNotNone(wrapper)

    def test_noise_multiplier_is_positive(self):
        wrapper = self._make_wrapper()
        self.assertGreater(wrapper.noise_multiplier, 0)

    def test_higher_epsilon_gives_lower_noise(self):
        w_low = self._make_wrapper(target_epsilon=1.0)
        w_high = self._make_wrapper(target_epsilon=8.0)
        self.assertGreater(w_low.noise_multiplier, w_high.noise_multiplier)

    def test_training_completes_without_error(self):
        wrapper = self._make_wrapper(epochs=2)
        result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=False)
        self.assertGreater(result.total_steps, 0)

    def test_budget_respected_after_training(self):
        wrapper = self._make_wrapper(target_epsilon=3.0, epochs=3)
        result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=False)
        self.assertTrue(result.final_epsilon <= EPSILON_MAX)

    def test_budget_log_has_one_row_per_step(self):
        wrapper = self._make_wrapper(epochs=2)
        result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=False)
        self.assertEqual(len(result.budget_log), result.total_steps)

    def test_budget_log_epsilon_is_monotonically_increasing(self):
        wrapper = self._make_wrapper(epochs=2)
        result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=False)
        eps_values = [r.cumulative_eps for r in result.budget_log]
        for i in range(1, len(eps_values)):
            self.assertGreaterEqual(eps_values[i], eps_values[i - 1])

    def test_halted_early_is_false_when_budget_not_exceeded(self):
        wrapper = self._make_wrapper(target_epsilon=3.0, epochs=3)
        result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=False)
        self.assertFalse(result.halted_early)

    def test_budget_halt_triggers_when_ceiling_very_low(self):
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = make_loader(n=500)
        wrapper = DPTrainingWrapper(
            model=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=9.0,
            epochs=50,
            epsilon_max=0.1,
            device="cpu",
        )
        result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=False)
        self.assertTrue(result.halted_early)

    def test_generated_at_present(self):
        wrapper = self._make_wrapper(epochs=1)
        result = wrapper.train(criterion=nn.BCEWithLogitsLoss(), verbose=False)
        self.assertIsNotNone(result.generated_at)


if __name__ == "__main__":
    unittest.main()
