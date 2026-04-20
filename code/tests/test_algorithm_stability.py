#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - test host may not have torch
    torch = None


@unittest.skipIf(torch is None, "torch is not available in the host test environment")
class AlgorithmStabilityTests(unittest.TestCase):
    def _make_algorithm(self):
        from agent_ppo.algorithm.algorithm import Algorithm

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return Algorithm(model=model, optimizer=optimizer, device=torch.device("cpu"))

    def test_first_nonfinite_tensor_returns_name_and_stats(self):
        alg = self._make_algorithm()

        invalid = alg._first_nonfinite_tensor({
            "good": torch.tensor([1.0, 2.0]),
            "bad": torch.tensor([1.0, float("nan")]),
        })

        self.assertIsNotNone(invalid)
        name, stats = invalid
        self.assertEqual(name, "bad")
        self.assertEqual(stats["shape"], [2])

    def test_normalize_advantage_returns_centered_values_when_std_tiny(self):
        alg = self._make_algorithm()

        advantage = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        valid_mask = torch.ones_like(advantage)
        normalized = alg._normalize_advantage(advantage, valid_mask, valid_mask.sum())

        self.assertTrue(torch.allclose(normalized, torch.zeros_like(advantage)))

    def test_record_invalid_batch_increments_counters_and_preserves_last_finite_loss(self):
        alg = self._make_algorithm()
        alg.total_batch_count = 5
        alg.last_finite_step = 3
        alg.last_finite_metrics = {"total_loss": 1.23}

        result = alg._record_invalid_batch("loss:total_loss", {"shape": [1]})

        self.assertEqual(alg.nan_batch_count, 1)
        self.assertEqual(alg.consecutive_invalid_batches, 1)
        self.assertEqual(result["loss_invalid"], 1.0)
        self.assertAlmostEqual(result["total_loss"], 1.23, places=5)
        self.assertAlmostEqual(result["nan_skip_rate"], 0.2, places=5)
        self.assertEqual(result["last_finite_step"], 3.0)

    def test_finalize_invalid_after_unscale_skips_optimizer_step_and_updates_scaler(self):
        alg = self._make_algorithm()
        alg.total_batch_count = 4
        alg.last_finite_step = 2
        alg.last_finite_metrics = {"total_loss": 2.5}

        class FakeScaler:
            def __init__(self):
                self.step_called = 0
                self.update_called = 0

            def step(self, optimizer):
                self.step_called += 1

            def update(self):
                self.update_called += 1

        fake_scaler = FakeScaler()
        alg.scaler = fake_scaler
        alg.optimizer.zero_grad = unittest.mock.Mock()

        result = alg._finalize_invalid_after_unscale("gradients", {"shape": None})

        self.assertEqual(fake_scaler.step_called, 0)
        self.assertEqual(fake_scaler.update_called, 1)
        self.assertGreaterEqual(alg.optimizer.zero_grad.call_count, 1)
        self.assertEqual(result["loss_invalid"], 1.0)
        self.assertEqual(alg.nan_batch_count, 1)

    def test_battery_process_cost_mean_uses_constraint_cost_not_reward_survive(self):
        from agent_ppo.conf.conf import Config

        batch = {
            "constraint_battery_process_cost": torch.tensor([[0.10] * Config.SEQ_CHUNK_LEN], dtype=torch.float32),
            "reward_survive": torch.tensor([[5.0] * Config.SEQ_CHUNK_LEN], dtype=torch.float32),
        }
        valid_mask = torch.ones((1, Config.SEQ_LEARN_LEN), dtype=torch.float32)
        valid_count = valid_mask.sum()

        mean = self._make_algorithm()._battery_process_cost_mean(
            batch,
            slice(Config.SEQ_BURN_IN, Config.SEQ_CHUNK_LEN),
            valid_mask,
            valid_count,
        )

        self.assertAlmostEqual(float(mean.item()), 0.10, places=5)


if __name__ == "__main__":
    unittest.main()
