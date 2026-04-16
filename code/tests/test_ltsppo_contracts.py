#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib
import sys
import types
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ppo.conf.conf import Config

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

if np is None:  # pragma: no cover
    raise unittest.SkipTest("numpy is not installed in this test environment")


def _install_create_cls_stub():
    common_python_mod = types.ModuleType("common_python")
    utils_mod = types.ModuleType("common_python.utils")
    common_func_mod = types.ModuleType("common_python.utils.common_func")

    def create_cls(name, **defaults):
        attrs = dict(defaults)

        def __init__(self, **kwargs):
            for key, default in defaults.items():
                setattr(self, key, kwargs.get(key, default))

        attrs["__init__"] = __init__
        return type(name, (), attrs)

    common_func_mod.create_cls = create_cls
    utils_mod.common_func = common_func_mod
    common_python_mod.utils = utils_mod

    sys.modules["common_python"] = common_python_mod
    sys.modules["common_python.utils"] = utils_mod
    sys.modules["common_python.utils.common_func"] = common_func_mod


def _import_definition_module():
    _install_create_cls_stub()
    sys.modules.pop("agent_ppo.feature.definition", None)
    return importlib.import_module("agent_ppo.feature.definition")


def _build_step_records(total_steps):
    records = []
    for idx in range(total_steps):
        records.append(
            {
                "obs": np.full((Config.DIM_OF_OBSERVATION,), fill_value=float(idx + 1), dtype=np.float32),
                "legal_action": np.full((Config.ACTION_NUM,), fill_value=float((idx % 2) + 1), dtype=np.float32),
                "act": idx,
                "prob": np.full((Config.ACTION_NUM,), fill_value=1.0 / Config.ACTION_NUM, dtype=np.float32),
                "done": 1.0 if idx == total_steps - 1 else 0.0,
                "reward_clean": 0.1 * idx,
                "reward_survive": 0.2 * idx,
                "value_clean": 0.0,
                "value_survive": 0.0,
                "mode_teacher": idx % Config.MODE_NUM,
                "route_anchor_teacher": idx % Config.ROUTE_ANCHOR_DIM,
                "target_teacher": idx % Config.TARGET_DIM,
                "mode_teacher_mask": 1.0,
                "route_anchor_teacher_mask": 1.0 if idx % 2 == 0 else 0.0,
                "target_teacher_mask": 0.0 if idx % 3 == 0 else 1.0,
                "return_action_teacher": idx % Config.ACTION_NUM,
                "return_action_teacher_mask": 1.0 if idx % 4 else 0.0,
                "battery_risk_label": float(idx % 2),
                "collision_risk_label": float((idx + 1) % 2),
                "fallback_mask": 0.0,
                "expert_weight": float(idx) / 100.0,
            }
        )
    return records


class LtsppoConfigAndDefinitionContractsTests(unittest.TestCase):
    def test_config_and_definition_dims_stay_derived_from_config(self):
        definition = _import_definition_module()

        self.assertEqual(Config.FEATURE_LEN, sum(Config.FEATURES))
        self.assertEqual(Config.DIM_OF_OBSERVATION, sum(Config.FEATURE_SPLIT_SHAPE))
        self.assertEqual(Config.DIM_OF_OBSERVATION, Config.FEATURE_LEN)

        expected_sample_dims = {
            "obs": Config.DIM_OF_OBSERVATION * Config.SEQ_CHUNK_LEN,
            "legal_action": Config.ACTION_NUM * Config.SEQ_CHUNK_LEN,
            "act": Config.SEQ_CHUNK_LEN,
            "reward_clean": Config.SEQ_CHUNK_LEN,
            "reward_survive": Config.SEQ_CHUNK_LEN,
            "done": Config.SEQ_CHUNK_LEN,
            "value_clean": Config.SEQ_CHUNK_LEN,
            "value_survive": Config.SEQ_CHUNK_LEN,
            "advantage_clean": Config.SEQ_CHUNK_LEN,
            "advantage_survive": Config.SEQ_CHUNK_LEN,
            "prob": Config.ACTION_NUM * Config.SEQ_CHUNK_LEN,
            "mode_teacher": Config.SEQ_CHUNK_LEN,
            "route_anchor_teacher": Config.SEQ_CHUNK_LEN,
            "target_teacher": Config.SEQ_CHUNK_LEN,
            "mode_teacher_mask": Config.SEQ_CHUNK_LEN,
            "route_anchor_teacher_mask": Config.SEQ_CHUNK_LEN,
            "target_teacher_mask": Config.SEQ_CHUNK_LEN,
            "return_action_teacher": Config.SEQ_CHUNK_LEN,
            "return_action_teacher_mask": Config.SEQ_CHUNK_LEN,
            "battery_risk_label": Config.SEQ_CHUNK_LEN,
            "collision_risk_label": Config.SEQ_CHUNK_LEN,
            "fallback_mask": Config.SEQ_CHUNK_LEN,
            "expert_weight": Config.EXPERT_WEIGHT_DIM,
        }

        for field_name, expected_dim in expected_sample_dims.items():
            self.assertEqual(getattr(definition.SampleData, field_name), expected_dim)

    def test_feature_vector_contract_excludes_legal_action(self):
        self.assertEqual(Config.DIM_OF_OBSERVATION, sum(Config.FEATURE_SPLIT_SHAPE))
        self.assertNotEqual(Config.DIM_OF_OBSERVATION, sum(Config.FEATURE_SPLIT_SHAPE) + Config.ACTION_NUM)
        self.assertEqual(Config.SAMPLE_OBS_DIM, Config.DIM_OF_OBSERVATION * Config.SEQ_CHUNK_LEN)
        self.assertEqual(Config.SAMPLE_LEGAL_ACTION_DIM, Config.ACTION_NUM * Config.SEQ_CHUNK_LEN)


class LtsppoSampleProcessChunkingTests(unittest.TestCase):
    def test_sample_process_chunk_size_stride_and_padding_semantics(self):
        definition = _import_definition_module()
        chunk_len = Config.SEQ_CHUNK_LEN
        stride = Config.SEQ_STRIDE
        total_steps = chunk_len + 5
        records = _build_step_records(total_steps)

        samples = definition.sample_process(records, episode_idx=0)

        self.assertEqual(len(samples), 2)

        first = samples[0]
        second = samples[1]

        first_obs = first.obs.reshape(chunk_len, Config.DIM_OF_OBSERVATION)
        second_obs = second.obs.reshape(chunk_len, Config.DIM_OF_OBSERVATION)
        second_legal = second.legal_action.reshape(chunk_len, Config.ACTION_NUM)

        expected_second_real_len = total_steps - stride
        self.assertEqual(expected_second_real_len, 13)

        self.assertEqual(int(first.act[0]), 0)
        self.assertEqual(int(first.act[-1]), chunk_len - 1)
        self.assertEqual(int(second.act[0]), stride)
        self.assertEqual(int(second.act[expected_second_real_len - 1]), total_steps - 1)

        self.assertAlmostEqual(float(second_obs[0, 0]), float(stride + 1))
        self.assertAlmostEqual(float(first_obs[stride, 0]), float(second_obs[0, 0]))

        self.assertTrue(np.allclose(second_obs[expected_second_real_len:, :], 0.0))
        self.assertTrue(np.allclose(second_legal[expected_second_real_len:, :], 0.0))
        self.assertTrue(np.allclose(second.act[expected_second_real_len:], 0))

        self.assertEqual(float(second.done[expected_second_real_len - 1]), 1.0)
        self.assertTrue(np.allclose(second.done[expected_second_real_len:], 1.0))
        self.assertTrue(np.allclose(second.mode_teacher_mask[:expected_second_real_len], 1.0))
        self.assertTrue(np.allclose(second.mode_teacher_mask[expected_second_real_len:], 0.0))
        self.assertAlmostEqual(float(second.route_anchor_teacher_mask[0]), 1.0)
        self.assertAlmostEqual(float(second.route_anchor_teacher_mask[1]), 0.0)
        self.assertTrue(np.allclose(second.route_anchor_teacher_mask[expected_second_real_len:], 0.0))
        self.assertAlmostEqual(float(second.target_teacher_mask[0]), 0.0)
        self.assertTrue(np.allclose(second.target_teacher_mask[1:expected_second_real_len], 1.0))
        self.assertTrue(np.allclose(second.target_teacher_mask[expected_second_real_len:], 0.0))
        self.assertAlmostEqual(float(second.return_action_teacher_mask[0]), 1.0)
        self.assertAlmostEqual(float(second.return_action_teacher_mask[4]), 1.0)
        self.assertTrue(np.allclose(second.return_action_teacher_mask[expected_second_real_len:], 0.0))
        self.assertTrue(np.allclose(second.fallback_mask[:expected_second_real_len], 0.0))
        self.assertTrue(np.allclose(second.fallback_mask[expected_second_real_len:], 1.0))


class LtsppoModelOutputShapeTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed in this test environment")
    def test_model_output_shapes_for_single_step_and_sequence(self):
        from agent_ppo.model.model import Model

        model = Model(device=torch.device("cpu"))
        model.set_eval_mode()

        with torch.no_grad():
            step_batch = torch.randn(3, Config.DIM_OF_OBSERVATION, dtype=torch.float32)
            step_outputs = model(step_batch)

            self.assertEqual(step_outputs["policy_logits"].shape, (3, Config.ACTION_NUM))
            self.assertEqual(step_outputs["mode_logits"].shape, (3, Config.MODE_NUM))
            self.assertEqual(step_outputs["route_anchor_logits"].shape, (3, Config.ROUTE_ANCHOR_DIM))
            self.assertEqual(step_outputs["target_logits"].shape, (3, Config.TARGET_DIM))
            self.assertEqual(step_outputs["return_action_logits"].shape, (3, Config.ACTION_NUM))
            self.assertEqual(step_outputs["value_clean"].shape, (3, 1))
            self.assertEqual(step_outputs["value_survive"].shape, (3, 1))
            self.assertEqual(step_outputs["aux_battery_risk"].shape, (3, 1))
            self.assertEqual(step_outputs["aux_collision_risk"].shape, (3, 1))
            self.assertEqual(step_outputs["mode_probs"].shape, (3, Config.MODE_NUM))
            self.assertEqual(step_outputs["route_anchor_probs"].shape, (3, Config.ROUTE_ANCHOR_DIM))
            self.assertEqual(step_outputs["target_probs"].shape, (3, Config.TARGET_DIM))
            self.assertEqual(step_outputs["next_rnn_state"].shape, (1, 3, Config.RNN_HIDDEN_DIM))

            seq_batch = torch.randn(2, 5, Config.DIM_OF_OBSERVATION, dtype=torch.float32)
            seq_outputs = model(seq_batch)

            self.assertEqual(seq_outputs["policy_logits"].shape, (2, 5, Config.ACTION_NUM))
            self.assertEqual(seq_outputs["mode_logits"].shape, (2, 5, Config.MODE_NUM))
            self.assertEqual(seq_outputs["route_anchor_logits"].shape, (2, 5, Config.ROUTE_ANCHOR_DIM))
            self.assertEqual(seq_outputs["target_logits"].shape, (2, 5, Config.TARGET_DIM))
            self.assertEqual(seq_outputs["return_action_logits"].shape, (2, 5, Config.ACTION_NUM))
            self.assertEqual(seq_outputs["value_clean"].shape, (2, 5, 1))
            self.assertEqual(seq_outputs["value_survive"].shape, (2, 5, 1))
            self.assertEqual(seq_outputs["aux_battery_risk"].shape, (2, 5, 1))
            self.assertEqual(seq_outputs["aux_collision_risk"].shape, (2, 5, 1))
            self.assertEqual(seq_outputs["mode_probs"].shape, (2, 5, Config.MODE_NUM))
            self.assertEqual(seq_outputs["route_anchor_probs"].shape, (2, 5, Config.ROUTE_ANCHOR_DIM))
            self.assertEqual(seq_outputs["target_probs"].shape, (2, 5, Config.TARGET_DIM))
            self.assertEqual(seq_outputs["next_rnn_state"].shape, (1, 2, Config.RNN_HIDDEN_DIM))


if __name__ == "__main__":
    unittest.main()
