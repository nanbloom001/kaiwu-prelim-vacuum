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
                "route_phase_action_teacher": idx % Config.ACTION_NUM,
                "route_phase_action_teacher_mask": 1.0 if idx % 5 else 0.0,
                "battery_risk_label": float(idx % 2),
                "collision_risk_label": float((idx + 1) % 2),
                "constraint_battery_process_cost": 0.05 * idx,
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
            "route_phase_action_teacher": Config.SEQ_CHUNK_LEN,
            "route_phase_action_teacher_mask": Config.SEQ_CHUNK_LEN,
            "battery_risk_label": Config.SEQ_CHUNK_LEN,
            "collision_risk_label": Config.SEQ_CHUNK_LEN,
            "constraint_battery_process_cost": Config.SEQ_CHUNK_LEN,
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

    def test_teacher_scale_anneals_to_nonzero_floor(self):
        definition = _import_definition_module()
        force_until = Config.TEACHER_FORCE_UNTIL_EPISODE
        anneal_end = Config.TEACHER_ANNEAL_END_EPISODE
        min_scale = Config.TEACHER_MIN_SCALE

        self.assertEqual(definition._teacher_scale(force_until), 1.0)

        middle_episode = (force_until + anneal_end) // 2
        middle_scale = definition._teacher_scale(middle_episode)
        self.assertGreater(middle_scale, min_scale)
        self.assertLess(middle_scale, 1.0)

        self.assertAlmostEqual(definition._teacher_scale(anneal_end), min_scale)
        self.assertAlmostEqual(definition._teacher_scale(anneal_end + 5000), min_scale)


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
        self.assertEqual(expected_second_real_len, 9)

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
        expected_target_mask = np.array(
            [0.0 if ((stride + offset) % 3 == 0) else 1.0 for offset in range(expected_second_real_len)],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(second.target_teacher_mask[:expected_second_real_len], expected_target_mask))
        self.assertTrue(np.allclose(second.target_teacher_mask[expected_second_real_len:], 0.0))
        expected_return_mask = np.array(
            [0.0 if ((stride + offset) % 4 == 0) else 1.0 for offset in range(expected_second_real_len)],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(second.return_action_teacher_mask[:expected_second_real_len], expected_return_mask))
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


class LtsppoResumeCompatibleBehaviorTests(unittest.TestCase):
    def test_reward_process_emits_charger_access_bonuses_when_route_knowledge_improves(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.step_no = 2
        prep.current_mode = prep.MODE_HARVEST
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 3
        prep.local_frontier_density = 0.2
        prep.nearest_npc_dist = 20.0
        prep.battery = 300
        prep.battery_max = 340
        prep.charge_count = 0
        prep.just_charged = 0.0
        prep.future_recoverability_score = 1.0
        prep.path_cross_count_50 = 1
        prep.coverage_efficiency_20 = 0.95
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep._last_action = -1
        prep._prev_all_charger_known_path_count = 0.0
        prep._prev_unknown_on_target_path_ratio = 0.75
        prep._prev_planner_best_target_route_diversity = 0.0

        guidance = {
            "slack": 24.0,
            "margin": 24.0,
            "all_charger_known_path_count": 1.0,
            "unknown_path_ratio": 0.20,
            "planner_best_target_route_diversity": 2.0,
            "target_reliable": True,
            "anchor_reliable": True,
            "on_charger": False,
            "suggested_action": None,
        }
        prep._get_guidance = lambda: guidance
        prep._get_teacher_guidance = lambda: None

        _, components = prep.reward_process()

        self.assertGreater(components["charger_access_discovery_bonus"], 0.0)
        self.assertGreaterEqual(components["charger_access_probe_bonus"], 0.0)

    def test_reward_process_emits_probe_bonus_when_route_knowledge_is_weak(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.step_no = 2
        prep.current_mode = prep.MODE_CONTRACT
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 4
        prep.local_frontier_density = 0.2
        prep.nearest_npc_dist = 20.0
        prep.battery = 280
        prep.battery_max = 340
        prep.charge_count = 0
        prep.just_charged = 0.0
        prep.future_recoverability_score = 1.0
        prep.path_cross_count_50 = 1
        prep.coverage_efficiency_20 = 0.95
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep._last_action = -1
        prep._prev_all_charger_known_path_count = 0.0
        prep._prev_unknown_on_target_path_ratio = 0.8
        prep._prev_planner_best_target_route_diversity = 0.0

        guidance = {
            "slack": 18.0,
            "margin": 18.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 0.0,
            "unknown_path_ratio": 0.55,
            "planner_best_target_route_diversity": 0.0,
            "target_reliable": False,
            "anchor_reliable": False,
            "on_charger": False,
            "suggested_action": None,
        }
        prep._get_guidance = lambda: guidance
        prep._get_teacher_guidance = lambda: None

        _, components = prep.reward_process()

        self.assertGreater(components["charger_access_probe_bonus"], 0.0)

    def test_reward_process_penalizes_skipping_needed_charge_when_critical_even_away_from_charger(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.step_no = 2
        prep.current_mode = prep.MODE_HARVEST
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.local_frontier_density = 0.05
        prep.nearest_npc_dist = 20.0
        prep.nearest_charger_dist = 5.0
        prep.battery = 20
        prep.battery_max = 200
        prep.charge_count = 0
        prep.just_charged = 0.0
        prep.future_recoverability_score = -0.2
        prep.path_cross_count_50 = 1
        prep.coverage_efficiency_20 = 0.95
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep._last_action = -1
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.2
        prep._prev_planner_best_target_route_diversity = 1.0
        prep._last_target_distance = 10.0
        prep.current_target_dist = 10.0
        prep.anchor_return_dist = 10.0
        prep._last_astar_dist = 10.0
        prep._astar_dist = 10.0

        guidance = {
            "slack": -6.0,
            "margin": -2.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.15,
            "planner_best_target_route_diversity": 1.0,
            "target_reliable": True,
            "anchor_reliable": True,
            "on_charger": False,
            "suggested_action": 0,
        }
        prep._get_guidance = lambda: guidance
        prep._get_teacher_guidance = lambda: None

        _, components = prep.reward_process()

        self.assertLess(components["skip_needed_charge_penalty"], 0.0)

    def test_reward_process_adds_continuous_return_progress_shaping_when_return_is_reliable(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.step_no = 2
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.local_frontier_density = 0.05
        prep.nearest_npc_dist = 20.0
        prep.nearest_charger_dist = 4.0
        prep.last_nearest_charger_dist = 7.0
        prep.battery = 36
        prep.battery_max = 200
        prep.charge_count = 0
        prep.just_charged = 0.0
        prep.future_recoverability_score = -0.1
        prep.path_cross_count_50 = 1
        prep.coverage_efficiency_20 = 0.95
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep._last_action = 2
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.12
        prep._prev_planner_best_target_route_diversity = 1.0
        prep._last_target_distance = 9.0
        prep.current_target_dist = 6.0
        prep.anchor_return_dist = 9.0
        prep.last_charger_slack = -2.0
        prep.charger_slack = 4.0
        prep._last_astar_dist = 9.0
        prep._astar_dist = 6.0

        guidance = {
            "slack": 4.0,
            "margin": 2.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.10,
            "planner_best_target_route_diversity": 1.0,
            "target_reliable": True,
            "anchor_reliable": True,
            "return_action_reliable": True,
            "on_charger": False,
            "suggested_action": 2,
        }
        prep._get_guidance = lambda: guidance
        prep._get_teacher_guidance = lambda: None

        _, components = prep.reward_process()

        self.assertGreater(components["return_progress_shaping_bonus"], 0.0)
        self.assertAlmostEqual(components["high_need_return_stall_penalty"], 0.0, places=6)

    def test_reward_process_penalizes_high_need_return_stall_without_progress(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.step_no = 2
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.local_frontier_density = 0.0
        prep.nearest_npc_dist = 20.0
        prep.nearest_charger_dist = 9.0
        prep.last_nearest_charger_dist = 6.0
        prep.battery = 18
        prep.battery_max = 200
        prep.charge_count = 0
        prep.just_charged = 0.0
        prep.future_recoverability_score = -0.25
        prep.path_cross_count_50 = 1
        prep.coverage_efficiency_20 = 0.95
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep._last_action = 0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.15
        prep._prev_planner_best_target_route_diversity = 1.0
        prep._last_target_distance = 8.0
        prep.current_target_dist = 8.5
        prep.anchor_return_dist = 8.0
        prep.last_charger_slack = -4.0
        prep.charger_slack = -8.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.5

        guidance = {
            "slack": -8.0,
            "margin": -3.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.12,
            "planner_best_target_route_diversity": 1.0,
            "target_reliable": True,
            "anchor_reliable": True,
            "return_action_reliable": True,
            "on_charger": False,
            "suggested_action": 3,
        }
        prep._get_guidance = lambda: guidance
        prep._get_teacher_guidance = lambda: None

        _, components = prep.reward_process()

        self.assertLess(components["high_need_return_stall_penalty"], 0.0)
        self.assertAlmostEqual(components["return_progress_shaping_bonus"], 0.0, places=6)

    def test_reward_process_keeps_route_and_target_masks_zero_when_unreliable_under_low_battery(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.local_frontier_density = 0.0
        prep.nearest_npc_dist = 20.0
        prep.nearest_charger_dist = 8.0
        prep.last_nearest_charger_dist = 8.0
        prep.battery = 20
        prep.battery_max = 200
        prep.charge_count = 0
        prep.just_charged = 0.0
        prep.future_recoverability_score = -0.2
        prep.path_cross_count_50 = 0
        prep.coverage_efficiency_20 = 0.9
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 5
        prep._last_action = 0
        prep._prev_all_charger_known_path_count = 0.0
        prep._prev_unknown_on_target_path_ratio = 0.3
        prep._prev_planner_best_target_route_diversity = 0.0
        prep._last_target_distance = 10.0
        prep.current_target_dist = 10.0
        prep.anchor_return_dist = 10.0
        prep.last_charger_slack = -3.0
        prep.charger_slack = -6.0
        prep._last_astar_dist = 10.0
        prep._astar_dist = 10.0

        guidance = {
            "slack": -6.0,
            "margin": -2.0,
            "all_charger_known_path_count": 0.0,
            "planner_topk_reachable_count": 0.0,
            "unknown_path_ratio": 0.3,
            "planner_best_target_route_diversity": 0.0,
            "target_reliable": False,
            "anchor_reliable": False,
            "return_action_reliable": True,
            "on_charger": False,
            "suggested_action": 3,
        }
        prep._get_guidance = lambda: guidance
        prep._get_teacher_guidance = lambda: {
            "route_mode": "return",
            "route_anchor": (0, 0),
            "target": (0, 0),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 0.0,
            "target_teacher_mask": 0.0,
            "return_action": 3,
            "return_action_teacher_mask": 1.0,
        }

        _, components = prep.reward_process()

        self.assertAlmostEqual(components["route_anchor_teacher_mask"], 0.0, places=6)
        self.assertAlmostEqual(components["target_teacher_mask"], 0.0, places=6)
        self.assertGreaterEqual(components["return_action_teacher_mask"], 0.8)

    def test_reward_process_emits_route_phase_action_teacher_for_reliable_contract_return(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_CONTRACT
        prep.battery = 40
        prep.battery_max = 200
        prep.charger_slack = -2.0
        prep.last_charger_slack = -1.0
        prep.nearest_charger_dist = 8.0
        prep.last_nearest_charger_dist = 7.0
        prep.future_recoverability_score = 0.05
        prep.anchor_return_dist = 8.0
        prep.last_anchor_return_dist = 9.0
        prep.current_target = (0, 0)
        prep.last_target = (0, 0)
        prep._last_action = 1
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.local_frontier_density = 0.0
        prep.path_cross_count_50 = 0.0
        prep.coverage_efficiency_20 = 1.0
        prep.dirty_adjacent = 0
        prep.just_charged = 0.0
        prep.charge_count = 0
        prep._get_guidance = lambda: {
            "slack": -2.0,
            "margin": 5.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.05,
            "planner_best_target_route_diversity": 1.0,
            "target_reliable": False,
            "anchor_reliable": False,
            "return_action_reliable": True,
            "on_charger": False,
            "suggested_action": 3,
        }
        prep._get_teacher_guidance = lambda: {
            "route_mode": "return",
            "route_anchor": (0, 0),
            "target": (0, 0),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 0.0,
            "target_teacher_mask": 0.0,
            "return_action": 3,
            "return_action_teacher_mask": 1.0,
        }

        _, components = prep.reward_process()

        self.assertEqual(components["route_phase_action_teacher"], 3)
        self.assertGreaterEqual(components["route_phase_action_teacher_mask"], 0.8)

    def test_reward_process_emits_route_phase_action_teacher_for_anchor_backed_contract_return(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.battery = 70
        prep.battery_max = 200
        prep.charger_slack = 3.0
        prep.last_charger_slack = 4.0
        prep.nearest_charger_dist = 8.0
        prep.last_nearest_charger_dist = 9.0
        prep.future_recoverability_score = 0.30
        prep.anchor_return_dist = 8.0
        prep.last_anchor_return_dist = 9.0
        prep.current_target = (0, 0)
        prep.last_target = (0, 0)
        prep._last_action = 1
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.local_frontier_density = 0.0
        prep.path_cross_count_50 = 0.0
        prep.coverage_efficiency_20 = 1.0
        prep.dirty_adjacent = 0
        prep.just_charged = 0.0
        prep.charge_count = 1
        prep._get_guidance = lambda: {
            "slack": 3.0,
            "margin": 12.0,
            "all_charger_known_path_count": 2.0,
            "planner_topk_reachable_count": 2.0,
            "unknown_path_ratio": 0.05,
            "planner_best_target_route_diversity": 1.0,
            "target_reliable": False,
            "anchor_reliable": True,
            "return_action_reliable": False,
            "on_charger": False,
            "suggested_action": 2,
        }
        prep._get_teacher_guidance = lambda: {
            "route_mode": "return",
            "route_anchor": (0, 0),
            "target": (0, 0),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 1.0,
            "target_teacher_mask": 0.0,
            "return_action": 2,
            "return_action_teacher_mask": 0.0,
        }

        _, components = prep.reward_process()

        self.assertEqual(components["route_phase_action_teacher"], 2)
        self.assertAlmostEqual(components["route_phase_action_teacher_mask"], 0.6, places=6)

    def test_reward_process_applies_contextual_cleaning_scale_and_margin_pressure(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 1
        prep.consecutive_clean_steps = 5
        prep.cur_visit_count = 3
        prep.wall_adjacent = 2
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.0
        prep.same_region_streak = 8
        prep.path_cross_count_50 = 14
        prep.no_progress_steps = 9
        prep.actual_legal_ratio = 0.5
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 1.0
        prep.battery = 48
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 40
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.route_anchor_center = (32, 32)
        prep.future_recoverability_score = -0.05
        prep._prev_future_recoverability_score = 0.0
        prep.current_target_dist = 12.0
        prep._last_target_distance = 11.0
        prep.return_stall_ema = 0.8
        prep._last_astar_dist = 11.0
        prep._astar_dist = 12.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.95
        prep._last_action = 0

        guidance = {
            "slack": 5.0,
            "margin": 4.0,
            "on_charger": False,
            "unknown_path_ratio": 0.4,
            "charger_target": (40, 40),
            "target_gap": 6.0,
            "suggested_action": 1,
            "return_action_reliable": True,
            "target_stable": True,
            "all_charger_known_path_count": 1,
        }
        prep._get_guidance = lambda: guidance
        prep._get_teacher_guidance = lambda: None

        _, components = prep.reward_process()

        self.assertLess(components["cleaning_context_scale"], 0.1)
        self.assertLess(components["skip_needed_charge_penalty"], 0.0)
        self.assertLess(components["planner_alignment"], 0.0)
        self.assertLess(components["sticky_anchor_penalty"], 0.0)

    def test_expert_teacher_guidance_keeps_return_action_only_signal(self):
        from agent_ppo.feature.expert import ExpertPolicy

        expert = ExpertPolicy()
        signal = {
            "target_reliable": False,
            "mode_reliable": False,
            "anchor_reliable": False,
            "return_action_reliable": True,
            "battery_ratio": 0.18,
            "slack": 4.0,
            "on_charger": False,
            "margin": 4.0,
            "unknown_path_ratio": 0.20,
            "all_charger_known_path_count": 1,
            "min_npc_dist": 10.0,
            "charger_target": (12, 12),
            "suggested_action": 3,
        }
        prep = type(
            "PrepStub",
            (),
            {
                "local_dirt_density": 0.0,
                "future_recoverability_score": 0.1,
                "route_contract_pressure": 0.0,
                "steps_since_charge": 50,
                "total_charger": 2,
            },
        )()

        guidance = expert.get_teacher_guidance(prep, signal=signal)

        self.assertIsNotNone(guidance)
        self.assertEqual(guidance["return_action"], 3)
        self.assertGreater(guidance["return_action_teacher_mask"], 0.0)

    def test_route_anchor_switches_when_better_target_is_compelling(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.battery = 80
        prep.battery_max = 200
        prep.charger_slack = 10.0
        prep.route_anchor_center = (20, 20)
        prep.route_anchor_idx = 2
        prep.last_route_anchor_idx = 2
        prep.sorted_charger_candidates = [
            {
                "center": (10, 10),
                "reachable": 1.0,
                "score": 8.0,
                "astar_dist": 8.0,
                "dist": 8.0,
                "unknown_path_ratio": 0.05,
            },
            {
                "center": (20, 20),
                "reachable": 1.0,
                "score": 18.0,
                "astar_dist": 18.0,
                "dist": 18.0,
                "unknown_path_ratio": 0.55,
            },
        ]
        prep._get_guidance = lambda: {
            "charger_target": (10, 10),
            "target_gap": 10.0,
            "slack": 10.0,
            "unknown_path_ratio": 0.05,
            "target_stable": True,
            "charger_dist": 8.0,
        }

        prep._update_route_anchor()
        self.assertEqual(prep.route_anchor_center, (10, 10))
        self.assertEqual(prep.route_anchor_idx, 1)

    def test_route_anchor_stays_when_advantage_is_small(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.battery = 120
        prep.battery_max = 200
        prep.charger_slack = 40.0
        prep.route_anchor_center = (20, 20)
        prep.route_anchor_idx = 2
        prep.last_route_anchor_idx = 2
        prep.sorted_charger_candidates = [
            {
                "center": (10, 10),
                "reachable": 1.0,
                "score": 15.0,
                "astar_dist": 15.0,
                "dist": 15.0,
                "unknown_path_ratio": 0.10,
            },
            {
                "center": (20, 20),
                "reachable": 1.0,
                "score": 15.4,
                "astar_dist": 15.4,
                "dist": 15.4,
                "unknown_path_ratio": 0.12,
            },
        ]
        prep._get_guidance = lambda: {
            "charger_target": (10, 10),
            "target_gap": 0.4,
            "slack": 40.0,
            "unknown_path_ratio": 0.10,
            "target_stable": False,
            "charger_dist": 15.0,
        }

        prep._update_route_anchor()
        self.assertEqual(prep.route_anchor_center, (20, 20))
        self.assertEqual(prep.route_anchor_idx, 2)


class LtsppoCurriculumAndCheckpointScoringTests(unittest.TestCase):
    def test_checkpoint_scoring_prefers_cps_and_behavior_health_over_raw_clean_score(self):
        from agent_ppo.workflow.checkpoint_score import compute_checkpoint_scores

        healthy_window = {
            "win_rate": 0.78,
            "battery_fail_rate": 0.08,
            "collision_fail_rate": 0.03,
            "late_return_rate": 0.04,
            "route_phase_return_stall_rate": 0.22,
            "avg_clean_per_step": 0.88,
            "cps_win": 0.97,
            "late_contract_rate": 0.05,
            "recoverability_violation_rate": 0.09,
            "wall_hugging_clean_floor_rate": 0.03,
            "stale_boundary_follow_rate": 0.02,
            "narrow_unknown_commit_rate": 0.05,
            "missed_charge_opportunity_rate": 0.01,
            "suboptimal_target_hold_rate": 0.03,
            "reliable_planner_divergence_rate": 0.18,
            "zero_charge_battery_fail_rate": 0.10,
            "avg_clean_score": 760.0,
        }
        unhealthy_window = {
            **healthy_window,
            "avg_clean_score": 1180.0,
            "avg_clean_per_step": 0.54,
            "cps_win": 0.58,
            "battery_fail_rate": 0.18,
            "route_phase_return_stall_rate": 0.49,
            "wall_hugging_clean_floor_rate": 0.14,
            "suboptimal_target_hold_rate": 0.16,
            "reliable_planner_divergence_rate": 0.39,
            "zero_charge_battery_fail_rate": 0.55,
        }
        learning = {
            "entropy_loss": 0.78,
            "value_clean_loss_trend_ratio": 0.96,
            "value_survive_loss_trend_ratio": 0.97,
            "mode_teacher_active_rate": 0.50,
            "route_anchor_teacher_active_rate": 0.82,
            "target_teacher_active_rate": 0.83,
            "return_action_teacher_active_rate": 0.06,
            "env_total_score": 880.0,
            "env_total_score_trend_ratio": 1.02,
        }

        healthy = compute_checkpoint_scores(healthy_window, learning)
        unhealthy = compute_checkpoint_scores(unhealthy_window, learning)

        self.assertGreater(healthy["resume_readiness_score"], unhealthy["resume_readiness_score"])
        self.assertGreater(healthy["checkpoint_preservation_score"], unhealthy["checkpoint_preservation_score"])

    def test_checkpoint_scoring_benchmark_falls_back_to_legacy_stall_and_planner_metrics(self):
        from agent_ppo.workflow.checkpoint_score import compute_checkpoint_scores

        window = {
            "_count": 20,
            "win_rate": 0.70,
            "battery_fail_rate": 0.10,
            "collision_fail_rate": 0.02,
            "zero_charge_battery_fail_rate": 0.10,
            "route_phase_return_stall_rate": 0.20,
            "reliable_planner_divergence_rate": 0.20,
        }
        learning = {"entropy_loss": 0.80}
        benchmark = {
            "completed_rate": 0.70,
            "battery_fail_rate": 0.08,
            "collision_fail_rate": 0.02,
            "broad_win_rate": 0.70,
            "avg_clean_per_step": 0.70,
            "cps_win": 0.75,
            "avg_clean_score_win": 800.0,
            "avg_remaining_charge": 80.0,
            "late_return_rate": 0.05,
            "return_stall_rate": 0.22,
            "recoverability_violation_rate": 0.10,
            "wall_hugging_clean_floor_rate": 0.03,
            "stale_boundary_follow_rate": 0.02,
            "narrow_unknown_commit_rate": 0.05,
            "missed_charge_opportunity_rate": 0.01,
            "suboptimal_target_hold_rate": 0.03,
            "planner_policy_divergence_rate": 0.18,
        }

        scores = compute_checkpoint_scores(window, learning, benchmark)

        self.assertGreater(scores["submission_score_stability"], 0.0)
        self.assertGreater(scores["submission_score_behavior"], 0.0)

    def test_curriculum_fast_skip_and_regression_rules(self):
        from agent_ppo.workflow.curriculum_policy import choose_stage, should_regress_stage

        bootstrap_metrics = {
            "_count": 10,
            "win_rate": 0.78,
            "battery_fail_rate": 0.05,
            "return_stall_rate": 0.22,
            "wall_hugging_clean_floor_rate": 0.03,
            "suboptimal_target_hold_rate": 0.03,
            "broad_win_rate": 0.67,
            "planner_policy_divergence_rate": 0.18,
        }
        fast_context = {
            "global_step_since_resume": 3200,
            "bootstrap_metrics": bootstrap_metrics,
            "window_metrics": None,
            "learning_metrics": {"entropy_loss": 0.84, "entropy_trend_ratio": 1.0},
            "resume_fast_track": True,
        }
        stage = choose_stage("warmup", fast_context, None)
        self.assertEqual(stage, "robust")

        entry_metrics = {
            "battery_fail_rate": 0.06,
            "return_stall_rate": 0.22,
            "env_total_score": 910.0,
        }
        current_metrics = {
            "battery_fail_rate": 0.14,
            "return_stall_rate": 0.34,
        }
        learning_metrics = {
            "entropy_loss": 1.01,
            "env_total_score": 780.0,
        }
        self.assertTrue(should_regress_stage("robust", entry_metrics, current_metrics, learning_metrics))


if __name__ == "__main__":
    unittest.main()
