#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
    workflow_mod = types.ModuleType("common_python.utils.workflow_disaster_recovery")
    tools_mod = types.ModuleType("tools")
    metrics_utils_mod = types.ModuleType("tools.metrics_utils")
    train_env_conf_validate_mod = types.ModuleType("tools.train_env_conf_validate")

    def create_cls(name, **defaults):
        attrs = dict(defaults)

        def __init__(self, **kwargs):
            for key, default in defaults.items():
                setattr(self, key, kwargs.get(key, default))

        attrs["__init__"] = __init__
        return type(name, (), attrs)

    common_func_mod.create_cls = create_cls

    def handle_disaster_recovery(*args, **kwargs):
        return False

    workflow_mod.handle_disaster_recovery = handle_disaster_recovery
    metrics_utils_mod.get_training_metrics = lambda: {}
    train_env_conf_validate_mod.read_usr_conf = lambda *args, **kwargs: {}
    utils_mod.common_func = common_func_mod
    utils_mod.workflow_disaster_recovery = workflow_mod
    common_python_mod.utils = utils_mod

    sys.modules["common_python"] = common_python_mod
    sys.modules["common_python.utils"] = utils_mod
    sys.modules["common_python.utils.common_func"] = common_func_mod
    sys.modules["common_python.utils.workflow_disaster_recovery"] = workflow_mod
    sys.modules["tools"] = tools_mod
    sys.modules["tools.metrics_utils"] = metrics_utils_mod
    sys.modules["tools.train_env_conf_validate"] = train_env_conf_validate_mod


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


class LtsppoSlice2SignalHelpersTests(unittest.TestCase):
    def test_route_phase_shadow_risk_activates_when_no_reachable_route_even_without_unknown_ratio(self):
        from agent_ppo.utils.constraint_utils import compute_route_phase_shadow_risk

        risk = compute_route_phase_shadow_risk(
            min_recoverability=0.8,
            charger_slack=12.0,
            charge_margin_now=20.0,
            planner_topk_reachable_count=0,
            unknown_target_ratio=0.0,
            route_contract_pressure=0.0,
            recoverability_warn=0.35,
            recoverability_span=0.70,
            prepare_return_slack_threshold=6.0,
            charge_margin_warn=17.0,
            unknown_ratio_threshold=0.20,
        )

        self.assertGreater(risk, 0.5)

    def test_route_phase_shadow_risk_unknown_ratio_threshold_is_effective(self):
        from agent_ppo.utils.constraint_utils import compute_route_phase_shadow_risk

        below = compute_route_phase_shadow_risk(
            min_recoverability=0.8,
            charger_slack=20.0,
            charge_margin_now=30.0,
            planner_topk_reachable_count=1,
            unknown_target_ratio=0.15,
            route_contract_pressure=0.0,
            recoverability_warn=0.35,
            recoverability_span=0.70,
            prepare_return_slack_threshold=6.0,
            charge_margin_warn=17.0,
            unknown_ratio_threshold=0.20,
        )
        above = compute_route_phase_shadow_risk(
            min_recoverability=0.8,
            charger_slack=20.0,
            charge_margin_now=30.0,
            planner_topk_reachable_count=1,
            unknown_target_ratio=0.55,
            route_contract_pressure=0.0,
            recoverability_warn=0.35,
            recoverability_span=0.70,
            prepare_return_slack_threshold=6.0,
            charge_margin_warn=17.0,
            unknown_ratio_threshold=0.20,
        )

        self.assertAlmostEqual(below, 0.0, places=6)
        self.assertGreater(above, 0.0)

    def test_route_phase_reward_ready_uses_route_context_and_shadow_risk_not_battery_state(self):
        from agent_ppo.utils.constraint_utils import compute_route_phase_reward_ready

        ready = compute_route_phase_reward_ready(
            current_mode=3,
            mode_contract=3,
            mode_return=4,
            route_phase_reliable_active=False,
            return_action_reliable=False,
            anchor_reliable=True,
            target_reliable=False,
            known_route_available=False,
            route_phase_shadow_risk=0.2,
            route_phase_shadow_risk_threshold=0.12,
        )
        not_ready = compute_route_phase_reward_ready(
            current_mode=1,
            mode_contract=3,
            mode_return=4,
            route_phase_reliable_active=True,
            return_action_reliable=True,
            anchor_reliable=True,
            target_reliable=True,
            known_route_available=True,
            route_phase_shadow_risk=0.5,
            route_phase_shadow_risk_threshold=0.12,
        )

        self.assertTrue(ready)
        self.assertFalse(not_ready)

    def test_route_phase_reward_ready_accepts_target_or_known_route_evidence(self):
        from agent_ppo.utils.constraint_utils import compute_route_phase_reward_ready

        ready_from_target = compute_route_phase_reward_ready(
            current_mode=4,
            mode_contract=3,
            mode_return=4,
            route_phase_reliable_active=False,
            return_action_reliable=False,
            anchor_reliable=False,
            target_reliable=True,
            known_route_available=False,
            route_phase_shadow_risk=0.16,
            route_phase_shadow_risk_threshold=0.12,
        )
        ready_from_known_route = compute_route_phase_reward_ready(
            current_mode=3,
            mode_contract=3,
            mode_return=4,
            route_phase_reliable_active=False,
            return_action_reliable=False,
            anchor_reliable=False,
            target_reliable=False,
            known_route_available=True,
            route_phase_shadow_risk=0.16,
            route_phase_shadow_risk_threshold=0.12,
        )

        self.assertTrue(ready_from_target)
        self.assertTrue(ready_from_known_route)


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

    def test_infer_mode_control_simplify_v1_uses_primary_hits_with_route_pressure_secondary(self):
        from agent_ppo.conf.conf import Config
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor.__new__(Preprocessor)
        prep.battery = 70.0
        prep.battery_max = 200.0
        prep.nearest_npc_dist = 10.0
        prep.charger_slack = 9.0
        prep.future_recoverability_score = 0.50
        prep.route_contract_pressure = 0.60
        prep.total_charger = 4
        prep.steps_since_charge = 100
        prep.local_dirt_density = 0.0
        prep.dirty_adjacent = 0
        prep._get_guidance = lambda: {
            "margin": 20.0,
            "all_charger_known_path_count": 4,
            "unknown_path_ratio": 0.0,
            "planner_topk_reachable_count": 4,
            "planner_multi_route_recoverability": 0.50,
        }

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_control_simplify_v1"}, clear=False), \
            patch.object(Config, "PREPARE_RETURN_SLACK_THRESHOLD", 10.0), \
            patch.object(Config, "CONTRACT_BATTERY_RATIO", 0.27), \
            patch.object(Config, "CONTRACT_RECOVERABILITY_THRESHOLD", 0.12), \
            patch.object(Config, "CHARGE_MARGIN_WARN", 17.0), \
            patch.object(Config, "CONTRACT_ROUTE_PRESSURE_THRESHOLD", 0.52):
            inferred = Preprocessor._infer_mode(prep)
            self.assertEqual(inferred, Preprocessor.MODE_CONTRACT)

            prep.route_contract_pressure = 0.10
            inferred_without_secondary = Preprocessor._infer_mode(prep)
            self.assertNotEqual(inferred_without_secondary, Preprocessor.MODE_CONTRACT)

            prep.battery = 40.0
            prep.future_recoverability_score = 0.10
            inferred_two_primary = Preprocessor._infer_mode(prep)
            self.assertEqual(inferred_two_primary, Preprocessor.MODE_CONTRACT)

    def test_expert_teacher_guidance_control_simplify_v1_matches_pre_return_readiness(self):
        from agent_ppo.conf.conf import Config
        from agent_ppo.feature.expert import ExpertPolicy

        expert = ExpertPolicy()
        signal = {
            "target_reliable": True,
            "mode_reliable": True,
            "anchor_reliable": False,
            "return_action_reliable": True,
            "battery_ratio": 0.35,
            "slack": 9.0,
            "on_charger": False,
            "margin": 20.0,
            "unknown_path_ratio": 0.0,
            "all_charger_known_path_count": 4,
            "planner_topk_reachable_count": 4,
            "planner_multi_route_recoverability": 0.50,
            "min_npc_dist": 10.0,
            "charger_target": (12, 12),
            "suggested_action": 3,
        }
        prep = type(
            "PrepStub",
            (),
            {
                "local_dirt_density": 0.0,
                "future_recoverability_score": 0.50,
                "route_contract_pressure": 0.60,
                "steps_since_charge": 50,
                "total_charger": 4,
            },
        )()

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_control_simplify_v1"}, clear=False), \
            patch.object(Config, "PREPARE_RETURN_SLACK_THRESHOLD", 10.0), \
            patch.object(Config, "CONTRACT_BATTERY_RATIO", 0.27), \
            patch.object(Config, "CONTRACT_RECOVERABILITY_THRESHOLD", 0.12), \
            patch.object(Config, "CHARGE_MARGIN_WARN", 17.0), \
            patch.object(Config, "CONTRACT_ROUTE_PRESSURE_THRESHOLD", 0.52):
            guidance = expert.get_teacher_guidance(prep, signal=signal)
            self.assertIsNotNone(guidance)
            self.assertEqual(guidance["mode"], "contract")

            signal["planner_topk_reachable_count"] = 0
            signal["all_charger_known_path_count"] = 0
            signal["unknown_path_ratio"] = 0.80
            signal["battery_ratio"] = 0.22
            signal["slack"] = 3.0
            signal["margin"] = 4.0
            guidance_return = expert.get_teacher_guidance(prep, signal=signal)
            self.assertIsNotNone(guidance_return)
            self.assertEqual(guidance_return["mode"], "return")

    def test_reward_process_control_simplify_v1_keeps_contract_mode_productive(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_CONTRACT
        prep.cleaned_this_step = 1
        prep.consecutive_clean_steps = 3
        prep.cur_visit_count = 1
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.4
        prep.same_region_streak = 1
        prep.path_cross_count_50 = 4
        prep.coverage_efficiency_20 = 0.8
        prep.no_progress_steps = 0
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 4.0
        prep.battery = 70
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 20
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.route_anchor_center = (16, 16)
        prep.future_recoverability_score = 0.18
        prep._prev_future_recoverability_score = 0.18
        prep.current_target_dist = 8.0
        prep._last_target_distance = 8.0
        prep.return_stall_ema = 0.2
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.5
        prep._last_action = 0
        prep.new_explored_cells = 2
        prep.explored_ratio = 0.4
        prep.dirt_cleaned = 10
        prep.total_dirt = 100
        prep.last_charger_slack = 9.0
        prep.charger_slack = 9.0
        prep.charge_count = 0
        prep.pre_charge_battery = 70
        prep._prev_charge_need_score = 0.0
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.05
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": 9.0,
            "margin": 18.0,
            "on_charger": False,
            "unknown_path_ratio": 0.05,
            "charger_target": (20, 20),
            "target_gap": 2.0,
            "suggested_action": 1,
            "return_action_reliable": True,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "target_stable": True,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "planner_best_target_route_diversity": 1.0,
            "planner_multi_route_recoverability": 0.18,
        }
        prep._get_teacher_guidance = lambda: {
            "route_mode": "contract",
            "route_anchor": (20, 20),
            "target": (20, 20),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 1.0,
            "target_teacher_mask": 1.0,
            "return_action": 1,
            "return_action_teacher_mask": 1.0,
        }

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_control_simplify_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertGreater(components["cleaning_context_scale"], 0.9)
        self.assertGreater(components["explore"], 0.0)
        self.assertGreater(components["frontier"], 0.0)
        self.assertAlmostEqual(components["charge_detour_cost"], 0.0, places=6)
        self.assertAlmostEqual(components["charge_interrupt_cost"], 0.0, places=6)
        self.assertAlmostEqual(components["charger_access_discovery_bonus"], 0.0, places=6)
        self.assertAlmostEqual(components["charger_access_probe_bonus"], 0.0, places=6)

    def test_reward_process_cps_align_v1_replaces_cps_bonus_with_effective_coverage_bonus(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_HARVEST
        prep.cleaned_this_step = 2
        prep.new_explored_cells = 2
        prep.consecutive_clean_steps = 3
        prep.cur_visit_count = 1
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 1
        prep.local_frontier_density = 0.25
        prep.same_region_streak = 1
        prep.path_cross_count_50 = 2
        prep.coverage_efficiency_20 = 0.92
        prep.no_progress_steps = 0
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 6.0
        prep.battery = 120
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 20
        prep.route_anchor_idx = 0
        prep.last_route_anchor_idx = 0
        prep.route_anchor_center = None
        prep.future_recoverability_score = 0.55
        prep._prev_future_recoverability_score = 0.55
        prep.current_target_dist = 8.0
        prep._last_target_distance = 8.0
        prep.return_stall_ema = 0.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.9
        prep._last_action = 0
        prep.explored_ratio = 0.5
        prep.dirt_cleaned = 10
        prep.total_dirt = 100
        prep.last_charger_slack = 18.0
        prep.charger_slack = 18.0
        prep.charge_count = 0
        prep.pre_charge_battery = 120
        prep._prev_charge_need_score = 0.0
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.05
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep.cur_pos = (20, 20)
        prep.explored_map[20, 20] = 1.0
        prep.passable_map[20, 20] = 1.0
        prep.dirty_memory[20, 20] = 1.0
        prep.charger_map[20, 20] = 0.0
        prep._get_guidance = lambda: {
            "slack": 18.0,
            "margin": 20.0,
            "on_charger": False,
            "unknown_path_ratio": 0.05,
            "charger_target": (20, 20),
            "target_gap": 0.0,
            "suggested_action": 1,
            "return_action_reliable": True,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "target_stable": True,
            "all_charger_known_path_count": 2.0,
            "planner_topk_reachable_count": 2.0,
            "planner_best_target_route_diversity": 1.0,
            "planner_multi_route_recoverability": 0.55,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_cps_align_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertAlmostEqual(components["cps_bonus"], 0.0, places=6)
        self.assertAlmostEqual(components["coverage_efficiency_bonus"], 0.0, places=6)
        self.assertGreater(components["effective_coverage_bonus"], 0.0)
        self.assertAlmostEqual(components["clean_floor_revisit_penalty"], 0.0, places=6)

    def test_reward_process_cps_align_v1_only_penalizes_safe_low_value_clean_floor_revisits(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_HARVEST
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.consecutive_clean_steps = 0
        prep.cur_visit_count = 3
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.01
        prep.same_region_streak = 4
        prep.recent_unique_cells_20 = 8
        prep.path_cross_count_50 = 9
        prep.coverage_efficiency_20 = 0.55
        prep.no_progress_steps = 8
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 8.0
        prep.battery = 160
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 10
        prep.route_anchor_idx = 0
        prep.last_route_anchor_idx = 0
        prep.future_recoverability_score = 0.65
        prep._prev_future_recoverability_score = 0.65
        prep.current_target_dist = 8.0
        prep._last_target_distance = 8.0
        prep.return_stall_ema = 0.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.4
        prep._last_action = 0
        prep.explored_ratio = 0.7
        prep.dirt_cleaned = 20
        prep.total_dirt = 100
        prep.last_charger_slack = 20.0
        prep.charger_slack = 20.0
        prep.charge_count = 0
        prep.pre_charge_battery = 160
        prep._prev_charge_need_score = 0.0
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 2.0
        prep._prev_unknown_on_target_path_ratio = 0.02
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep.cur_pos = (30, 30)
        prep.explored_map[30, 30] = 1.0
        prep.passable_map[30, 30] = 1.0
        prep.dirty_memory[30, 30] = 0.0
        prep.charger_map[30, 30] = 0.0
        prep._get_guidance = lambda: {
            "slack": 20.0,
            "margin": 24.0,
            "on_charger": False,
            "unknown_path_ratio": 0.02,
            "charger_target": (20, 20),
            "target_gap": 0.0,
            "suggested_action": 1,
            "return_action_reliable": True,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "target_stable": True,
            "all_charger_known_path_count": 2.0,
            "planner_topk_reachable_count": 2.0,
            "planner_best_target_route_diversity": 1.0,
            "planner_multi_route_recoverability": 0.65,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_cps_align_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertLess(components["clean_floor_revisit_penalty"], 0.0)
        self.assertAlmostEqual(components["current_cell_is_clean_floor"], 1.0, places=6)
        self.assertAlmostEqual(components["low_value_revisit_flag"], 1.0, places=6)

        prep.current_mode = prep.MODE_RETURN
        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_cps_align_v1"}, clear=False):
            _, components_return = prep.reward_process()
        self.assertAlmostEqual(components_return["clean_floor_revisit_penalty"], 0.0, places=6)

        prep.current_mode = prep.MODE_HARVEST
        prep.dirty_memory[30, 30] = 1.0
        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_cps_align_v1"}, clear=False):
            _, components_dirty = prep.reward_process()
        self.assertAlmostEqual(components_dirty["clean_floor_revisit_penalty"], 0.0, places=6)

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
    def test_episode_sequence_diagnostics_tracks_readiness_transition_quality(self):
        from agent_ppo.workflow.train_workflow import EpisodeRunner

        step_records = [
            {"mode": 2, "control_stack_simplify_active": 1.0, "pre_return_readiness_flag": 0.0, "route_anchor": 0, "target": 0, "charger_slack": 12.0, "future_recoverability_score": 0.9, "anchor_return_dist": 12.0, "is_diag_action": 0.0, "planner_policy_divergence": 0.0, "route_phase_active": 0.0, "route_phase_reliable_active": 0.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.9, "constraint_battery_process_cost": 0.0, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.0, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0},
            {"mode": 3, "control_stack_simplify_active": 1.0, "pre_return_readiness_flag": 1.0, "route_anchor": 1, "target": 1, "charger_slack": 8.0, "future_recoverability_score": 0.2, "anchor_return_dist": 10.0, "is_diag_action": 1.0, "planner_policy_divergence": 0.0, "route_phase_active": 1.0, "route_phase_reliable_active": 1.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.2, "constraint_battery_process_cost": 0.1, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.5, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0},
            {"mode": 4, "control_stack_simplify_active": 1.0, "pre_return_readiness_flag": 0.0, "route_anchor": 1, "target": 1, "charger_slack": 4.0, "future_recoverability_score": 0.1, "anchor_return_dist": 7.0, "is_diag_action": 1.0, "planner_policy_divergence": 0.0, "route_phase_active": 1.0, "route_phase_reliable_active": 1.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.1, "constraint_battery_process_cost": 0.1, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.7, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0},
            {"mode": 2, "control_stack_simplify_active": 1.0, "pre_return_readiness_flag": 0.0, "route_anchor": 0, "target": 0, "charger_slack": 12.0, "future_recoverability_score": 0.9, "anchor_return_dist": 12.0, "is_diag_action": 0.0, "planner_policy_divergence": 0.0, "route_phase_active": 0.0, "route_phase_reliable_active": 0.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.9, "constraint_battery_process_cost": 0.0, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.0, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0},
            {"mode": 4, "control_stack_simplify_active": 1.0, "pre_return_readiness_flag": 0.0, "route_anchor": 1, "target": 1, "charger_slack": 2.0, "future_recoverability_score": 0.05, "anchor_return_dist": 9.0, "is_diag_action": 1.0, "planner_policy_divergence": 0.0, "route_phase_active": 1.0, "route_phase_reliable_active": 1.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.05, "constraint_battery_process_cost": 0.1, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.8, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0},
        ]

        diagnostics = EpisodeRunner._episode_sequence_diagnostics(step_records)

        self.assertAlmostEqual(diagnostics["return_entry_count"], 2.0, places=6)
        self.assertAlmostEqual(diagnostics["readiness_supported_return_entry_count"], 1.0, places=6)
        self.assertAlmostEqual(diagnostics["pre_return_readiness_hit_rate"], 0.2, places=6)
        self.assertAlmostEqual(diagnostics["readiness_to_return_transition_rate"], 0.5, places=6)
        self.assertAlmostEqual(diagnostics["direct_return_without_readiness_rate"], 0.5, places=6)

    def test_episode_sequence_diagnostics_marks_readiness_rates_not_applicable_without_return_entries(self):
        from agent_ppo.workflow.train_workflow import EpisodeRunner

        step_records = [
            {"mode": 2, "control_stack_simplify_active": 1.0, "pre_return_readiness_flag": 0.0, "route_anchor": 0, "target": 0, "charger_slack": 12.0, "future_recoverability_score": 0.9, "anchor_return_dist": 12.0, "is_diag_action": 0.0, "planner_policy_divergence": 0.0, "route_phase_active": 0.0, "route_phase_reliable_active": 0.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.9, "constraint_battery_process_cost": 0.0, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.0, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0},
            {"mode": 3, "control_stack_simplify_active": 1.0, "pre_return_readiness_flag": 1.0, "route_anchor": 1, "target": 1, "charger_slack": 8.0, "future_recoverability_score": 0.2, "anchor_return_dist": 10.0, "is_diag_action": 1.0, "planner_policy_divergence": 0.0, "route_phase_active": 1.0, "route_phase_reliable_active": 1.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.2, "constraint_battery_process_cost": 0.1, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.5, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0},
        ]

        diagnostics = EpisodeRunner._episode_sequence_diagnostics(step_records)

        self.assertAlmostEqual(diagnostics["return_entry_count"], 0.0, places=6)
        self.assertAlmostEqual(diagnostics["readiness_supported_return_entry_count"], 0.0, places=6)
        self.assertAlmostEqual(diagnostics["pre_return_readiness_hit_rate"], 0.5, places=6)
        self.assertIsNone(diagnostics["readiness_to_return_transition_rate"])
        self.assertIsNone(diagnostics["direct_return_without_readiness_rate"])

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


class StrongHeuristicStructureTests(unittest.TestCase):
    def test_strong_heuristic_phase_helper_and_mode_mapping(self):
        from agent_ppo.utils.strong_heuristic import (
            LOGICAL_MODE_CLEAN,
            LOGICAL_MODE_EVADE,
            LOGICAL_MODE_PRE_RETURN,
            LOGICAL_MODE_RETURN,
            logical_mode_to_training_mode,
            strong_heuristic_active,
            strong_heuristic_slice2a_active,
        )
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()

        self.assertTrue(strong_heuristic_active("s1_survival_strong_heuristic_v1"))
        self.assertTrue(strong_heuristic_active("s1_survival_strong_heuristic_slice2a_v1"))
        self.assertTrue(strong_heuristic_active("s1_survival_strong_heuristic_slice2a_fixed8_v1"))
        self.assertTrue(strong_heuristic_slice2a_active("s1_survival_strong_heuristic_slice2a_v1"))
        self.assertTrue(strong_heuristic_slice2a_active("s1_survival_strong_heuristic_slice2a_fixed8_v1"))
        self.assertFalse(strong_heuristic_slice2a_active("s1_survival_strong_heuristic_v1"))
        self.assertFalse(strong_heuristic_active("s1_survival_cps_align_v1"))
        self.assertEqual(logical_mode_to_training_mode(LOGICAL_MODE_CLEAN, prep), prep.MODE_EXPAND)
        self.assertEqual(logical_mode_to_training_mode(LOGICAL_MODE_PRE_RETURN, prep), prep.MODE_CONTRACT)
        self.assertEqual(logical_mode_to_training_mode(LOGICAL_MODE_RETURN, prep), prep.MODE_RETURN)
        self.assertEqual(logical_mode_to_training_mode(LOGICAL_MODE_EVADE, prep), prep.MODE_EVADE)

    def test_reward_process_in_slice2a_replaces_charging_reward_mainline_with_risk_terms(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        def build_prep():
            prep = Preprocessor()
            prep.current_mode = prep.MODE_RETURN
            prep.cleaned_this_step = 0
            prep.new_explored_cells = 0
            prep.consecutive_clean_steps = 0
            prep.cur_visit_count = 1
            prep.wall_adjacent = 0
            prep.dirty_adjacent = 0
            prep.local_frontier_density = 0.0
            prep.local_dirt_density = 0.0
            prep.same_region_streak = 1
            prep.recent_unique_cells_20 = 20
            prep.path_cross_count_50 = 4
            prep.coverage_efficiency_20 = 0.8
            prep.no_progress_steps = 0
            prep.actual_legal_ratio = 1.0
            prep.just_charged = 1.0
            prep.nearest_charger_dist = 4.0
            prep.last_nearest_charger_dist = 8.0
            prep.battery = 120
            prep.battery_max = 200
            prep.nearest_npc_dist = 20.0
            prep.last_move_invalid = 0.0
            prep.stuck_steps = 0
            prep.invalid_move_ema = 0.0
            prep.steps_since_charge = 20
            prep.route_anchor_idx = 1
            prep.last_route_anchor_idx = 1
            prep.route_anchor_center = (16, 16)
            prep.future_recoverability_score = 0.70
            prep._prev_future_recoverability_score = 0.70
            prep.current_target_dist = 8.0
            prep._last_target_distance = 10.0
            prep.return_stall_ema = 0.2
            prep._last_astar_dist = 8.0
            prep._astar_dist = 8.0
            prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
            prep._cps_ema = 0.5
            prep._last_action = 1
            prep.explored_ratio = 0.4
            prep.dirt_cleaned = 10
            prep.total_dirt = 100
            prep.last_charger_slack = 12.0
            prep.charger_slack = 18.0
            prep.charge_count = 1
            prep.last_charge_count = 0
            prep.pre_charge_battery = 90
            prep.step_no = 2
            prep._prev_charge_need_score = 0.60
            prep._prev_charge_detour_proxy = 0.45
            prep._prev_charge_interrupt_proxy = 0.30
            prep._prev_all_charger_known_path_count = 1.0
            prep._prev_unknown_on_target_path_ratio = 0.05
            prep._prev_planner_best_target_route_diversity = 1.0
            prep.training_global_step = 0
            prep._get_guidance = lambda: {
                "slack": 18.0,
                "margin": 18.0,
                "on_charger": False,
                "unknown_path_ratio": 0.05,
                "charger_target": (20, 20),
                "target_gap": 2.0,
                "suggested_action": 1,
                "return_action_reliable": True,
                "target_reliable": True,
                "anchor_reliable": True,
                "mode_reliable": True,
                "target_stable": True,
                "all_charger_known_path_count": 1.0,
                "planner_topk_reachable_count": 1.0,
                "planner_best_target_route_diversity": 1.0,
                "planner_multi_route_recoverability": 0.70,
            }
            prep._get_teacher_guidance = lambda: None
            return prep

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_v1"}, clear=False):
            _, legacy_components = build_prep().reward_process()
        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, slice2a_components = build_prep().reward_process()

        legacy_shadow_sum = (
            legacy_components["charge_route_progress_bonus"]
            + legacy_components["return_progress_shaping_bonus"]
            + legacy_components["necessary_charge_bonus"]
            + legacy_components["unnecessary_charge_penalty"]
            + legacy_components["charge_detour_cost"]
            + legacy_components["charge_interrupt_cost"]
            + legacy_components["skip_needed_charge_penalty"]
            + legacy_components["high_need_return_stall_penalty"]
            + legacy_components["charger_access_discovery_bonus"]
            + legacy_components["charger_access_probe_bonus"]
        )
        slice2a_reward_sum = (
            slice2a_components["risk_release_reward"]
            + slice2a_components["route_phase_risk_growth_penalty"]
            + slice2a_components["charge_opportunity_cost_penalty"]
        )
        reward_delta = slice2a_components["reward_total"] - legacy_components["reward_total"]

        self.assertNotAlmostEqual(legacy_shadow_sum, 0.0, places=6)
        self.assertGreater(slice2a_components["risk_release_reward"], 0.0)
        self.assertAlmostEqual(slice2a_components["risk_growth_while_clean_penalty"], 0.0, places=6)
        self.assertAlmostEqual(slice2a_components["route_phase_risk_growth_penalty"], 0.0, places=6)
        self.assertLessEqual(slice2a_components["charge_opportunity_cost_penalty"], 0.0)
        self.assertAlmostEqual(
            reward_delta,
            slice2a_reward_sum - legacy_shadow_sum,
            places=6,
        )
        self.assertAlmostEqual(slice2a_components["charge_reward_shadow_only_active"], 1.0, places=6)

    def test_reward_process_in_slice2a_keeps_clean_risk_growth_as_shadow_only(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_EXPAND
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.consecutive_clean_steps = 0
        prep.cur_visit_count = 1
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.0
        prep.local_dirt_density = 0.0
        prep.same_region_streak = 2
        prep.recent_unique_cells_20 = 8
        prep.path_cross_count_50 = 4
        prep.coverage_efficiency_20 = 0.7
        prep.no_progress_steps = 4
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 10.0
        prep.last_nearest_charger_dist = 8.0
        prep.battery = 30
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 30
        prep.route_anchor_idx = 0
        prep.last_route_anchor_idx = 0
        prep.route_anchor_center = None
        prep.future_recoverability_score = 0.05
        prep._prev_future_recoverability_score = 0.05
        prep.current_target_dist = 8.0
        prep._last_target_distance = 8.0
        prep.return_stall_ema = 0.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.4
        prep._last_action = 0
        prep.explored_ratio = 0.5
        prep.dirt_cleaned = 10
        prep.total_dirt = 100
        prep.last_charger_slack = 12.0
        prep.charger_slack = -2.0
        prep.charge_count = 0
        prep.pre_charge_battery = 70
        prep.step_no = 2
        prep._prev_charge_need_score = 0.05
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.05
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": -2.0,
            "margin": 12.0,
            "on_charger": False,
            "unknown_path_ratio": 0.05,
            "charger_target": (20, 20),
            "target_gap": 2.0,
            "suggested_action": 1,
            "return_action_reliable": False,
            "target_reliable": False,
            "anchor_reliable": False,
            "mode_reliable": False,
            "target_stable": False,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 0.0,
            "planner_best_target_route_diversity": 1.0,
            "planner_multi_route_recoverability": 0.05,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertLess(components["risk_growth_while_clean_penalty"], 0.0)
        self.assertAlmostEqual(components["route_phase_risk_growth_penalty"], 0.0, places=6)
        self.assertAlmostEqual(components["risk_release_reward"], 0.0, places=6)

    def test_reward_process_in_slice2a_penalizes_route_phase_risk_growth(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.consecutive_clean_steps = 0
        prep.cur_visit_count = 1
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.0
        prep.local_dirt_density = 0.0
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep.path_cross_count_50 = 2
        prep.coverage_efficiency_20 = 0.9
        prep.no_progress_steps = 2
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 10.0
        prep.last_nearest_charger_dist = 8.0
        prep.battery = 40
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 30
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.route_anchor_center = (16, 16)
        prep.future_recoverability_score = 0.05
        prep._prev_future_recoverability_score = 0.05
        prep.current_target_dist = 8.0
        prep._last_target_distance = 8.0
        prep.return_stall_ema = 0.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.4
        prep._last_action = 0
        prep.explored_ratio = 0.5
        prep.dirt_cleaned = 10
        prep.total_dirt = 100
        prep.last_charger_slack = 12.0
        prep.charger_slack = -2.0
        prep.charge_count = 0
        prep.pre_charge_battery = 70
        prep.step_no = 2
        prep._prev_charge_need_score = 0.05
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.05
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": -2.0,
            "margin": 12.0,
            "on_charger": False,
            "unknown_path_ratio": 0.05,
            "charger_target": (20, 20),
            "target_gap": 2.0,
            "suggested_action": 1,
            "return_action_reliable": True,
            "target_reliable": False,
            "anchor_reliable": False,
            "mode_reliable": False,
            "target_stable": False,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "planner_best_target_route_diversity": 1.0,
            "planner_multi_route_recoverability": 0.05,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertLess(components["route_phase_risk_growth_penalty"], 0.0)
        self.assertAlmostEqual(components["risk_growth_while_clean_penalty"], 0.0, places=6)

    def test_reward_process_in_slice2a_releases_risk_from_route_shadow_delta(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.consecutive_clean_steps = 0
        prep.cur_visit_count = 1
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.0
        prep.local_dirt_density = 0.0
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep.path_cross_count_50 = 2
        prep.coverage_efficiency_20 = 0.9
        prep.no_progress_steps = 0
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 2.0
        prep.last_nearest_charger_dist = 6.0
        prep.battery = 90
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 30
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.route_anchor_center = (16, 16)
        prep.future_recoverability_score = 0.30
        prep._prev_future_recoverability_score = 0.30
        prep.current_target_dist = 2.0
        prep._last_target_distance = 4.0
        prep.return_stall_ema = 0.0
        prep._last_astar_dist = 4.0
        prep._astar_dist = 2.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.4
        prep._last_action = 1
        prep.explored_ratio = 0.5
        prep.dirt_cleaned = 10
        prep.total_dirt = 100
        prep.last_charger_slack = 4.0
        prep.charger_slack = 8.0
        prep.charge_count = 0
        prep.pre_charge_battery = 90
        prep.step_no = 2
        prep._prev_charge_need_score = 0.08
        prep._prev_route_phase_shadow_risk = 0.8
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.0
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": 8.0,
            "margin": 18.0,
            "on_charger": False,
            "unknown_path_ratio": 0.0,
            "charger_target": (16, 16),
            "target_gap": 4.0,
            "suggested_action": 1,
            "return_action_reliable": True,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "target_stable": True,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "planner_best_target_route_diversity": 1.0,
            "planner_multi_route_recoverability": 0.30,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertGreater(components["risk_release_reward"], 0.0)
        self.assertGreater(components["risk_release_from_progress"], 0.0)

    def test_reward_process_in_slice2a_penalizes_early_charge_opportunity_cost(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.consecutive_clean_steps = 0
        prep.cur_visit_count = 1
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.0
        prep.local_dirt_density = 0.0
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep.path_cross_count_50 = 2
        prep.coverage_efficiency_20 = 0.9
        prep.no_progress_steps = 0
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 1.0
        prep.nearest_charger_dist = 1.0
        prep.last_nearest_charger_dist = 2.0
        prep.battery = 120
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 1
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.route_anchor_center = (16, 16)
        prep.future_recoverability_score = 0.80
        prep._prev_future_recoverability_score = 0.80
        prep.current_target_dist = 2.0
        prep._last_target_distance = 3.0
        prep.return_stall_ema = 0.0
        prep._last_astar_dist = 2.0
        prep._astar_dist = 2.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.3
        prep._last_action = 1
        prep.explored_ratio = 0.5
        prep.dirt_cleaned = 10
        prep.total_dirt = 100
        prep.last_charger_slack = 14.0
        prep.charger_slack = 18.0
        prep.charge_count = 1
        prep.last_charge_count = 0
        prep.pre_charge_battery = 95
        prep.step_no = 2
        prep._prev_charge_need_score = 0.10
        prep._prev_charge_detour_proxy = 0.60
        prep._prev_charge_interrupt_proxy = 0.20
        prep._prev_all_charger_known_path_count = 2.0
        prep._prev_unknown_on_target_path_ratio = 0.0
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": 18.0,
            "margin": 20.0,
            "on_charger": True,
            "unknown_path_ratio": 0.0,
            "charger_target": (16, 16),
            "target_gap": 0.0,
            "suggested_action": 1,
            "return_action_reliable": True,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "target_stable": True,
            "all_charger_known_path_count": 2.0,
            "planner_topk_reachable_count": 2.0,
            "planner_best_target_route_diversity": 1.0,
            "planner_multi_route_recoverability": 0.8,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertLess(components["charge_opportunity_cost_penalty"], 0.0)
        self.assertGreaterEqual(components["risk_release_reward"], 0.0)

    def test_reward_process_in_slice2a_zeroes_risk_release_without_reliable_return_context(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.cleaned_this_step = 0
        prep.new_explored_cells = 0
        prep.consecutive_clean_steps = 0
        prep.cur_visit_count = 1
        prep.wall_adjacent = 0
        prep.dirty_adjacent = 0
        prep.local_frontier_density = 0.0
        prep.local_dirt_density = 0.0
        prep.same_region_streak = 1
        prep.recent_unique_cells_20 = 20
        prep.path_cross_count_50 = 2
        prep.coverage_efficiency_20 = 0.9
        prep.no_progress_steps = 0
        prep.actual_legal_ratio = 1.0
        prep.just_charged = 0.0
        prep.nearest_charger_dist = 3.0
        prep.last_nearest_charger_dist = 8.0
        prep.battery = 120
        prep.battery_max = 200
        prep.nearest_npc_dist = 20.0
        prep.last_move_invalid = 0.0
        prep.stuck_steps = 0
        prep.invalid_move_ema = 0.0
        prep.steps_since_charge = 20
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.route_anchor_center = (16, 16)
        prep.future_recoverability_score = 0.70
        prep._prev_future_recoverability_score = 0.70
        prep.current_target_dist = 8.0
        prep._last_target_distance = 10.0
        prep.return_stall_ema = 0.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.0
        prep.directional_dirty = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        prep._cps_ema = 0.4
        prep._last_action = 0
        prep.explored_ratio = 0.4
        prep.dirt_cleaned = 10
        prep.total_dirt = 100
        prep.last_charger_slack = 12.0
        prep.charger_slack = 18.0
        prep.charge_count = 0
        prep.pre_charge_battery = 120
        prep.step_no = 2
        prep._prev_charge_need_score = 0.60
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 0.0
        prep._prev_unknown_on_target_path_ratio = 0.25
        prep._prev_planner_best_target_route_diversity = 0.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": 18.0,
            "margin": 18.0,
            "on_charger": False,
            "unknown_path_ratio": 0.25,
            "charger_target": (20, 20),
            "target_gap": 2.0,
            "suggested_action": 1,
            "return_action_reliable": False,
            "target_reliable": False,
            "anchor_reliable": False,
            "mode_reliable": False,
            "target_stable": False,
            "all_charger_known_path_count": 0.0,
            "planner_topk_reachable_count": 0.0,
            "planner_best_target_route_diversity": 0.0,
            "planner_multi_route_recoverability": 0.70,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertAlmostEqual(components["risk_release_reward"], 0.0, places=6)
        self.assertAlmostEqual(components["risk_release_from_progress"], 0.0, places=6)
        self.assertAlmostEqual(components["risk_release_from_charge_event"], 0.0, places=6)

    def test_infer_mode_in_strong_heuristic_prioritizes_evade_return_hysteresis_and_pre_return(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.total_charger = 4
        prep.battery_max = 200
        prep.future_recoverability_score = 0.9
        prep.route_contract_pressure = 0.0
        prep.charger_slack = 12.0
        prep.nearest_charger_dist = 10.0
        prep.current_mode = prep.MODE_EXPAND
        prep._get_guidance = lambda: {
            "margin": 18.0,
            "on_charger": False,
            "charger_dist": 10.0,
            "all_charger_known_path_count": 2.0,
            "unknown_path_ratio": 0.0,
            "planner_topk_reachable_count": 2.0,
            "planner_multi_route_recoverability": 0.9,
        }

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_v1"}, clear=False):
            prep.nearest_npc_dist = 3.0
            prep.battery = 50
            self.assertEqual(prep._infer_mode(), prep.MODE_EVADE)

            prep.nearest_npc_dist = 20.0
            prep.current_mode = prep.MODE_RETURN
            prep.battery = 70
            prep._get_guidance = lambda: {
                "margin": 18.0,
                "on_charger": False,
                "charger_dist": 10.0,
                "all_charger_known_path_count": 2.0,
                "unknown_path_ratio": 0.0,
                "planner_topk_reachable_count": 2.0,
                "planner_multi_route_recoverability": 0.9,
            }
            self.assertEqual(prep._infer_mode(), prep.MODE_RETURN)

            prep._get_guidance = lambda: {
                "margin": 18.0,
                "on_charger": True,
                "charger_dist": 10.0,
                "all_charger_known_path_count": 2.0,
                "unknown_path_ratio": 0.0,
                "planner_topk_reachable_count": 2.0,
                "planner_multi_route_recoverability": 0.9,
            }
            prep.battery = 180
            self.assertEqual(prep._infer_mode(), prep.MODE_EXPAND)

            prep.current_mode = prep.MODE_EXPAND
            prep.battery = 100
            prep.charger_slack = 7.0
            prep.route_contract_pressure = 0.55
            prep._get_guidance = lambda: {
                "margin": 18.0,
                "on_charger": False,
                "charger_dist": 10.0,
                "all_charger_known_path_count": 1.0,
                "unknown_path_ratio": 0.25,
                "planner_topk_reachable_count": 1.0,
                "planner_multi_route_recoverability": 0.4,
            }
            self.assertEqual(prep._infer_mode(), prep.MODE_CONTRACT)

    def test_reward_process_in_strong_heuristic_zeroes_heavy_teacher_masks(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.battery = 60
        prep.battery_max = 200
        prep.charger_slack = 2.0
        prep.future_recoverability_score = 0.2
        prep.local_frontier_density = 0.0
        prep.nearest_npc_dist = 20.0
        prep._get_guidance = lambda: {
            "slack": 2.0,
            "margin": 6.0,
            "on_charger": False,
            "charger_dist": 8.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.05,
            "planner_multi_route_recoverability": 0.2,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "return_action_reliable": True,
            "suggested_action": 2,
            "charger_target": (10, 10),
        }
        prep._get_teacher_guidance = lambda: {
            "route_mode": "return",
            "route_anchor": (10, 10),
            "target": (10, 10),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 1.0,
            "target_teacher_mask": 1.0,
            "return_action": 2,
            "return_action_teacher_mask": 1.0,
        }
        prep.sorted_charger_candidates = [
            {
                "center": (10, 10),
                "reachable": 1.0,
                "score": 4.0,
                "astar_dist": 4.0,
                "dist": 4.0,
                "unknown_path_ratio": 0.0,
            }
        ]

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertEqual(components["route_anchor_teacher"], 1)
        self.assertEqual(components["target_teacher"], 1)
        self.assertAlmostEqual(components["route_anchor_teacher_mask"], 0.0, places=6)
        self.assertAlmostEqual(components["target_teacher_mask"], 0.0, places=6)
        self.assertEqual(components["route_phase_action_teacher"], -1)
        self.assertAlmostEqual(components["route_phase_action_teacher_mask"], 0.0, places=6)
        self.assertGreaterEqual(components["return_action_teacher_mask"], 1.0)

    def test_reward_process_in_strong_heuristic_clears_return_action_teacher_outside_return(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_CONTRACT
        prep.battery = 60
        prep.battery_max = 200
        prep.charger_slack = 2.0
        prep.future_recoverability_score = 0.2
        prep.local_frontier_density = 0.0
        prep.nearest_npc_dist = 20.0
        prep._get_guidance = lambda: {
            "slack": 2.0,
            "margin": 6.0,
            "on_charger": False,
            "charger_dist": 8.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.05,
            "planner_multi_route_recoverability": 0.2,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "return_action_reliable": True,
            "suggested_action": 2,
            "charger_target": (10, 10),
        }
        prep._get_teacher_guidance = lambda: {
            "route_mode": "contract",
            "route_anchor": (10, 10),
            "target": (10, 10),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 0.0,
            "target_teacher_mask": 0.0,
            "return_action": 2,
            "return_action_teacher_mask": 0.8,
        }

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertEqual(components["return_action_teacher"], -1)
        self.assertAlmostEqual(components["return_action_teacher_mask"], 0.0, places=6)

    def test_reward_process_in_slice2a_keeps_route_phase_action_teacher_chain(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.battery = 70
        prep.battery_max = 200
        prep.charger_slack = 2.0
        prep.future_recoverability_score = 0.2
        prep.local_frontier_density = 0.0
        prep.nearest_npc_dist = 20.0
        prep.current_target_dist = 6.0
        prep._last_target_distance = 8.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 6.0
        prep.route_anchor_center = (10, 10)
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.step_no = 2
        prep.last_charger_slack = 2.0
        prep.charge_count = 0
        prep.pre_charge_battery = 70
        prep._prev_charge_need_score = 0.18
        prep._prev_route_phase_shadow_risk = 0.25
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.0
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": 2.0,
            "margin": 8.0,
            "on_charger": False,
            "charger_dist": 6.0,
            "all_charger_known_path_count": 1.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.0,
            "planner_multi_route_recoverability": 0.2,
            "target_reliable": True,
            "anchor_reliable": True,
            "mode_reliable": True,
            "return_action_reliable": True,
            "suggested_action": 2,
            "charger_target": (10, 10),
            "target_gap": 4.0,
            "target_stable": True,
            "planner_best_target_route_diversity": 1.0,
        }
        prep._get_teacher_guidance = lambda: {
            "route_mode": "return",
            "route_anchor": (10, 10),
            "target": (10, 10),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 1.0,
            "target_teacher_mask": 1.0,
            "return_action": 2,
            "return_action_teacher_mask": 1.0,
        }

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertAlmostEqual(components["route_anchor_teacher_mask"], 0.0, places=6)
        self.assertAlmostEqual(components["target_teacher_mask"], 0.0, places=6)
        self.assertEqual(components["route_phase_action_teacher"], 2)
        self.assertGreaterEqual(components["route_phase_action_teacher_mask"], 0.8)
        self.assertGreaterEqual(components["return_action_teacher_mask"], 1.0)

    def test_reward_process_in_slice2a_does_not_restore_critical_route_teacher_without_reliability(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_RETURN
        prep.battery = 20
        prep.battery_max = 200
        prep.charger_slack = -3.0
        prep.future_recoverability_score = -0.1
        prep.local_frontier_density = 0.0
        prep.nearest_npc_dist = 20.0
        prep.current_target_dist = 10.0
        prep._last_target_distance = 10.0
        prep._last_astar_dist = 10.0
        prep._astar_dist = 10.0
        prep.route_anchor_center = (10, 10)
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.step_no = 2
        prep.last_charger_slack = -2.0
        prep.charge_count = 0
        prep.pre_charge_battery = 20
        prep._prev_charge_need_score = 0.30
        prep._prev_route_phase_shadow_risk = 0.4
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 0.0
        prep._prev_unknown_on_target_path_ratio = 0.4
        prep._prev_planner_best_target_route_diversity = 0.2
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": -3.0,
            "margin": 4.0,
            "on_charger": False,
            "charger_dist": 10.0,
            "all_charger_known_path_count": 0.0,
            "planner_topk_reachable_count": 0.0,
            "unknown_path_ratio": 0.6,
            "planner_multi_route_recoverability": -0.1,
            "target_reliable": False,
            "anchor_reliable": False,
            "mode_reliable": False,
            "return_action_reliable": False,
            "suggested_action": 2,
            "charger_target": (10, 10),
            "target_gap": 1.0,
            "target_stable": False,
            "planner_best_target_route_diversity": 0.2,
        }
        prep._get_teacher_guidance = lambda: {
            "route_mode": "return",
            "route_anchor": (10, 10),
            "target": (10, 10),
            "mode_teacher_mask": 1.0,
            "route_anchor_teacher_mask": 1.0,
            "target_teacher_mask": 1.0,
            "return_action": 2,
            "return_action_teacher_mask": 1.0,
        }

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertEqual(components["route_phase_action_teacher"], 2)
        self.assertAlmostEqual(components["route_phase_action_teacher_mask"], 0.0, places=6)

    def test_reward_process_in_slice2a_contract_ready_uses_known_route_evidence(self):
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor()
        prep.current_mode = prep.MODE_CONTRACT
        prep.battery = 58
        prep.battery_max = 200
        prep.charger_slack = 1.0
        prep.future_recoverability_score = 0.2
        prep.local_frontier_density = 0.0
        prep.nearest_npc_dist = 20.0
        prep.current_target_dist = 8.0
        prep._last_target_distance = 8.0
        prep._last_astar_dist = 8.0
        prep._astar_dist = 8.0
        prep.route_anchor_center = (10, 10)
        prep.route_anchor_idx = 1
        prep.last_route_anchor_idx = 1
        prep.step_no = 2
        prep.last_charger_slack = 2.0
        prep.charge_count = 0
        prep.pre_charge_battery = 58
        prep._prev_charge_need_score = 0.18
        prep._prev_route_phase_shadow_risk = 0.08
        prep._prev_charge_detour_proxy = 0.0
        prep._prev_charge_interrupt_proxy = 0.0
        prep._prev_all_charger_known_path_count = 1.0
        prep._prev_unknown_on_target_path_ratio = 0.0
        prep._prev_planner_best_target_route_diversity = 1.0
        prep.training_global_step = 0
        prep._get_guidance = lambda: {
            "slack": 1.0,
            "margin": 8.0,
            "on_charger": False,
            "charger_dist": 8.0,
            "all_charger_known_path_count": 0.0,
            "planner_topk_reachable_count": 1.0,
            "unknown_path_ratio": 0.0,
            "planner_multi_route_recoverability": 0.2,
            "target_reliable": False,
            "anchor_reliable": False,
            "mode_reliable": False,
            "return_action_reliable": False,
            "suggested_action": 2,
            "charger_target": (10, 10),
            "target_gap": 1.0,
            "target_stable": False,
            "planner_best_target_route_diversity": 1.0,
        }
        prep._get_teacher_guidance = lambda: None

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_slice2a_v1"}, clear=False):
            _, components = prep.reward_process()

        self.assertGreater(components["route_phase_shadow_risk"], 0.10)
        self.assertAlmostEqual(components["route_phase_reward_ready"], 1.0, places=6)
        self.assertLess(components["route_phase_risk_growth_penalty"], 0.0)

    def test_expert_logit_bias_does_not_refresh_emergency_fallback_state(self):
        from agent_ppo.feature.expert import ExpertPolicy

        expert = ExpertPolicy()
        prep = type("Prep", (), {"cur_pos": (10, 10), "_npcs": [], "current_mode": 4, "MODE_RETURN": 4, "MODE_CONTRACT": 3})()

        with patch.dict(os.environ, {"KAIWU_TRAIN_PHASE": "s1_survival_strong_heuristic_v1"}, clear=False):
            with patch.object(expert, "get_emergency_fallback", side_effect=AssertionError("should not call fallback")):
                with patch.object(
                    expert,
                    "get_charger_signal",
                    return_value={"suggested_action": 2},
                ):
                    bias = expert.get_logit_bias(prep, [1] * Config.ACTION_NUM, last_action=-1)

        self.assertEqual(tuple(bias.shape), (Config.ACTION_NUM,))
        self.assertGreater(float(bias[2]), 0.0)

    def test_episode_sequence_diagnostics_tracks_strong_heuristic_bias_rates(self):
        _install_create_cls_stub()
        from agent_ppo.workflow.train_workflow import EpisodeRunner

        step_records = [
            {"mode": 1, "route_anchor": 0, "target": 0, "charger_slack": 12.0, "future_recoverability_score": 0.9, "anchor_return_dist": 12.0, "is_diag_action": 0.0, "planner_policy_divergence": 0.0, "route_phase_active": 0.0, "route_phase_reliable_active": 0.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.9, "constraint_battery_process_cost": 0.0, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.0, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0, "control_stack_simplify_active": 0.0, "pre_return_readiness_flag": 0.0, "expert_weight_nonzero": 0.0, "pre_return_bias_active": 0.0, "return_bias_active": 0.0},
            {"mode": 3, "route_anchor": 1, "target": 1, "charger_slack": 7.0, "future_recoverability_score": 0.3, "anchor_return_dist": 10.0, "is_diag_action": 0.0, "planner_policy_divergence": 0.0, "route_phase_active": 1.0, "route_phase_reliable_active": 1.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.3, "constraint_battery_process_cost": 0.0, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.0, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0, "control_stack_simplify_active": 0.0, "pre_return_readiness_flag": 0.0, "expert_weight_nonzero": 1.0, "pre_return_bias_active": 1.0, "return_bias_active": 0.0},
            {"mode": 4, "route_anchor": 1, "target": 1, "charger_slack": 2.0, "future_recoverability_score": 0.1, "anchor_return_dist": 7.0, "is_diag_action": 0.0, "planner_policy_divergence": 0.0, "route_phase_active": 1.0, "route_phase_reliable_active": 1.0, "path_cross_count_50": 0.0, "coverage_efficiency_20": 1.0, "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0, "planner_topk_reachable_count": 1.0, "planner_known_route_count_total": 1.0, "planner_best_target_route_diversity": 1.0, "planner_best_target_tangle_cost": 0.0, "planner_best_target_edge_break_cost": 0.0, "planner_best_target_region_fragment_cost": 0.0, "planner_multi_route_recoverability": 0.1, "constraint_battery_process_cost": 0.0, "constraint_collision_process_cost": 0.0, "constraint_high_need_stall_indicator": 0.0, "constraint_charge_need_score": 0.0, "constraint_slack_confidence": 1.0, "wall_hugging_clean_floor": 0.0, "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0, "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0, "control_stack_simplify_active": 0.0, "pre_return_readiness_flag": 0.0, "expert_weight_nonzero": 1.0, "pre_return_bias_active": 0.0, "return_bias_active": 1.0},
        ]

        diagnostics = EpisodeRunner._episode_sequence_diagnostics(step_records)

        self.assertAlmostEqual(diagnostics["expert_weight_nonzero_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(diagnostics["pre_return_bias_active_rate"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(diagnostics["return_bias_active_rate"], 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
