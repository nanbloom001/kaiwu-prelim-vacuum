#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - host test environment may not have torch
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_runtime_stubs():
    common_python_mod = types.ModuleType("common_python")
    config_mod = types.ModuleType("common_python.config")
    config_control_mod = types.ModuleType("common_python.config.config_control")
    utils_mod = types.ModuleType("common_python.utils")
    common_func_mod = types.ModuleType("common_python.utils.common_func")

    class _Config:
        pass

    def create_cls(name, **defaults):
        attrs = dict(defaults)

        def __init__(self, **kwargs):
            for key, default in defaults.items():
                setattr(self, key, kwargs.get(key, default))

        attrs["__init__"] = __init__
        return type(name, (), attrs)

    config_control_mod.CONFIG = _Config()
    config_mod.config_control = config_control_mod
    common_func_mod.create_cls = create_cls
    utils_mod.common_func = common_func_mod
    common_python_mod.config = config_mod
    common_python_mod.utils = utils_mod

    sys.modules["common_python"] = common_python_mod
    sys.modules["common_python.config"] = config_mod
    sys.modules["common_python.config.config_control"] = config_control_mod
    sys.modules["common_python.utils"] = utils_mod
    sys.modules["common_python.utils.common_func"] = common_func_mod


_install_runtime_stubs()

if torch is not None:
    from agent_ppo.agent import Agent, _build_runtime_probe_payload, _emit_runtime_probe_once
    from agent_ppo.algorithm.algorithm import Algorithm
    from agent_ppo.conf.conf import Config
    from agent_ppo.workflow.train_workflow import PerfWindow


@unittest.skipIf(torch is None, "torch is not available in the host test environment")
class AgentLoadModelCacheTests(unittest.TestCase):
    def _build_agent_stub(self):
        agent = Agent.__new__(Agent)
        agent.device = torch.device("cpu")
        agent.model = mock.Mock()
        agent.model.state_dict.return_value = {"w": torch.tensor([1.0])}
        agent.model.load_state_dict.return_value = ([], [])
        agent.logger = mock.Mock()
        agent.archive = mock.Mock()
        agent.enable_load_model_cache = True
        agent.current_model_ref = {"path": None, "id": None, "checkpoint_id": None}
        agent._last_loaded_model_path = None
        agent._last_loaded_model_mtime_ns = None
        agent._model_load_call_count = 0
        agent._model_load_reload_count = 0
        agent._model_load_cache_hit_count = 0
        agent._last_loaded_checkpoint_step = None
        agent._last_real_model_reload_ts = 0.0
        agent._load_model_transition_guard = False
        agent._load_model_stage_transition_cooldown = False
        return agent

    def test_load_model_skips_reload_when_file_unchanged(self):
        agent = self._build_agent_stub()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.ckpt-latest.pkl"
            model_path.write_bytes(b"model-v1")

            with mock.patch("agent_ppo.agent.torch.load", return_value={"w": torch.tensor([1.0])}) as mocked_load:
                Agent._business_load_model(agent, path=tmp_dir, id="latest")
                Agent._business_load_model(agent, path=tmp_dir, id="latest")

            self.assertEqual(mocked_load.call_count, 1)
            self.assertEqual(agent._model_load_reload_count, 1)
            self.assertEqual(agent._model_load_cache_hit_count, 1)
            agent.model.load_state_dict.assert_called_once()

    def test_load_model_skips_small_checkpoint_gap_within_reload_interval(self):
        agent = self._build_agent_stub()
        agent._last_loaded_model_path = "/tmp/model.ckpt-1000.pkl"
        agent._last_loaded_model_mtime_ns = 10
        agent._last_loaded_checkpoint_step = 1000
        agent._last_real_model_reload_ts = 1000.0
        with mock.patch("agent_ppo.agent.time.time", return_value=1100.0):
            should_reload = Agent._should_reload_model(
                agent,
                "/tmp/model.ckpt-1500.pkl",
                20,
                checkpoint_id="1500",
                now_ts=1100.0,
            )
        self.assertFalse(should_reload)

    def test_load_model_allows_reload_for_large_checkpoint_gap(self):
        agent = self._build_agent_stub()
        agent._last_loaded_model_path = "/tmp/model.ckpt-1000.pkl"
        agent._last_loaded_model_mtime_ns = 10
        agent._last_loaded_checkpoint_step = 1000
        agent._last_real_model_reload_ts = 1000.0
        should_reload = Agent._should_reload_model(
            agent,
            "/tmp/model.ckpt-2500.pkl",
            20,
            checkpoint_id="2500",
            now_ts=1101.0,
        )
        self.assertTrue(should_reload)

    def test_observation_process_uses_runtime_progress_payload_instead_of_latest_checkpoint_id(self):
        agent = self._build_agent_stub()
        agent.preprocessor = mock.Mock()
        agent.preprocessor.feature_process.return_value = ([0.0] * Config.DIM_OF_OBSERVATION, [1.0] * Config.ACTION_NUM, 0.0, {})
        agent.last_action = -1
        agent.last_reward = 0.0
        agent.reward_components = {}
        agent._last_fallback_active = 0.0
        agent.current_model_ref = {
            "path": "/tmp/model.ckpt-latest.pkl",
            "id": "latest",
            "checkpoint_id": "latest",
            "checkpoint_step": None,
            "global_step_since_resume": None,
        }

        Agent.observation_process(agent, {"runtime": {"global_step_since_resume": 4321}})

        passed_payload = agent.preprocessor.feature_process.call_args.args[0]
        self.assertEqual(passed_payload["runtime"]["global_step_since_resume"], 4321)

    def test_safe_fallback_route_anchor_prob_matches_config_dim(self):
        agent = self._build_agent_stub()

        act_data = Agent._build_safe_fallback_act_data(agent, [1.0] * Config.ACTION_NUM, 0.0, 0.0, 0.0)

        self.assertEqual(tuple(act_data.route_anchor_prob.shape), (Config.ROUTE_ANCHOR_DIM,))


@unittest.skipIf(torch is None, "torch is not available in the host test environment")
class AlgorithmBatchTensorTests(unittest.TestCase):
    def test_unpack_batch_tensor_preserves_expected_shapes(self):
        algo = Algorithm.__new__(Algorithm)
        algo.device = torch.device("cpu")
        algo.use_amp = False
        if not hasattr(algo, "_unpack_batch_tensor"):
            self.skipTest("Algorithm batch unpack helper no longer exists in current implementation")

        total_dim = (
            Config.DIM_OF_OBSERVATION
            + Config.ACTION_NUM
            + 1
            + Config.VALUE_NUM
            + Config.VALUE_NUM
            + 1
            + Config.VALUE_NUM
            + Config.VALUE_NUM
            + Config.VALUE_NUM
            + Config.ACTION_NUM
            + Config.SAMPLE_CONSTRAINT_COST_DIM
        )
        batch = torch.arange(total_dim * 2, dtype=torch.float32).view(2, total_dim)

        obs, legal, act, old_prob, old_value, reward_sum, advantage, reward = algo._unpack_batch_tensor(batch)

        self.assertEqual(obs.shape, (2, Config.DIM_OF_OBSERVATION))
        self.assertEqual(legal.shape, (2, Config.ACTION_NUM))
        self.assertEqual(act.shape, (2, 1))
        self.assertEqual(old_prob.shape, (2, Config.ACTION_NUM))
        self.assertEqual(old_value.shape, (2, Config.VALUE_NUM))
        self.assertEqual(reward_sum.shape, (2, Config.VALUE_NUM))
        self.assertEqual(advantage.shape, (2, Config.VALUE_NUM))
        self.assertEqual(reward.shape, (2, Config.VALUE_NUM))
        self.assertTrue(torch.equal(obs[0, :4], batch[0, :4]))

    def test_build_batch_from_field_map_keeps_expert_weight(self):
        algo = Algorithm.__new__(Algorithm)
        algo.device = torch.device("cpu")
        algo.use_amp = False

        field_map = {
            "obs": torch.zeros((1, Config.SEQ_CHUNK_LEN, Config.DIM_OF_OBSERVATION)),
            "legal_action": torch.zeros((1, Config.SEQ_CHUNK_LEN, Config.ACTION_NUM)),
            "act": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "prob": torch.zeros((1, Config.SEQ_CHUNK_LEN, Config.ACTION_NUM)),
            "reward_clean": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "reward_survive": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "done": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "value_clean": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "value_survive": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "advantage_clean": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "advantage_survive": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "mode_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "route_anchor_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "target_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "mode_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "route_anchor_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "target_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "return_action_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "return_action_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "battery_risk_label": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "collision_risk_label": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "constraint_battery_process_cost": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "fallback_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "expert_weight": torch.full((1, Config.SEQ_CHUNK_LEN), 0.25),
        }

        batch = Algorithm._build_batch_from_field_map(algo, field_map)

        self.assertIn("expert_weight", batch)
        self.assertTrue(torch.allclose(batch["expert_weight"], torch.full((1, Config.SEQ_CHUNK_LEN), 0.25)))

    def test_build_batch_from_field_map_accepts_scalar_expert_weight_from_flat_batch_path(self):
        algo = Algorithm.__new__(Algorithm)
        algo.device = torch.device("cpu")
        algo.use_amp = False

        field_map = {
            "obs": torch.zeros((1, Config.SEQ_CHUNK_LEN, Config.DIM_OF_OBSERVATION)),
            "legal_action": torch.zeros((1, Config.SEQ_CHUNK_LEN, Config.ACTION_NUM)),
            "act": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "prob": torch.zeros((1, Config.SEQ_CHUNK_LEN, Config.ACTION_NUM)),
            "reward_clean": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "reward_survive": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "done": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "value_clean": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "value_survive": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "advantage_clean": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "advantage_survive": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "mode_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "route_anchor_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "target_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "mode_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "route_anchor_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "target_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "return_action_teacher": torch.zeros((1, Config.SEQ_CHUNK_LEN), dtype=torch.long),
            "return_action_teacher_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "battery_risk_label": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "collision_risk_label": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "constraint_battery_process_cost": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "fallback_mask": torch.zeros((1, Config.SEQ_CHUNK_LEN)),
            "expert_weight": torch.tensor([[0.25]]),
        }

        batch = Algorithm._build_batch_from_field_map(algo, field_map)

        self.assertEqual(tuple(batch["expert_weight"].shape), (1, 1))
        self.assertTrue(torch.allclose(batch["expert_weight"], torch.tensor([[0.25]])))

    def test_battery_process_cost_mean_uses_constraint_cost_not_reward_survive(self):
        batch = {
            "constraint_battery_process_cost": torch.tensor([[0.10] * Config.SEQ_CHUNK_LEN], dtype=torch.float32),
            "reward_survive": torch.tensor([[5.0] * Config.SEQ_CHUNK_LEN], dtype=torch.float32),
        }
        valid_mask = torch.ones((1, Config.SEQ_LEARN_LEN), dtype=torch.float32)
        valid_count = valid_mask.sum()

        mean = Algorithm._battery_process_cost_mean(
            batch,
            slice(Config.SEQ_BURN_IN, Config.SEQ_CHUNK_LEN),
            valid_mask,
            valid_count,
        )

        self.assertAlmostEqual(float(mean.item()), 0.10, places=5)


@unittest.skipIf(torch is None, "torch is not available in the host test environment")
class PerfWindowTests(unittest.TestCase):
    def test_flush_returns_averages_and_resets(self):
        perf = PerfWindow()
        perf.add("predict", 10.0)
        perf.add("predict", 20.0)
        perf.add("samples_sent", 0.0, count=8)

        payload = perf.flush("episode")

        self.assertEqual(payload["episode_predict_count"], 2)
        self.assertEqual(payload["episode_predict_total_ms"], 30.0)
        self.assertEqual(payload["episode_predict_avg_ms"], 15.0)
        self.assertEqual(payload["episode_samples_sent_count"], 8)
        self.assertEqual(perf.values, {})


@unittest.skipIf(torch is None, "torch is not available in the host test environment")
class RuntimeProbeTests(unittest.TestCase):
    def test_build_runtime_probe_payload_includes_devices_and_config(self):
        model = torch.nn.Linear(2, 2)
        algorithm = types.SimpleNamespace(device=torch.device("cpu"))
        config_stub = types.SimpleNamespace(
            svr_name="aisrv",
            train_batch_size=4096,
            predict_batch_size=128,
            proxy_batch_size=128,
            send_sample_size=4096,
            replay_buffer_type="zmq",
            reverb_sampler="reverb.selectors.Fifo",
            reverb_rate_limiter="MinSize",
            pytorch_read_data_from_reverb_type=1,
        )

        with mock.patch.dict(
            "os.environ",
            {
                "CUDA_VISIBLE_DEVICES": "1",
                "NVIDIA_VISIBLE_DEVICES": "1",
                "KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE": "zmq",
                "KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE": "1",
            },
            clear=False,
        ), mock.patch("agent_ppo.agent.KAIWU_CONFIG", new=config_stub):
            payload = _build_runtime_probe_payload(
                stage="predict",
                service_name="aisrv",
                requested_device=torch.device("cpu"),
                model=model,
                algorithm=algorithm,
                use_amp=False,
                extra={"input": {"device": "cpu", "type": "torch.Tensor"}},
            )

        self.assertEqual(payload["stage"], "predict")
        self.assertEqual(payload["requested_device"], "cpu")
        self.assertEqual(payload["model_param_device"], "cpu")
        self.assertEqual(payload["algorithm_device"], "cpu")
        self.assertEqual(payload["config"]["replay_buffer_type"], "zmq")
        self.assertEqual(payload["env"]["KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE"], "zmq")
        self.assertEqual(payload["input"]["device"], "cpu")

    def test_emit_runtime_probe_once_logs_only_once_per_stage(self):
        logger = mock.Mock()
        seen = set()
        payload = {"stage": "learn", "service_name": "learner"}

        first = _emit_runtime_probe_once(logger, seen, "learn", payload)
        second = _emit_runtime_probe_once(logger, seen, "learn", payload)

        self.assertTrue(first)
        self.assertFalse(second)
        logger.info.assert_called_once()
        self.assertIn("runtime_probe", logger.info.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
