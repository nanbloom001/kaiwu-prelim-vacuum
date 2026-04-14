#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, "/workspace/code")

from agent_ppo.agent import Agent, _build_runtime_probe_payload, _emit_runtime_probe_once
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.workflow.train_workflow import PerfWindow


class AgentLoadModelCacheTests(unittest.TestCase):
    def _build_agent_stub(self):
        agent = Agent.__new__(Agent)
        agent.device = torch.device("cpu")
        agent.model = mock.Mock()
        agent.logger = mock.Mock()
        agent.archive = mock.Mock()
        agent.enable_load_model_cache = True
        agent.current_model_ref = {"path": None, "id": None, "checkpoint_id": None}
        agent._last_loaded_model_path = None
        agent._last_loaded_model_mtime_ns = None
        agent._model_load_call_count = 0
        agent._model_load_reload_count = 0
        agent._model_load_cache_hit_count = 0
        return agent

    def test_load_model_skips_reload_when_file_unchanged(self):
        agent = self._build_agent_stub()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.ckpt-latest.pkl"
            model_path.write_bytes(b"model-v1")

            with mock.patch("agent_ppo.agent.torch.load", return_value={"w": 1}) as mocked_load:
                Agent._business_load_model(agent, path=tmp_dir, id="latest")
                Agent._business_load_model(agent, path=tmp_dir, id="latest")

            self.assertEqual(mocked_load.call_count, 1)
            self.assertEqual(agent._model_load_reload_count, 1)
            self.assertEqual(agent._model_load_cache_hit_count, 1)
            agent.model.load_state_dict.assert_called_once()


class AlgorithmBatchTensorTests(unittest.TestCase):
    def test_unpack_batch_tensor_preserves_expected_shapes(self):
        algo = Algorithm.__new__(Algorithm)
        algo.device = torch.device("cpu")
        algo.use_amp = False

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
