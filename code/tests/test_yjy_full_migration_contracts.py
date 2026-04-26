#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import ast
import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_runtime_stubs():
    common_python_mod = types.ModuleType("common_python")
    config_mod = types.ModuleType("common_python.config")
    config_control_mod = types.ModuleType("common_python.config.config_control")
    utils_mod = types.ModuleType("common_python.utils")
    common_func_mod = types.ModuleType("common_python.utils.common_func")
    kaiwudrl_mod = types.ModuleType("kaiwudrl")
    kaiwudrl_interface_mod = types.ModuleType("kaiwudrl.interface")
    kaiwudrl_interface_agent_mod = types.ModuleType("kaiwudrl.interface.agent")

    class _Config:
        pass

    class _BaseAgent:
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
    kaiwudrl_interface_agent_mod.BaseAgent = _BaseAgent
    kaiwudrl_interface_mod.agent = kaiwudrl_interface_agent_mod
    kaiwudrl_mod.interface = kaiwudrl_interface_mod

    sys.modules["common_python"] = common_python_mod
    sys.modules["common_python.config"] = config_mod
    sys.modules["common_python.config.config_control"] = config_control_mod
    sys.modules["common_python.utils"] = utils_mod
    sys.modules["common_python.utils.common_func"] = common_func_mod
    sys.modules["kaiwudrl"] = kaiwudrl_mod
    sys.modules["kaiwudrl.interface"] = kaiwudrl_interface_mod
    sys.modules["kaiwudrl.interface.agent"] = kaiwudrl_interface_agent_mod


_install_runtime_stubs()

from agent_ppo.conf.conf import Config
from agent_ppo.model.model import Model


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_PATH = REPO_ROOT / "code" / "agent_ppo" / "agent.py"
WORKFLOW_PATH = REPO_ROOT / "code" / "agent_ppo" / "workflow" / "train_workflow.py"
TRAIN_ENV_CONF_PATH = REPO_ROOT / "code" / "agent_ppo" / "conf" / "train_env_conf.toml"


def _read_assignment_map(path: Path):
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


class YjyFullMigrationContractsTests(unittest.TestCase):
    def test_config_matches_yjy_simple_84d_contract(self):
        self.assertEqual(Config.FEATURE_SPLIT_SHAPE, [49, 27, 8])
        self.assertEqual(Config.DIM_OF_OBSERVATION, 84)
        self.assertEqual(Config.VALUE_NUM, 1)
        self.assertEqual(Config.HIDDEN_DIM_1, 256)
        self.assertEqual(Config.HIDDEN_DIM_2, 128)

    def test_model_is_simple_actor_critic_for_flat_obs(self):
        model = Model(device=torch.device("cpu"))
        batch = torch.randn(3, Config.DIM_OF_OBSERVATION, dtype=torch.float32)
        outputs = model(batch, inference=True)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].shape), (3, Config.ACTION_NUM))
        self.assertEqual(tuple(outputs[1].shape), (3, 1))

    def test_agent_ast_exposes_guided_predict(self):
        tree = ast.parse(AGENT_PATH.read_text(encoding="utf-8"))
        agent_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Agent"
        )
        method_names = {node.name for node in agent_class.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("guided_predict", method_names)

    def test_workflow_ast_imports_coverage_planner_from_algorithm(self):
        tree = ast.parse(WORKFLOW_PATH.read_text(encoding="utf-8"))
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "agent_ppo.algorithm.algorithm"
        ]
        imported_names = {alias.name for node in imports for alias in node.names}
        self.assertIn("CoveragePlanner", imported_names)

    def test_train_env_conf_matches_yjy_profile(self):
        assignments = _read_assignment_map(TRAIN_ENV_CONF_PATH)
        self.assertEqual(assignments["map_random"], "false")
        self.assertEqual(assignments["robot_count"], "4")
        self.assertEqual(assignments["charger_count"], "3")
        self.assertEqual(assignments["max_step"], "1000")
        self.assertEqual(assignments["battery_max"], "150")


if __name__ == "__main__":
    unittest.main()
