#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_ppo.feature.expert import ExpertPolicy


def _load_agent_symbols():
    common_python_mod = types.ModuleType("common_python")
    config_pkg = types.ModuleType("common_python.config")
    config_control_mod = types.ModuleType("common_python.config.config_control")
    setattr(config_control_mod, "CONFIG", types.SimpleNamespace(svr_name="aisrv"))
    setattr(common_python_mod, "config", config_pkg)
    setattr(config_pkg, "config_control", config_control_mod)
    sys.modules.setdefault("common_python", common_python_mod)
    sys.modules.setdefault("common_python.config", config_pkg)
    sys.modules.setdefault("common_python.config.config_control", config_control_mod)

    kaiwu_define_mod = types.ModuleType("kaiwudrl.common.utils.kaiwudrl_define")
    setattr(kaiwu_define_mod, "KaiwuDRLDefine", types.SimpleNamespace(SERVER_LEARNER="learner"))
    sys.modules.setdefault("kaiwudrl", types.ModuleType("kaiwudrl"))
    sys.modules.setdefault("kaiwudrl.common", types.ModuleType("kaiwudrl.common"))
    sys.modules.setdefault("kaiwudrl.common.utils", types.ModuleType("kaiwudrl.common.utils"))
    sys.modules.setdefault("kaiwudrl.common.utils.kaiwudrl_define", kaiwu_define_mod)

    agent_interface_mod = types.ModuleType("kaiwudrl.interface.agent")
    setattr(agent_interface_mod, "BaseAgent", type("BaseAgent", (), {"__init__": lambda self, *args, **kwargs: None}))
    remote_agent_mod = types.ModuleType("kaiwudrl.interface.remote_agent")
    setattr(remote_agent_mod, "RemoteAgent", type("RemoteAgent", (), {"learn": lambda self, *args, **kwargs: None}))
    sys.modules.setdefault("kaiwudrl.interface", types.ModuleType("kaiwudrl.interface"))
    sys.modules.setdefault("kaiwudrl.interface.agent", agent_interface_mod)
    sys.modules.setdefault("kaiwudrl.interface.remote_agent", remote_agent_mod)

    from agent_ppo.agent import Agent, _fallback_allowed_for_action

    return Agent, _fallback_allowed_for_action


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / ".sisyphus" / "evidence" / "benchmark-900"


class Task7EvalSafetyTests(unittest.TestCase):
    def test_agent_gate_keeps_unsafe_slack_eval_only(self):
        _, allowed = _load_agent_symbols()
        fallback = {"active": True, "action": 0, "reason": "unsafe_slack_return"}

        self.assertFalse(allowed(fallback, use_hard_override=False))
        self.assertTrue(allowed(fallback, use_hard_override=True))

    def test_agent_gate_preserves_existing_critical_runtime_fallbacks(self):
        _, allowed = _load_agent_symbols()

        for reason in ("battery_and_slack_critical", "battery_ratio_critical", "path_margin_critical"):
            with self.subTest(reason=reason):
                fallback = {"active": True, "action": 0, "reason": reason}
                self.assertTrue(allowed(fallback, use_hard_override=False))

    def test_eval_override_summary_reset_makes_episode_counts_independent(self):
        Agent, _ = _load_agent_symbols()
        agent = Agent.__new__(Agent)
        agent.reset_eval_override_summary()

        agent._eval_decision_count = 4
        agent._eval_override_count = 2
        agent._eval_override_reason_counts = {"unsafe_slack_return": 2}
        agent._last_eval_override_reason = "unsafe_slack_return"
        first = agent.get_eval_override_summary()
        self.assertEqual(first["eval_override_count"], 2)
        self.assertEqual(first["eval_override_reason_counts"], {"unsafe_slack_return": 2})

        agent.reset_eval_override_summary()
        cleared = agent.get_eval_override_summary()
        self.assertEqual(cleared["eval_decision_count"], 0)
        self.assertEqual(cleared["eval_override_count"], 0)
        self.assertEqual(cleared["eval_override_reason_counts"], {})
        self.assertIsNone(cleared["last_eval_override_reason"])

        agent._eval_decision_count = 1
        agent._eval_override_count = 1
        agent._eval_override_reason_counts = {"battery_ratio_critical": 1}
        agent._last_eval_override_reason = "battery_ratio_critical"
        second = agent.get_eval_override_summary()
        self.assertEqual(second["eval_override_count"], 1)
        self.assertEqual(second["eval_override_rate"], 1.0)
        self.assertEqual(second["eval_override_reason_counts"], {"battery_ratio_critical": 1})

    def test_unsafe_slack_forces_reachable_eval_return_and_writes_evidence(self):
        expert = ExpertPolicy()
        prep = _PrepStub(battery=9, battery_max=150, stuck_steps=0)
        signal = _signal(slack=-3.0, suggested_action=0, reachable=True)
        expert.get_charger_signal = lambda *_args, **_kwargs: signal

        fallback = expert.get_emergency_fallback(prep, [1] * 8, last_action=-1)

        self.assertTrue(fallback["active"])
        self.assertEqual(fallback["action"], 0)
        self.assertEqual(fallback["reason"], "unsafe_slack_return")

        payload = {
            "schema_version": 1,
            "case": "unsafe_battery_forced_return",
            "controller": "ExpertPolicy.get_emergency_fallback",
            "scope": "eval_hard_override_only",
            "inputs": {
                "battery": prep.battery,
                "battery_max": prep.battery_max,
                "reachable": signal["reachable"],
                "slack": signal["slack"],
                "suggested_action": signal["suggested_action"],
            },
            "result": {
                "active": fallback["active"],
                "action": fallback["action"],
                "reason": fallback["reason"],
            },
            "map_id_used": False,
            "coordinate_table_used": False,
        }
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / "task-7-forced-return.json").write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_positive_slack_does_not_force_eval_return(self):
        expert = ExpertPolicy()
        prep = _PrepStub(battery=80, battery_max=150, stuck_steps=0)
        expert.get_charger_signal = lambda *_args, **_kwargs: _signal(
            slack=20.0,
            suggested_action=0,
            reachable=True,
            battery_ratio=0.53,
        )

        fallback = expert.get_emergency_fallback(prep, [1] * 8, last_action=-1)

        self.assertFalse(fallback["active"])
        self.assertIsNone(fallback["action"])

    def test_return_stall_fallback_prefers_alternate_progress_action(self):
        expert = ExpertPolicy()
        prep = _PrepStub(battery=9, battery_max=150, stuck_steps=3, cur_pos=(5, 5))
        signal = _signal(slack=-3.0, suggested_action=0, reachable=True, charger_target=(8, 5))
        expert.get_charger_signal = lambda *_args, **_kwargs: signal

        fallback = expert.get_emergency_fallback(prep, [1] * 8, last_action=0)

        self.assertTrue(fallback["active"])
        self.assertEqual(fallback["reason"], "unsafe_slack_return")
        self.assertNotEqual(fallback["action"], 0)


class _PrepStub:
    def __init__(self, *, battery, battery_max, stuck_steps, cur_pos=(5, 5)):
        self.battery = battery
        self.battery_max = battery_max
        self.stuck_steps = stuck_steps
        self.cur_pos = cur_pos


def _signal(*, slack, suggested_action, reachable, charger_target=(8, 5), battery_ratio=0.06):
    return {
        "battery_ratio": battery_ratio,
        "on_charger": False,
        "slack": slack,
        "charger_dist": 12.0,
        "margin": 4.0,
        "suggested_action": suggested_action,
        "reachable": reachable,
        "charger_path": [(5, 5), (6, 5), (7, 5), (8, 5)] if reachable else [],
        "charger_target": charger_target,
    }


if __name__ == "__main__":
    unittest.main()
