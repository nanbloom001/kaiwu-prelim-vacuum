#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import unittest
from io import StringIO
from unittest import mock

sys.path.insert(0, "/workspace/code")

from agent_ppo.utils.container_routing import (
    AisrvRoutingResult,
    _read_current_container_name,
    _validate_gpu_value,
    build_shell_exports,
    detect_aisrv_routing,
    main,
    resolve_service_index,
    select_aisrv_gpu_target,
)


class ContainerRoutingTests(unittest.TestCase):
    def test_resolve_service_index_matches_current_ip(self):
        service_ips = ["172.19.0.14", "172.19.0.15", "172.19.0.16"]

        self.assertEqual(resolve_service_index("172.19.0.15", service_ips), 2)

    def test_resolve_service_index_returns_none_when_ip_missing(self):
        service_ips = ["172.19.0.14", None, "172.19.0.16"]

        self.assertIsNone(resolve_service_index("172.19.0.15", service_ips))

    def test_select_aisrv_gpu_target_uses_group_boundaries(self):
        self.assertEqual(
            select_aisrv_gpu_target(
                aisrv_index=1,
                gpu1_num=1,
                gpu2_num=1,
                gpu1="1",
                gpu2="2",
                gpu3="3",
            ),
            "1",
        )
        self.assertEqual(
            select_aisrv_gpu_target(
                aisrv_index=2,
                gpu1_num=1,
                gpu2_num=1,
                gpu1="1",
                gpu2="2",
                gpu3="3",
            ),
            "2",
        )
        self.assertEqual(
            select_aisrv_gpu_target(
                aisrv_index=3,
                gpu1_num=1,
                gpu2_num=1,
                gpu1="1",
                gpu2="2",
                gpu3="3",
            ),
            "3",
        )

    def test_select_aisrv_gpu_target_returns_none_for_invalid_index(self):
        self.assertIsNone(
            select_aisrv_gpu_target(
                aisrv_index=None,
                gpu1_num=1,
                gpu2_num=1,
                gpu1="1",
                gpu2="2",
                gpu3="3",
            )
        )

    def test_detect_aisrv_routing_resolves_index_and_gpu_with_mocked_dns(self):
        dns_table = {
            "container-abc": "172.19.0.15",
            "kaiwu-train-aisrv-1": "172.19.0.14",
            "kaiwu-train-aisrv-2": "172.19.0.15",
        }

        def fake_resolve(*names):
            for name in names:
                if name in dns_table:
                    return dns_table[name]
            return None

        with mock.patch("agent_ppo.utils.container_routing._resolve_name_ip", side_effect=fake_resolve):
            result = detect_aisrv_routing(
                current_name="container-abc",
                service_count=2,
                project_name="kaiwu-train",
                gpu1_num=1,
                gpu2_num=1,
                gpu1="1",
                gpu2="2",
                gpu3="3",
            )

        self.assertEqual(result.current_ip, "172.19.0.15")
        self.assertEqual(result.aisrv_index, 2)
        self.assertEqual(result.target_gpu, "2")
        self.assertTrue(result.is_resolved)

    def test_detect_aisrv_routing_returns_unresolved_when_dns_missing(self):
        with mock.patch("agent_ppo.utils.container_routing._resolve_name_ip", return_value=None):
            result = detect_aisrv_routing(
                current_name="container-abc",
                service_count=2,
                project_name="kaiwu-train",
                gpu1_num=1,
                gpu2_num=1,
                gpu1="1",
                gpu2="2",
                gpu3="3",
                resolution_attempts=2,
                retry_delay_seconds=0.0,
            )

        self.assertIsNone(result.current_ip)
        self.assertIsNone(result.aisrv_index)
        self.assertIsNone(result.target_gpu)
        self.assertFalse(result.is_resolved)

    def test_build_shell_exports_emits_only_meaningful_values(self):
        result = AisrvRoutingResult(
            current_name="container-abc",
            current_ip="172.19.0.15",
            aisrv_index=2,
            target_gpu="2",
        )

        exports = build_shell_exports(result)

        self.assertIn("export KAIWU_AISRV_CURRENT_NAME=container-abc", exports)
        self.assertIn("export KAIWU_AISRV_CURRENT_IP=172.19.0.15", exports)
        self.assertIn("export KAIWU_AISRV_INDEX=2", exports)
        self.assertIn("export KAIWU_AISRV_TARGET_GPU=2", exports)
        self.assertIn("export CUDA_VISIBLE_DEVICES=2", exports)
        self.assertIn("export NVIDIA_VISIBLE_DEVICES=2", exports)

    def test_build_shell_exports_skips_empty_routing_values(self):
        result = AisrvRoutingResult(
            current_name="container-abc",
            current_ip=None,
            aisrv_index=None,
            target_gpu=None,
        )

        exports = build_shell_exports(result)

        self.assertIn("export KAIWU_AISRV_CURRENT_NAME=container-abc", exports)
        self.assertNotIn("KAIWU_AISRV_INDEX", exports)
        self.assertNotIn("KAIWU_AISRV_TARGET_GPU", exports)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", exports)

    def test_read_current_container_name_uses_explicit_value(self):
        self.assertEqual(_read_current_container_name("container-abc"), "container-abc")

    def test_read_current_container_name_raises_when_hostname_file_missing(self):
        with mock.patch("agent_ppo.utils.container_routing.os.environ.get", return_value=None), mock.patch(
            "builtins.open", side_effect=FileNotFoundError("missing")
        ):
            with self.assertRaises(RuntimeError):
                _read_current_container_name(None)

    def test_validate_gpu_value_rejects_empty_string(self):
        with self.assertRaises(ValueError):
            _validate_gpu_value("   ", "gpu1")

    def test_main_returns_zero_on_success(self):
        with mock.patch(
            "agent_ppo.utils.container_routing._read_current_container_name",
            return_value="container-abc",
        ), mock.patch(
            "agent_ppo.utils.container_routing.detect_aisrv_routing",
            return_value=AisrvRoutingResult(
                current_name="container-abc",
                current_ip="172.19.0.15",
                aisrv_index=2,
                target_gpu="2",
            ),
        ), mock.patch("sys.stdout", new=StringIO()):
            exit_code = main(
                [
                    "--service-count",
                    "2",
                    "--project-name",
                    "kaiwu-train",
                    "--gpu1-num",
                    "1",
                    "--gpu2-num",
                    "1",
                    "--gpu1",
                    "1",
                    "--gpu2",
                    "2",
                    "--gpu3",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 0)

    def test_main_returns_one_on_input_validation_failure(self):
        with mock.patch("sys.stderr", new=StringIO()):
            exit_code = main(
                [
                    "--service-count",
                    "2",
                    "--project-name",
                    "kaiwu-train",
                    "--gpu1-num",
                    "1",
                    "--gpu2-num",
                    "1",
                    "--gpu1",
                    "   ",
                    "--gpu2",
                    "2",
                    "--gpu3",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()