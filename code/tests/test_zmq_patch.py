#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "/workspace/code")

from agent_ppo.utils.zmq_patch import (
    ENTRYPOINT_SPAWN_TARGETS,
    START_METHOD_PATCH_MARKER,
    ZMQ_SPAWN_TARGETS,
    SPAWN_PATCH_MARKER,
    inject_spawn_context_prelude,
    inject_spawn_start_method_prelude,
    patch_zmq_entrypoints,
    patch_zmq_runtime_files,
)


class ZmqPatchTests(unittest.TestCase):
    def test_inject_spawn_start_method_prelude_is_idempotent(self):
        source = "import os\n"

        updated, changed = inject_spawn_start_method_prelude(source)
        updated_again, changed_again = inject_spawn_start_method_prelude(updated)

        self.assertTrue(changed)
        self.assertFalse(changed_again)
        self.assertIn(START_METHOD_PATCH_MARKER, updated)
        self.assertEqual(updated, updated_again)

    def test_patch_zmq_entrypoints_only_patches_when_zmq_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for relative in ENTRYPOINT_SPAWN_TARGETS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("import multiprocessing\n")

            skipped = patch_zmq_entrypoints(root, "reverb")
            self.assertEqual(skipped, [])

            patched = patch_zmq_entrypoints(root, "zmq")

            self.assertEqual(sorted(path.relative_to(root) for path in patched), sorted(ENTRYPOINT_SPAWN_TARGETS))
            for relative in ENTRYPOINT_SPAWN_TARGETS:
                self.assertIn(START_METHOD_PATCH_MARKER, (root / relative).read_text())

    def test_inject_spawn_context_prelude_handles_invalid_python_with_fallback(self):
        source = "this is not valid python !!!\n"

        updated, changed = inject_spawn_context_prelude(source)

        self.assertTrue(changed)
        self.assertIn("import multiprocessing as _copilot_mp", updated)
        self.assertTrue(updated.endswith(source))

    def test_inject_spawn_context_prelude_preserves_encoding_declaration(self):
        source = "# -*- coding: utf-8 -*-\nimport os\n"

        updated, changed = inject_spawn_context_prelude(source)

        self.assertTrue(changed)
        self.assertLess(updated.index("# -*- coding: utf-8 -*-"), updated.index("import multiprocessing as _copilot_mp"))

    def test_inject_spawn_context_prelude_preserves_future_import_position(self):
        source = (
            "#!/usr/bin/env python3\n"
            "# -*- coding: UTF-8 -*-\n"
            '\"\"\"module docstring\"\"\"\n'
            "from __future__ import annotations\n"
            "import os\n"
            "from multiprocessing import Process\n"
        )

        updated, changed = inject_spawn_context_prelude(source)

        self.assertTrue(changed)
        self.assertIn(SPAWN_PATCH_MARKER, updated)
        self.assertEqual(updated.count(f"{SPAWN_PATCH_MARKER} = True"), 1)
        self.assertLess(updated.index("from __future__ import annotations"), updated.index("import multiprocessing as _copilot_mp"))
        self.assertLess(updated.index("import multiprocessing as _copilot_mp"), updated.index("import os"))

    def test_inject_spawn_context_prelude_is_idempotent(self):
        source = "import os\n"

        updated, changed = inject_spawn_context_prelude(source)
        updated_again, changed_again = inject_spawn_context_prelude(updated)

        self.assertTrue(changed)
        self.assertFalse(changed_again)
        self.assertEqual(updated, updated_again)

    def test_patch_zmq_runtime_files_only_patches_when_zmq_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for relative in ZMQ_SPAWN_TARGETS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("import multiprocessing\n")

            skipped = patch_zmq_runtime_files(root, "reverb")
            self.assertEqual(skipped, [])
            for relative in ZMQ_SPAWN_TARGETS:
                self.assertNotIn(SPAWN_PATCH_MARKER, (root / relative).read_text())

            patched = patch_zmq_runtime_files(root, "zmq")

            self.assertEqual(sorted(path.relative_to(root) for path in patched), sorted(ZMQ_SPAWN_TARGETS))
            for relative in ZMQ_SPAWN_TARGETS:
                self.assertIn(SPAWN_PATCH_MARKER, (root / relative).read_text())


if __name__ == "__main__":
    unittest.main()