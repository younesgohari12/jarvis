from __future__ import annotations

import unittest

from jarvis.agent.permissions import PermissionLayer
from jarvis.tools.registry import ToolError, ToolRegistry, ToolSpec
from tests.helpers import TemporaryRuntime


class PermissionTests(unittest.TestCase):
    def test_safe_read_only_tool_runs(self) -> None:
        registry = ToolRegistry(PermissionLayer())
        registry.register(ToolSpec("safe", "safe", "read_only", lambda args: args.get("x")))
        self.assertEqual(registry.invoke("safe", {"x": 7}), 7)

    def test_dangerous_tool_requires_confirmation(self) -> None:
        registry = ToolRegistry(PermissionLayer())
        registry.register(ToolSpec("danger", "danger", "system_control", lambda _args: True))
        with self.assertRaises(ToolError):
            registry.invoke("danger")
        self.assertTrue(registry.invoke("danger", confirmed=True))

    def test_network_tool_is_safe_and_runs_only_when_registry_is_invoked(self) -> None:
        called = False

        def handler(_args: dict[str, object]) -> bool:
            nonlocal called
            called = True
            return True

        registry = ToolRegistry(PermissionLayer())
        registry.register(ToolSpec("network", "network", "network", handler))
        self.assertTrue(registry.invoke("network"))
        self.assertTrue(called)

    def test_unknown_permission_category_is_denied(self) -> None:
        decision = PermissionLayer().check("magic_admin", confirmed=True)
        self.assertFalse(decision.allowed)
        self.assertIn("Unknown", decision.reason)

    def test_duplicate_or_invalid_tool_names_are_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolSpec("demo", "demo", "information", lambda _args: True))
        with self.assertRaises(ToolError):
            registry.register(ToolSpec("demo", "duplicate", "information", lambda _args: True))
        with self.assertRaises(ToolError):
            registry.register(ToolSpec("bad name", "invalid", "information", lambda _args: True))

    def test_default_runtime_exposes_guarded_desktop_tools(self) -> None:
        with TemporaryRuntime() as runtime:
            names = set(runtime.tools.names())
            self.assertTrue({"shutdown_system", "restart_system", "delete_file", "run_command"} <= names)
            for name in ("shutdown_system", "restart_system", "delete_file"):
                specification = runtime.tools.spec(name)
                self.assertIsNotNone(specification)
                decision = PermissionLayer().check(specification.category, confirmed=False)
                self.assertFalse(decision.allowed)
                self.assertTrue(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
