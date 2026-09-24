# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dependency-light checks for the SQLite autotune cache lock policy."""

import ast
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock


SQL_SOURCE = (
    Path(__file__).parents[2]
    / "src"
    / "flag_gems"
    / "utils"
    / "models"
    / "sql.py"
)


class FakeOperationalError(Exception):
    pass


def _load_retry(sqlite_name: str):
    tree = ast.parse(SQL_SOURCE.read_text())
    retry = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "retry_sqlite_locked"
    )
    sleep = Mock()
    namespace = {
        "wraps": wraps,
        "sqlalchemy": SimpleNamespace(
            exc=SimpleNamespace(OperationalError=FakeOperationalError)
        ),
        "time": SimpleNamespace(sleep=sleep),
    }
    module = ast.fix_missing_locations(ast.Module(body=[retry], type_ignores=[]))
    exec(compile(module, str(SQL_SOURCE), "exec"), namespace)
    owner = SimpleNamespace(
        engine=SimpleNamespace(dialect=SimpleNamespace(name=sqlite_name))
    )
    return namespace["retry_sqlite_locked"], owner, sleep


class SQLiteLockRetryTest(TestCase):
    def test_transient_sqlite_lock_retries(self) -> None:
        retry, owner, sleep = _load_retry("sqlite")
        call_count = 0

        def operation(_owner):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeOperationalError("database is locked")
            return "ready"

        self.assertEqual(retry(operation)(owner), "ready")
        self.assertEqual(call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.1, 0.2])

    def test_other_errors_and_other_databases_are_not_retried(self) -> None:
        for dialect, message in (
            ("sqlite", "disk I/O error"),
            ("postgresql", "database is locked"),
        ):
            with self.subTest(dialect=dialect, message=message):
                retry, owner, sleep = _load_retry(dialect)
                operation = Mock(side_effect=FakeOperationalError(message))
                with self.assertRaises(FakeOperationalError):
                    retry(operation)(owner)
                operation.assert_called_once_with(owner)
                sleep.assert_not_called()

    def test_persistent_lock_has_bounded_attempts(self) -> None:
        retry, owner, sleep = _load_retry("sqlite")
        operation = Mock(side_effect=FakeOperationalError("database is locked"))
        with self.assertRaises(FakeOperationalError):
            retry(operation)(owner)
        self.assertEqual(operation.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list], [0.1, 0.2, 0.4]
        )
