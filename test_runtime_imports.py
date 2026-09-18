"""Offline smoke tests of real entry points, without replacing project modules."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class RuntimeImportTests(unittest.TestCase):
    def run_isolated(self, source):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=Path(__file__).parent,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True, text=True, encoding="utf-8", timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_actual_backtest_and_training_help_import_all_dependencies(self):
        output = self.run_isolated('''
            import os
            import runpy
            import sys
            from unittest.mock import patch

            with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True), \
                    patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
                for filename in ("backtest.py", "train_bist.py"):
                    with patch.object(sys, "argv", [filename, "--help"]):
                        try:
                            runpy.run_path(filename, run_name="__main__")
                        except SystemExit as exc:
                            assert exc.code == 0, exc.code
                        else:
                            raise AssertionError("CLI did not handle --help")
                print("real CLI imports passed")
        ''')
        self.assertIn("real CLI imports passed", output)

    def test_script_startup_and_handlers_share_one_bot_module(self):
        output = self.run_isolated('''
            import os
            import runpy
            import sys
            from unittest.mock import patch

            class StopBeforeNetwork(Exception):
                pass

            def inspect_builder():
                import bot
                import scheduler
                import telegram_handlers
                assert bot is sys.modules["__main__"]
                assert bot.main.__globals__ is bot.__dict__
                assert bot._update_portfolio_state.__globals__ is bot.__dict__
                raise StopBeforeNetwork()

            with patch("dotenv.load_dotenv"), \
                    patch.dict(os.environ, {"BOT_TOKEN": "offline-fixture"}, clear=True), \
                    patch("logging.basicConfig"), patch("logging.FileHandler"), \
                    patch("telegram.ext.Application.builder", side_effect=inspect_builder), \
                    patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
                try:
                    runpy.run_module("bot", run_name="__main__", alter_sys=True)
                except StopBeforeNetwork:
                    print("single runtime module passed")
                else:
                    raise AssertionError("Startup did not reach application builder")
        ''')
        self.assertIn("single runtime module passed", output)


if __name__ == "__main__":
    unittest.main()
