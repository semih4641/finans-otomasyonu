import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import bot
    import access_control
    import telegram_handlers as handlers


class AdditionalAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_friend_can_use_analysis_but_not_owner_commands(self):
        action = AsyncMock()
        update = SimpleNamespace(effective_chat=SimpleNamespace(id=42),
                                 effective_message=SimpleNamespace(reply_text=AsyncMock()))
        with patch.object(bot, "CHAT_ID", "1"), \
             patch.object(handlers, "additional_chat_ids", return_value={"42"}):
            await handlers.authorized(action)(update, None)
            action.assert_awaited_once()
            action.reset_mock()
            await handlers.authorized(action, owner_only=True)(update, None)
            action.assert_not_awaited()
            for command in (handlers.mltrain_command, handlers.portfoy_command,
                            handlers.performans_command, handlers.grafik_command):
                await command(update, None)
            update.effective_message.reply_text.assert_not_awaited()

    async def test_owner_retains_access_and_missing_owner_still_denies(self):
        action = AsyncMock()
        update = SimpleNamespace(effective_chat=SimpleNamespace(id=1),
                                 effective_message=SimpleNamespace(reply_text=AsyncMock()))
        with patch.object(bot, "CHAT_ID", "1"):
            await handlers.authorized(action, owner_only=True)(update, None)
        action.assert_awaited_once()
        action.reset_mock()
        with patch.object(bot, "CHAT_ID", None), \
             patch.object(handlers, "additional_chat_ids", return_value={"1"}):
            await handlers.authorized(action)(update, None)
        action.assert_not_awaited()

    async def test_file_is_reloadable_and_malformed_file_denies_additional_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorized_chats.json"
            with patch.object(access_control, "ACCESS_FILE", path):
                self.assertEqual(access_control.additional_chat_ids(), set())
                path.write_text(json.dumps([42, "43"]), encoding="utf-8")
                self.assertEqual(access_control.additional_chat_ids(), {"42", "43"})
                for value in ("not json", '{}', '[true]', '["everyone"]'):
                    path.write_text(value, encoding="utf-8")
                    self.assertEqual(access_control.additional_chat_ids(), set())


if __name__ == "__main__":
    unittest.main()
