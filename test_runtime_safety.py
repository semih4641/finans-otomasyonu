"""Offline regressions for authorization, safe logging, and BIST commands."""

import logging
import os
import sys
import unittest
from html import escape
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True), \
        patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
    import bot
    import telegram_handlers as handlers


class RuntimeCommandSafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.network = patch("socket.socket.connect", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.chat = patch.object(bot, "CHAT_ID", "12345")
        self.chat.start()
        self.addCleanup(self.chat.stop)
        self.message = SimpleNamespace(reply_text=AsyncMock())
        self.query = SimpleNamespace(answer=AsyncMock())
        self.update = SimpleNamespace(
            effective_chat=SimpleNamespace(id=12345),
            effective_message=self.message,
            callback_query=None,
        )
        self.context = SimpleNamespace(args=[])
        self.action = AsyncMock(return_value="completed")

        async def protected(update, context):
            return await self.action(update, context)

        self.protected = handlers.authorized(protected)

    async def test_missing_or_blank_chat_id_blocks_commands_with_setup_guidance(self):
        for configured in (None, "", "   ", "\t\n"):
            with self.subTest(configured=configured), patch.object(bot, "CHAT_ID", configured):
                self.message.reply_text.reset_mock()
                result = await self.protected(self.update, self.context)
                self.assertIsNone(result)
                self.action.assert_not_awaited()
                self.message.reply_text.assert_awaited_once()
                guidance = self.message.reply_text.await_args.args[0]
                self.assertIn("/chatid", guidance)
                self.assertIn("CHAT_ID", guidance)

    async def test_missing_or_blank_chat_id_answers_callbacks_without_running_command(self):
        self.update.callback_query = self.query
        for configured in (None, "", "   "):
            with self.subTest(configured=configured), patch.object(bot, "CHAT_ID", configured):
                self.query.answer.reset_mock()
                await self.protected(self.update, self.context)
                self.action.assert_not_awaited()
                self.message.reply_text.assert_not_awaited()
                self.query.answer.assert_awaited_once()
                self.assertIn("/chatid", self.query.answer.await_args.args[0])
                self.assertTrue(self.query.answer.await_args.kwargs["show_alert"])

    async def test_configured_chat_accepts_matching_id_and_preserves_result(self):
        for configured in ("12345", " 12345 ", 12345):
            with self.subTest(configured=configured), patch.object(bot, "CHAT_ID", configured):
                self.action.reset_mock()
                result = await self.protected(self.update, self.context)
                self.assertEqual(result, "completed")
                self.action.assert_awaited_once_with(self.update, self.context)
                self.message.reply_text.assert_not_awaited()

    async def test_configured_chat_denies_other_or_missing_chat(self):
        for chat in (SimpleNamespace(id=99999), None):
            with self.subTest(chat=chat):
                self.update.effective_chat = chat
                with self.assertLogs(handlers.logger, level="WARNING"):
                    result = await self.protected(self.update, self.context)
                self.assertIsNone(result)
                self.action.assert_not_awaited()
                self.message.reply_text.assert_not_awaited()

    async def test_denied_callback_receives_alert_without_command_execution(self):
        self.update.effective_chat.id = 99999
        self.update.callback_query = self.query
        with self.assertLogs(handlers.logger, level="WARNING"):
            await self.protected(self.update, self.context)
        self.action.assert_not_awaited()
        self.query.answer.assert_awaited_once()
        self.assertTrue(self.query.answer.await_args.kwargs["show_alert"])
        self.message.reply_text.assert_not_awaited()

    async def test_chatid_bootstrap_remains_available_without_configuration_or_owner_match(self):
        for configured in (None, "", "99999"):
            with self.subTest(configured=configured), patch.object(bot, "CHAT_ID", configured):
                self.message.reply_text.reset_mock()
                await handlers.chatid_command(self.update, self.context)
                self.message.reply_text.assert_awaited_once()
                response = self.message.reply_text.await_args.args[0]
                self.assertIn("<code>12345</code>", response)
                self.assertIn("CHAT_ID", response)

    async def test_sinyal_bist_uses_explanatory_report_and_never_direct_detector(self):
        for requested, returned in (("asels", "ASELS.IS"), ("ASELS.IS", "asels.is")):
            with self.subTest(requested=requested, returned=returned):
                self.message.reply_text.reset_mock()
                self.context.args = [requested]
                fixture = {"ticker": returned}
                with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock(return_value=fixture)) as fetch, \
                        patch("analysis_report.build_stock_report", return_value="<b>BIST report</b>") as report, \
                        patch.object(handlers, "detect_buy_signals", side_effect=AssertionError("Direct detector used")) as direct:
                    await handlers.sinyal_command(self.update, self.context)
                fetch.assert_awaited_once_with(requested.upper())
                report.assert_called_once_with(fixture)
                direct.assert_not_called()
                self.assertEqual(self.message.reply_text.await_count, 2)
                self.message.reply_text.assert_awaited_with("<b>BIST report</b>", parse_mode="HTML")

    async def test_sinyal_bist_falls_back_to_requested_ticker_when_response_omits_it(self):
        self.context.args = ["ASELS.IS"]
        fixture = {}
        with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock(return_value=fixture)), \
                patch("analysis_report.build_stock_report", return_value="BIST report") as report, \
                patch.object(handlers, "detect_buy_signals", side_effect=AssertionError("Direct detector used")) as direct:
            await handlers.sinyal_command(self.update, self.context)
        report.assert_called_once_with({"ticker": "ASELS.IS"})
        direct.assert_not_called()
        self.message.reply_text.assert_awaited_with("BIST report", parse_mode="HTML")

    async def test_sinyal_without_configuration_never_fetches_market_data(self):
        self.context.args = ["ASELS"]
        with patch.object(bot, "CHAT_ID", None), \
                patch.object(handlers, "fetch_stock_data_async", new=AsyncMock()) as fetch:
            await handlers.sinyal_command(self.update, self.context)
        fetch.assert_not_awaited()
        self.assertIn("/chatid", self.message.reply_text.await_args.args[0])

    async def test_sinyal_escapes_user_ticker_in_progress_and_failure_messages(self):
        ticker = "<BAD&>"
        self.context.args = [ticker]
        with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock(return_value=None)) as fetch, \
                patch("analysis_report.build_stock_report") as report, \
                patch.object(handlers, "detect_buy_signals") as direct:
            await handlers.sinyal_command(self.update, self.context)
        fetch.assert_awaited_once_with(ticker)
        report.assert_not_called()
        direct.assert_not_called()
        self.assertEqual(self.message.reply_text.await_count, 2)
        for call in self.message.reply_text.await_args_list:
            self.assertIn(escape(ticker), call.args[0])
            self.assertNotIn(ticker, call.args[0])
            self.assertEqual(call.kwargs["parse_mode"], "HTML")


class TelegramRedactionTests(unittest.TestCase):
    def record(self, message, args=(), exc_info=None):
        return logging.LogRecord("httpx", logging.WARNING, __file__, 1, message, args, exc_info)

    def test_httpx_argument_urls_hide_token_but_keep_method_and_endpoint(self):
        token = "123456:offline-test-token"
        for base in ("https://api.telegram.org/bot", "HTTP://API.TELEGRAM.ORG/bot",
                     "https://api.telegram.org/file/bot"):
            with self.subTest(base=base):
                record = self.record('HTTP Request: %s %s "%s"',
                                     ("POST", f"{base}{token}/sendMessage?fixture=1", "HTTP/1.1 200 OK"))
                result = bot.TelegramRedactingFormatter("%(message)s").format(record)
                self.assertNotIn(token, result)
                self.assertIn(f"{base}[REDACTED]/sendMessage?fixture=1", result)
                self.assertIn("POST", result)
                self.assertIn("HTTP/1.1 200 OK", result)

    def test_exception_tracebacks_redact_telegram_url_tokens(self):
        token = "654321:offline-exception-token"
        try:
            raise RuntimeError(f"request failed: https://api.telegram.org/bot{token}/getUpdates")
        except RuntimeError:
            record = self.record("Polling failed", exc_info=sys.exc_info())
        result = bot.TelegramRedactingFormatter("%(message)s").format(record)
        self.assertNotIn(token, result)
        self.assertIn("Traceback (most recent call last)", result)
        self.assertIn("RuntimeError: request failed:", result)
        self.assertIn("https://api.telegram.org/bot[REDACTED]/getUpdates", result)

    def test_configured_raw_token_is_removed_from_message_arguments_and_exception(self):
        token = "configured-offline-token"
        try:
            raise ValueError(f"invalid credential {token}")
        except ValueError:
            record = self.record("Credential rejected: %s", (token,), sys.exc_info())
        result = bot.TelegramRedactingFormatter("%(message)s", token=token).format(record)
        self.assertNotIn(token, result)
        self.assertIn("Credential rejected: [REDACTED]", result)
        self.assertIn("ValueError: invalid credential [REDACTED]", result)

    def test_unrelated_urls_and_message_content_are_preserved(self):
        message = "Fixture https://example.test/bot/public/path returned 503"
        self.assertEqual(bot.TelegramRedactingFormatter("%(message)s").format(self.record(message)), message)


if __name__ == "__main__":
    unittest.main()
