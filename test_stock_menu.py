"""Offline tests for stock menu navigation and actual report delivery."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import bot
    import telegram_handlers as handlers


class StockMenuTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = patch.object(bot, "CHAT_ID", "owner")
        self.chat.start()
        self.addCleanup(self.chat.stop)
        self.message = SimpleNamespace(reply_text=AsyncMock())
        self.query = SimpleNamespace(data="stockinfo:THYAO.IS", answer=AsyncMock(), edit_message_text=AsyncMock())
        self.update = SimpleNamespace(effective_chat=SimpleNamespace(id="owner"),
                                      effective_message=self.message, callback_query=self.query)
        self.context = SimpleNamespace(args=[])

    def test_all_configured_stocks_are_reachable_once(self):
        expected = handlers.stock_menu_symbols()
        found = []
        pages = max(1, (len(expected) + 11) // 12)
        for page in range(pages):
            markup = handlers.stock_menu_markup(page)
            symbols = [button.callback_data.split(":", 1)[1] for row in markup.inline_keyboard
                       for button in row if button.callback_data.startswith("stockinfo:")]
            self.assertLessEqual(len(symbols), 12)
            found.extend(symbols)
        self.assertEqual(found, expected)
        self.assertIn("THYAO.IS", found)

    async def test_start_and_empty_hisse_show_stock_buttons(self):
        for handler in (handlers.start_command, handlers.hisse_command, handlers.hisseler_command):
            with self.subTest(handler=handler.__name__):
                await handler(self.update, self.context)
                self.assertTrue(self.message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard)

    async def test_navigation_does_not_fetch_prices(self):
        self.query.data = "stockmenu:1"
        with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock()) as fetch:
            await handlers.stock_menu_callback(self.update, self.context)
        fetch.assert_not_awaited()
        self.query.answer.assert_awaited_once()
        self.assertIn("2/", self.query.edit_message_text.await_args.args[0])

    async def test_selection_delivers_report_and_return_button(self):
        fixture = {"ticker": "THYAO.IS"}
        with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock(return_value=fixture)) as fetch, \
             patch("analysis_report.build_stock_report", return_value="Açıklamalı hisse raporu") as report:
            await handlers.stock_menu_callback(self.update, self.context)
        fetch.assert_awaited_once_with("THYAO.IS")
        report.assert_called_once_with(fixture)
        self.assertTrue(any(call.args[0] == "Açıklamalı hisse raporu" for call in self.message.reply_text.await_args_list))
        button = self.message.reply_text.await_args.kwargs["reply_markup"].inline_keyboard[0][0]
        self.assertEqual(button.callback_data, "stockmenu:0")
        self.assertEqual(self.context.args, [])

    async def test_unknown_stock_is_rejected_without_fetch(self):
        self.query.data = "stockinfo:NOTINLIST.IS"
        with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock()) as fetch:
            await handlers.stock_menu_callback(self.update, self.context)
        fetch.assert_not_awaited()
        self.assertTrue(self.query.answer.await_args.kwargs["show_alert"])

    async def test_unauthorized_click_is_answered_without_analysis(self):
        self.update.effective_chat.id = "other"
        with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock()) as fetch:
            await handlers.stock_menu_callback(self.update, self.context)
        fetch.assert_not_awaited()
        self.query.answer.assert_awaited_once()

    async def test_missing_market_data_shows_recovery_message(self):
        with patch.object(handlers, "fetch_stock_data_async", new=AsyncMock(return_value=None)):
            await handlers.stock_menu_callback(self.update, self.context)
        self.assertIn("verisi çekilemedi", self.message.reply_text.await_args.args[0])


if __name__ == "__main__":
    unittest.main()
