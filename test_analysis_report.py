"""User-facing report regressions without network access or saved models."""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import analysis_report
    import telegram_handlers
    import bot


def data(count=220):
    frame = pd.DataFrame({"Open": [100.] * count, "High": [102.] * count,
                          "Low": [98.] * count, "Close": [101.] * count,
                          "Volume": [1000.] * count},
                         index=pd.date_range("2025-01-01", periods=count, tz="Europe/Istanbul"))
    return {"ticker": "TEST.IS", "info": {"shortName": "Test & <Firma>"},
            "price": 999, "history_df": frame}


class AnalysisReportTests(unittest.TestCase):
    def render(self, fixture, result=None, now="2026-01-01T12:00:00+03:00"):
        with patch.object(analysis_report, "detect_buy_signals", return_value=result or {
                "signals": [], "decision_reason": "Teknik koşulların toplam puanı 1; en az 2 gerekiyor."
        }) as detect:
            report = analysis_report.build_stock_report(fixture, now)
        return report, detect

    def test_report_explains_no_signal_and_uses_closing_price_not_live_quote(self):
        report, detect = self.render(data())
        self.assertIn("101.00 TL", report)
        self.assertNotIn("999", report)
        self.assertIn("en az 2 gerekiyor", report)
        self.assertNotIn("Senaryo hedefleri", report)
        self.assertIn("200 kapanışın", report)
        self.assertTrue(detect.call_args.args[0].attrs["completed_only"])
        self.assertLess(len(report), 4000)

    def test_current_incomplete_bar_is_excluded_and_date_is_visible(self):
        report, detect = self.render(data(40), now="2025-02-09T12:00:00+03:00")
        self.assertEqual(len(detect.call_args.args[0]), 39)
        self.assertIn("Mum tarihi:</b> 08.02.2025", report)
        self.assertIn("1 takvim günü", report)

    def test_invalid_prices_suppress_analysis(self):
        for invalid in (float("nan"), -1, 105):
            fixture = data()
            fixture["history_df"].iloc[-1, 2] = invalid
            report, detect = self.render(fixture)
            self.assertIn("Veri kontrolü geçilemedi", report)
            detect.assert_not_called()

    def test_insufficient_history_does_not_invent_indicators(self):
        report, detect = self.render(data(12))
        self.assertIn("en az 30 mum gerekli", report)
        detect.assert_not_called()

    def test_names_and_reasons_are_html_escaped(self):
        report, _ = self.render(data(), {"signals": [], "decision_reason": "puan < 2"})
        self.assertIn("Test &amp; &lt;Firma&gt;", report)
        self.assertIn("puan &lt; 2", report)

    def test_accepted_setup_has_reasons_and_conditional_levels(self):
        report, _ = self.render(data(), {"signals": ["MACD kesişimi"], "sl": 97, "tp1": 108, "tp2": 115})
        self.assertIn("MACD kesişimi", report)
        self.assertIn("97.00 TL", report)
        self.assertIn("108.00 / 115.00", report)
        self.assertIn("gerçekleşme garantisi taşımaz", report)

    def test_hisse_command_routes_real_bist_response_to_explanatory_report(self):
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_message=message, effective_chat=SimpleNamespace(id="offline"))
        with patch.object(bot, "CHAT_ID", "offline"), \
             patch.object(telegram_handlers, "fetch_stock_data_async", new=AsyncMock(return_value=data())), \
             patch.object(analysis_report, "detect_buy_signals", return_value={"signals": []}):
            asyncio.run(telegram_handlers.hisse_command(update, SimpleNamespace(args=["TEST.IS"])))
        report = next(call.args[0] for call in message.reply_text.await_args_list if "Analizin dayandığı kapanış" in call.args[0])
        self.assertIn("Analizin dayandığı kapanış", report)
        self.assertIn("Takip edilecekler", report)


if __name__ == "__main__":
    unittest.main()
