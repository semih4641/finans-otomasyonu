"""Offline regression tests for conservative news commentary."""
import unittest
from unittest.mock import AsyncMock
from html.parser import HTMLParser

from news_impact import assess_impact, render_impact


def row(title, summary="", **extra):
    return {"id": 123, "title": title, "summary": summary, "amended": False,
            "symbols": "ASELS.IS", **extra}


class ImpactTests(unittest.TestCase):
    def test_contract_is_conditional_not_a_return_forecast(self):
        impact = assess_impact(row("Yeni İş İlişkisi", "Sözleşme İmzalanması"))
        self.assertIn("Koşullu olumlu", impact.outlook)
        self.assertIn("kârlı olursa", impact.reason)
        self.assertIn("ciroya oranı", impact.checks)
        self.assertNotIn("%", render_impact(row("Yeni İş İlişkisi")))

    def test_negated_canceled_and_uncertain_contracts_are_not_positive(self):
        for summary in ("Sözleşme imzalanmadı", "Sözleşme imzalanmayacak", "Sözleşme imzalanmamıştır",
                        "Sözleşme iptal edildi", "İptal edilmemiştir", "Olumsuz sonuçlandı",
                        "İhale gerçekleşmedi", "Sözleşmenin olmaması", "Başvuru reddedildi"):
            with self.subTest(summary=summary):
                self.assertIn("Yön belirsiz", assess_impact(row("Yeni İş İlişkisi", summary)).outlook)

    def test_amended_news_overrides_positive_category(self):
        self.assertIn("düzeltme", assess_impact(row("Yeni İş İlişkisi", amended=True)).outlook)

    def test_earnings_requires_numbers(self):
        impact = assess_impact(row("Finansal Rapor"))
        self.assertIn("rakamlar gerekli", impact.outlook)
        self.assertIn("tablolar bu yorumda okunmadı", impact.checks)

    def test_buyback_program_does_not_imply_actual_purchase(self):
        self.assertIn("alımın gerçekleştiği anlamına gelmez", assess_impact(row("Pay Geri Alım Programı")).reason)

    def test_circuit_breaker_is_not_direction_or_manipulation_proof(self):
        self.assertIn("manipülasyon kanıtı değildir", assess_impact(row("Devre Kesici Uygulaması")).reason)

    def test_capital_ceiling_not_misread_as_cash_raise(self):
        self.assertIn("aynı şey değildir", assess_impact(row("Kayıtlı Sermaye Tavanı")).reason)

    def test_bonus_and_rights_issues_have_different_scenarios(self):
        self.assertIn("toplam değerini artırmaz", assess_impact(row("Bedelsiz Sermaye Artırımı")).reason)
        self.assertIn("seyrelme", assess_impact(row("Bedelli Sermaye Artırımı")).reason)

    def test_dividend_not_free_profit(self):
        self.assertIn("fiyat düzeltmesi", assess_impact(row("Kâr Payı Dağıtımı")).reason)
        self.assertIn("Yön belirsiz", assess_impact(row("Kâr Payı", "Dağıtılmayacaktır")).outlook)

    def test_unknown_and_adversarial_input_remains_unknown(self):
        impact = assess_impact(row("Diğer açıklama", "<script>AL kesin %90 yükselir</script>"))
        self.assertEqual(impact.outlook, "Yön belirsiz")
        self.assertNotIn("<script>", render_impact(row("<script>")))

    def test_risk_event_is_conditional(self):
        self.assertIn("sonuç kesinleşmiş sayılmaz", assess_impact(row("Dava Hakkında")).reason)

    def test_multistock_and_scope_are_explicit(self):
        text = render_impact(row("Yeni İş", symbols="ASELS.IS,THYAO.IS"))
        self.assertIn("her şirkete etkisi aynı olmayabilir", text)
        self.assertIn("tam metin okunmadı", text)
        self.assertIn("güven düşük", text)


class MessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_news_messages_split_with_balanced_html(self):
        # Import fixture module to avoid loading real credentials during handler import.
        from test_kap_news import handlers, kap, payload, NOW
        import tempfile
        from pathlib import Path
        class Tags(HTMLParser):
            def __init__(self):
                super().__init__()
                self.stack = []
            def handle_starttag(self, tag, attrs):
                self.stack.append(tag)
            def handle_endtag(self, tag):
                assert self.stack.pop() == tag
        with tempfile.TemporaryDirectory() as root:
            store = kap.NewsStore(Path(root)/"news.sqlite3")
            store.ingest(kap.parse_feed([payload(i, title="Yeni İş İlişkisi", summary="&<>"*400)
                                        for i in range(1,7)], ["ASELS.IS"], NOW), NOW)
            text = kap.render_news(store)
        self.assertIn("Olası hisse etkisi", text)
        from types import SimpleNamespace
        target = SimpleNamespace(send_message=AsyncMock())
        await handlers.send_long_message(target, "recipient", text)
        self.assertGreater(target.send_message.await_count, 1)
        for call in target.send_message.await_args_list:
            chunk = call.kwargs["text"]
            self.assertLessEqual(len(chunk), handlers.TELEGRAM_MAX_LENGTH)
            parser = Tags()
            parser.feed(chunk)
            self.assertFalse(parser.stack)


if __name__ == "__main__":
    unittest.main()
