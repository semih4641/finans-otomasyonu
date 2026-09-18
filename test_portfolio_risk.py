import unittest
from dataclasses import replace
from unittest.mock import patch

import pandas as pd

with patch("dotenv.load_dotenv"), patch.dict("os.environ", {}, clear=True):
    import portfolio_risk as pr
    from risk import RiskConfig


def state(positions=None, equity=10_000, price_data=None):
    return pr.PortfolioState(
        positions=positions or [], equity=equity, peak_equity=10_000,
        daily_pnl=0, account_size=10_000, price_data=price_data or {},
    )


def position(symbol="stock:AAA", quantity=10, **changes):
    result = {"symbol": symbol, "entry": 100, "sl": 90, "quantity": quantity}
    result.update(changes)
    return result


class PortfolioRiskTests(unittest.TestCase):
    def setUp(self):
        self.config = pr.PortfolioConfig(sector_mapping={"AAA": "TECH", "BBB": "TECH"}, correlation_lookback_days=5)
        self.risk = RiskConfig(account_size=10_000)

    def test_invalid_environment_does_not_silently_enable_defaults(self):
        for env in ({"MAX_SECTOR_EXPOSURE_PCT": "nan"},
                    {"MAX_PORTFOLIO_RISK_PCT": "oops"},
                    {"CORRELATION_LOOKBACK_DAYS": "2"},
                    {"MIN_CORRELATION_THRESHOLD": "1.1"}):
            with self.subTest(env=env):
                config = pr.load_portfolio_config(env)
                self.assertFalse(config.is_valid)
                decision = pr.portfolio_gate(position(), state(), self.risk, config)
                self.assertFalse(decision.allowed)
                self.assertTrue(decision.kill_switch)
        self.assertTrue(pr.load_portfolio_config({"MIN_CORRELATION_THRESHOLD": "0"}).is_valid)

    def test_direct_invalid_configuration_is_rejected(self):
        for changes in ({"max_portfolio_risk_pct": True}, {"max_sector_exposure_pct": "30"},
                        {"max_drawdown_pct": float("inf")}, {"min_correlation_threshold": float("nan")},
                        {"min_correlation_threshold": True}, {"correlation_lookback_days": 2.5}):
            with self.subTest(changes=changes):
                self.assertFalse(replace(self.config, **changes).is_valid)

    def test_sector_exposure_uses_equity_including_cash(self):
        decision = pr.sector_exposure_gate(position(), state(), self.config)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.details["current_pct"], 10)
        decision = pr.sector_exposure_gate(position("stock:BBB", 25), state([position()]), self.config)
        self.assertEqual(decision.code, "SECTOR_LIMIT_EXCEEDED")
        self.assertEqual(decision.details["current_pct"], 35)

    def test_correlated_exposure_includes_proposed_size_and_uses_equity(self):
        prices = pd.DataFrame({"close": [100, 110, 99, 118.8, 112.86]},
                              index=pd.date_range("2026-01-01", periods=5))
        portfolio = state([position(quantity=15)], price_data={"AAA": prices, "BBB": prices * 2})
        decision = pr.correlation_exposure_gate(position("stock:BBB", 20), portfolio, self.config)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.details["correlated_pct"], 35)
        strict = replace(self.config, max_correlation_exposure_pct=30)
        decision = pr.correlation_exposure_gate(position("stock:BBB", 20), portfolio, strict)
        self.assertEqual(decision.code, "CORRELATION_LIMIT_EXCEEDED")

    def test_same_symbol_exposure_is_checked_without_price_history(self):
        decision = pr.correlation_exposure_gate(position(quantity=30), state([position(quantity=15)]), self.config)
        self.assertEqual(decision.code, "CORRELATION_LIMIT_EXCEEDED")
        self.assertEqual(decision.details["correlated_pct"], 45)

    def test_invalid_existing_quantity_is_rejected_without_correlation_data(self):
        for quantity in (None, float("nan"), -1, True):
            with self.subTest(quantity=quantity):
                decision = pr.correlation_exposure_gate(position("stock:BBB"), state([position(quantity=quantity)]), self.config)
                self.assertEqual(decision.code, "INVALID_EXISTING_POSITION")
                self.assertTrue(decision.kill_switch)

    def test_stop_risk_uses_current_equity_after_losses(self):
        portfolio = state([position(quantity=50)], equity=9_000)
        decision = pr.portfolio_risk_gate(position(), portfolio, self.risk, self.config)
        self.assertEqual(decision.code, "PORTFOLIO_RISK_EXCEEDED")
        self.assertAlmostEqual(decision.details["total_risk_pct"], 600 / 9_000 * 100)

    def test_invalid_candidate_stop_and_quantity_fail_closed(self):
        for changes in ({"sl": 0}, {"sl": 100}, {"sl": 110}, {"sl": float("nan")},
                        {"quantity": None}, {"quantity": True}, {"quantity": -1}):
            with self.subTest(changes=changes):
                decision = pr.portfolio_risk_gate(position(**changes), state(), self.risk, self.config)
                self.assertEqual(decision.code, "INVALID_POSITION")
                self.assertFalse(decision.allowed)

    def test_invalid_existing_stop_cannot_be_counted_as_zero_risk(self):
        decision = pr.portfolio_risk_gate(position("stock:BBB"), state([position(sl=None)]), self.risk, self.config)
        self.assertEqual(decision.code, "INVALID_EXISTING_POSITION")

    def test_same_batch_candidate_entry_alias_counts_toward_risk(self):
        candidate = position(quantity=60)
        candidate["entry_num"] = candidate.pop("entry")
        decision = pr.portfolio_risk_gate(position("stock:BBB"), state([candidate]), self.risk, self.config)
        self.assertEqual(decision.code, "PORTFOLIO_RISK_EXCEEDED")
        self.assertEqual(decision.details["current_risk"], 600)

    def test_existing_trailing_stop_cannot_offset_another_positions_risk(self):
        portfolio = state([position(sl=110, quantity=100), position("stock:BBB", quantity=60)])
        decision = pr.portfolio_risk_gate(position("stock:CCC"), portfolio, self.risk, self.config)
        self.assertEqual(decision.code, "PORTFOLIO_RISK_EXCEEDED")
        self.assertEqual(decision.details["current_risk"], 600)

    def test_invalid_equity_fails_all_exposure_checks(self):
        for equity in (None, 0, -1, float("nan"), float("inf"), True):
            with self.subTest(equity=equity):
                portfolio = state(equity=equity)
                self.assertEqual(pr.sector_exposure_gate(position(), portfolio, self.config).code, "INVALID_EQUITY")
                self.assertEqual(pr.correlation_exposure_gate(position(), portfolio, self.config).code, "INVALID_EQUITY")
                self.assertEqual(pr.portfolio_risk_gate(position(), portfolio, self.risk, self.config).code, "INVALID_EQUITY")

    def test_drawdown_limit_is_inclusive_and_stops_later_gates(self):
        with patch.object(pr, "portfolio_risk_gate", side_effect=AssertionError("must short-circuit")):
            decision = pr.portfolio_gate(position(), state(equity=9_000), self.risk, self.config)
        self.assertEqual(decision.code, "MAX_DRAWDOWN_EXCEEDED")
        self.assertTrue(decision.kill_switch)


if __name__ == "__main__":
    unittest.main()
