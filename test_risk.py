import unittest
from dataclasses import replace
from datetime import date
from unittest.mock import patch

# Unit tests must not read the developer's .env or account configuration.
with patch("dotenv.load_dotenv"), patch.dict("os.environ", {}, clear=True):
    from risk import (
        RiskConfig,
        calculate_position_risk,
        count_open_positions,
        daily_loss_gate,
        daily_realized_net_pnl,
        load_risk_config,
        open_position_gate,
    )


class RiskEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = RiskConfig(
            account_size=10_000,
            position_risk_pct=1.0,
            max_open_positions=2,
            daily_loss_limit_pct=2.0,
        )

    def test_position_size_uses_entry_stop_distance(self):
        result = calculate_position_risk(110, 100, self.config)
        self.assertEqual(result.status, "OK")
        self.assertAlmostEqual(result.risk_amount, 100)
        self.assertAlmostEqual(result.quantity, 10)

    def test_missing_account_does_not_assume_live_size(self):
        result = calculate_position_risk(110, 100, RiskConfig())
        self.assertEqual(result.status, "ACCOUNT_SIZE_UNSET")
        self.assertIsNone(result.quantity)

    def test_open_positions_and_daily_pnl_helpers(self):
        positions = [{"status": "open"}, {"status": "closed"}, {"is_open": True}]
        self.assertEqual(count_open_positions(positions), 2)
        records = [
            {"closed_at": "2026-09-11T10:00:00", "realized_net_pnl": -100},
            {"closed_at": "2026-09-11T11:00:00", "net_pnl_amount": 25},
            {"closed_at": "2026-09-10T11:00:00", "realized_net_pnl": -999},
        ]
        self.assertEqual(
            daily_realized_net_pnl(records, date(2026, 9, 11)),
            -75,
        )

    def test_limits_report_explicit_block(self):
        self.assertFalse(open_position_gate(2, self.config).allowed)
        decision = daily_loss_gate(-200, self.config)
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.kill_switch)
        self.assertEqual(decision.code, "DAILY_LOSS_LIMIT")

    def test_invalid_environment_is_not_silent(self):
        config = load_risk_config(
            {
                "ACCOUNT_SIZE": "not-a-number",
                "POSITION_RISK_PCT": "0",
                "MAX_OPEN_POSITIONS": "0",
                "DAILY_LOSS_LIMIT_PCT": "2",
            }
        )
        self.assertFalse(config.is_valid)
        self.assertEqual(config.account_size, None)

    def test_invalid_open_count_fails_closed(self):
        for count in (float("nan"), float("inf"), -1, 0.5, True, "1", None):
            with self.subTest(count=count):
                decision = open_position_gate(count, self.config)
                self.assertFalse(decision.allowed)
                self.assertTrue(decision.kill_switch)
                self.assertEqual(decision.code, "INVALID_OPEN_COUNT")

    def test_direct_invalid_config_is_rejected_by_all_calculations(self):
        invalid_fields = (
            {"account_size": float("nan")},
            {"account_size": -100},
            {"account_size": "10000"},
            {"position_risk_pct": -1},
            {"position_risk_pct": 101},
            {"position_risk_pct": True},
            {"daily_loss_limit_pct": float("inf")},
            {"daily_loss_limit_pct": 0},
            {"max_open_positions": float("nan")},
            {"max_open_positions": 1.5},
            {"max_open_positions": True},
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                config = replace(self.config, **fields)
                self.assertFalse(config.is_valid)
                self.assertEqual(calculate_position_risk(110, 100, config).status, "CONFIG_INVALID")
                self.assertEqual(open_position_gate(0, config).code, "CONFIG_INVALID")
                self.assertEqual(daily_loss_gate(0, config).code, "CONFIG_INVALID")

    def test_position_size_never_reports_infinite_quantity(self):
        config = replace(self.config, account_size=1e308)
        result = calculate_position_risk(2e-300, 1e-300, config)
        self.assertEqual(result.status, "INVALID_INPUT")
        self.assertIsNone(result.quantity)
        self.assertIsNone(result.notional_value)

    def test_large_finite_account_does_not_overflow_intermediate_budget(self):
        config = replace(self.config, account_size=1e308, position_risk_pct=2)
        result = calculate_position_risk(100, 50, config)
        self.assertEqual(result.status, "OK")
        self.assertAlmostEqual(result.risk_amount / 1e306, 2)

    def test_unrepresentable_numeric_input_fails_closed(self):
        huge_integer = 10 ** 1000
        self.assertEqual(calculate_position_risk(huge_integer, 100, self.config).status, "INVALID_INPUT")
        self.assertEqual(daily_loss_gate(huge_integer, self.config).code, "INVALID_DAILY_PNL")

    def test_daily_loss_gate_large_account_preserves_finite_limit(self):
        config = replace(self.config, account_size=1e308, daily_loss_limit_pct=2)
        decision = daily_loss_gate(-3e306, config)
        self.assertEqual(decision.code, "DAILY_LOSS_LIMIT")


if __name__ == "__main__":
    unittest.main()
