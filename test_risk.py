import unittest
from datetime import date

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


if __name__ == "__main__":
    unittest.main()
