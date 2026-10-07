"""Additional analysis-only chats, stored outside the deployed source tree."""
import json
import logging
import os
from pathlib import Path

ACCESS_FILE = Path(os.getenv("BOT_DATA_DIR", ".")) / "authorized_chats.json"


def additional_chat_ids() -> set[str]:
    try:
        values = json.loads(ACCESS_FILE.read_text(encoding="utf-8"))
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not str(value).lstrip("-").isdigit()
            for value in values
        ):
            raise ValueError("Invalid chat allowlist")
        return {str(value) for value in values}
    except FileNotFoundError:
        return set()
    except (OSError, ValueError, TypeError):
        logging.getLogger(__name__).warning("Ek sohbet yetki listesi okunamadı; ek erişim kapalı.")
        return set()
