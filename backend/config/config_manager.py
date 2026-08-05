import json
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "content_config.json"
LOG_PATH = Path(__file__).parent.parent / "logs" / "config_changes.log"


class ConfigManager:
    """Loads content_config.json and detects changes so callers can hot-reload
    without restarting the server or frontend. Also logs every change with a
    timestamp, so config states can be correlated with observed participant
    reactions during Phase 2 analysis.
    """

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self._mtime = 0.0
        self.config: dict = {}
        self.reload()

    def reload(self) -> dict:
        self.config = json.loads(self.path.read_text())
        self._mtime = self.path.stat().st_mtime
        return self.config

    def has_changed(self) -> bool:
        return self.path.stat().st_mtime != self._mtime

    def save(self, new_config: dict, note: str = "") -> dict:
        self.path.write_text(json.dumps(new_config, indent=2))
        self.config = new_config
        self._mtime = self.path.stat().st_mtime
        self._log_change(note)
        return self.config

    def _log_change(self, note: str) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": time.time(), "note": note, "config": self.config}
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
