from __future__ import annotations

import json
from datetime import datetime

from src.config import CONTROL_FILE, PID_FILE, STATE_FILE
from src.utils.files import write_json


def request_stop() -> None:
    write_json(CONTROL_FILE, {"stop_requested": True, "updated_at": datetime.now().isoformat()})


def clear_stop() -> None:
    write_json(CONTROL_FILE, {"stop_requested": False, "updated_at": datetime.now().isoformat()})


def stop_requested() -> bool:
    if not CONTROL_FILE.exists():
        return False
    try:
        return bool(json.loads(CONTROL_FILE.read_text(encoding="utf-8")).get("stop_requested"))
    except json.JSONDecodeError:
        return False


def write_state(payload: dict, merge: bool = True) -> None:
    current = read_state() if merge else {}
    payload = {**current, **payload, "updated_at": datetime.now().isoformat(timespec="seconds")}
    write_json(STATE_FILE, payload)


def read_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_pid(pid: int) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="utf-8")


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def clear_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()
