"""
src/decision/store.py — JSON state persistence for the decision layer.

Manages:
  1. Recommendation history  — rolling 24h window; most recent HEDGE always kept
  2. Hedge state             — current open position
  3. Review state            — sticky REVIEW flag (only clears via manual API call)
  4. Kill switch             — emergency halt flag
  5. Pending hedge           — awaiting manual approval
  6. Approval log            — immutable audit trail of all approvals/rejections
  7. Review clearance log    — immutable audit trail of all REVIEW clears

State file
----------
Defaults to data/state/decision_state.json inside the repo. Override with the
DECISION_STATE_DIR environment variable to point at a directory outside the
repo (e.g. for a long-running deployment).

Writes are atomic: write to a temp file, then os.replace() to the target.
This prevents partial-write corruption on crash.

Thread safety: single-process writes only. If concurrency is needed in future,
replace _save_raw with a file-lock wrapper.

Isolated demo state vs. persistent operational state
------------------------------------------------------
DecisionStore has two distinct usage modes, distinguished by whether
`state_dir` is passed explicitly:

- **Persistent operational state** (state_dir=None, the default): resolves
  to DECISION_STATE_DIR or data/state/ as described above. This is what the
  FastAPI ops endpoints (/decision/state, /decision/kill-switch,
  /decision/review/clear in src/api/main.py) read and write — it is meant
  to survive across process restarts and accumulate a real audit trail.

- **Isolated demo/test state** (state_dir=<some Path>, dependency-injected
  by the caller): reads and writes ONLY inside that directory, bypassing
  DECISION_STATE_DIR and data/state/ entirely — it never touches the
  persistent location, not even to read it. `src/demo.py::main()` creates a
  fresh `tempfile.TemporaryDirectory()` for every run and passes it down
  explicitly for exactly this reason: the offline demo must be deterministic
  and unaffected by whatever operational state happens to exist locally, and
  running the demo must never mutate that operational state. See
  `docs/architecture.md` for the full rationale and
  tests/test_demo_e2e.py::test_demo_isolation_ignores_stale_persistent_state
  for the regression test that proves it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.decision.engine import DecisionOutput, DecisionRecord, HedgeState, ReviewState

_log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HISTORY_WINDOW_HOURS = 24


def _state_dir() -> Path:
    override = os.getenv("DECISION_STATE_DIR")
    return Path(override).expanduser() if override else _PROJECT_ROOT / "data" / "state"


def _state_file(asset: Optional[str] = None, state_dir: Optional[Path] = None) -> Path:
    """
    Per-asset state file.

    Each asset gets an independent history, hedge state, and anti-whipsaw
    window — sharing one file across assets would let BTC's anti-whipsaw
    suppress an ETH hedge (and vice versa) and would mix their decision
    histories together. asset=None keeps the legacy single-file layout for
    single-asset callers.

    state_dir, when provided, takes priority over DECISION_STATE_DIR and the
    data/state/ default — this is the isolation seam described in the module
    docstring above. It is never merged with or falls back to the persistent
    location; passing state_dir means "only this directory, full stop."
    """
    base = state_dir if state_dir is not None else _state_dir()
    name = f"decision_state_{asset}.json" if asset else "decision_state.json"
    return base / name


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _empty_state() -> dict:
    return {
        "version": 1,
        "history": [],
        "hedge_state": None,
        "review_state": {"active": False, "reason": "", "activated_at": None},
        "kill_switch": False,
        "pending_hedge": None,
        "approval_log": [],
        "review_clearance_log": [],
    }


def _load_raw(asset: Optional[str] = None, state_dir: Optional[Path] = None) -> dict:
    path = _state_file(asset, state_dir)
    if not path.exists():
        return _empty_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Forward-compat: add any missing top-level keys from the empty template
        template = _empty_state()
        for key, default in template.items():
            data.setdefault(key, default)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Could not load decision state from %s: %s — using empty state", path, exc)
        return _empty_state()


def _save_raw(state: dict, asset: Optional[str] = None, state_dir: Optional[Path] = None) -> None:
    path = _state_file(asset, state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".decision_state_tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        _log.error("Could not persist decision state to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Deserializers
# ---------------------------------------------------------------------------


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _parse_history(raw: dict) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    for item in raw.get("history", []):
        try:
            records.append(
                DecisionRecord(
                    timestamp=_parse_dt(item["timestamp"]) or datetime.now(timezone.utc),
                    action=item["action"],
                    policy_action=item.get("policy_action", item["action"]),
                    regime=item["regime"],
                    confidence=float(item["confidence"]),
                    asset=item["asset"],
                    rationale=item.get("rationale", ""),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return records


def _parse_hedge_state(raw: dict) -> HedgeState:
    h = raw.get("hedge_state")
    if not h:
        return HedgeState(
            is_hedged=False,
            instrument=None,
            strike=None,
            notional=None,
            premium_paid_usd=None,
            opened_at=None,
            hedge_ratio=0.0,
        )
    return HedgeState(
        is_hedged=bool(h.get("is_hedged", False)),
        instrument=h.get("instrument"),
        strike=h.get("strike"),
        notional=h.get("notional"),
        premium_paid_usd=h.get("premium_paid_usd"),
        opened_at=_parse_dt(h.get("opened_at")),
        hedge_ratio=float(h.get("hedge_ratio", 0.0)),
    )


def _parse_review_state(raw: dict) -> ReviewState:
    r = raw.get("review_state") or {}
    return ReviewState(
        active=bool(r.get("active", False)),
        reason=r.get("reason", ""),
        activated_at=_parse_dt(r.get("activated_at")),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class DecisionStore:
    """
    Lightweight JSON state persistence for the decision layer.

    Pattern: load once, mutate in memory, call save() at end of run.
    All public mutating methods update self._state in place.

    One store per asset: pass asset="BTC" / asset="ETH" so each symbol keeps
    its own history, hedge state, and anti-whipsaw window (see _state_file).

    Pass state_dir to isolate this store to a specific directory (e.g. a
    tempfile.TemporaryDirectory()) instead of the persistent operational
    location — see "Isolated demo state vs. persistent operational state" in
    the module docstring.
    """

    def __init__(self, asset: Optional[str] = None, state_dir: Optional[Path] = None) -> None:
        self.asset = asset
        self.state_dir = state_dir
        self._state = _load_raw(asset, state_dir)

    def save(self) -> None:
        """Atomically persist state to disk."""
        _save_raw(self._state, self.asset, self.state_dir)

    # -- Read -----------------------------------------------------------

    @property
    def kill_switch(self) -> bool:
        return bool(self._state.get("kill_switch", False))

    def load_history(self) -> list[DecisionRecord]:
        return _parse_history(self._state)

    def load_hedge_state(self) -> HedgeState:
        return _parse_hedge_state(self._state)

    def load_review_state(self) -> ReviewState:
        return _parse_review_state(self._state)

    def load_pending_hedge(self) -> Optional[dict]:
        return self._state.get("pending_hedge")

    def recent_actions_strings(self, n: int = 10) -> list[str]:
        """Return the last n decisions as human-readable strings for report context."""
        records = self.load_history()
        return [
            f"{r.timestamp.strftime('%Y-%m-%dT%H:%MZ')}  {r.action:<6}  "
            f"{r.regime}  conf={r.confidence:.0%}"
            for r in records[-n:]
        ]

    # -- Write ------------------------------------------------------------

    def append_decision(
        self,
        decision: DecisionOutput,
        regime: str,
        confidence: float,
        asset: str,
        report_id: str,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Add a decision to rolling history.

        Pruning rules:
          - Keep all entries within the last 24h window.
          - Always keep the most recent HEDGE entry regardless of age.
        """
        now = now or datetime.now(timezone.utc)
        new_entry = {
            "timestamp": now.isoformat(),
            "action": decision.action,
            "policy_action": decision.policy_action,
            "regime": regime,
            "confidence": confidence,
            "asset": asset,
            "rationale": decision.rationale,
            "report_id": report_id,
        }

        history = self._state.setdefault("history", [])
        history.append(new_entry)

        cutoff = (now - timedelta(hours=_HISTORY_WINDOW_HOURS)).isoformat()
        hedges = [r for r in history if r["action"] == "HEDGE"]
        last_hedge = max(hedges, key=lambda r: r["timestamp"]) if hedges else None

        pruned = [r for r in history if r["timestamp"] >= cutoff]
        if last_hedge and last_hedge not in pruned:
            pruned.append(last_hedge)

        self._state["history"] = pruned

    def activate_review_state(self, reason: str, now: Optional[datetime] = None) -> None:
        """
        Enter sticky REVIEW state. Idempotent — reason and timestamp are only
        set on the first activation; subsequent calls with the same flag active
        do not overwrite the original reason or timestamp.
        """
        now = now or datetime.now(timezone.utc)
        rs = self._state.setdefault("review_state", {})
        if not rs.get("active", False):
            rs["active"] = True
            rs["reason"] = reason
            rs["activated_at"] = now.isoformat()
            _log.info("[store] REVIEW state activated: %s", reason)

    def clear_review_state(
        self,
        cleared_by: str,
        note: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        """
        Manually clear the REVIEW state.

        Every clearance is appended to review_clearance_log — this log is
        append-only and must not be pruned (audit requirement).
        """
        now = now or datetime.now(timezone.utc)
        rs = self._state.setdefault("review_state", {})
        rs["active"] = False
        rs["reason"] = ""
        rs["activated_at"] = None

        log = self._state.setdefault("review_clearance_log", [])
        log.append(
            {
                "cleared_at": now.isoformat(),
                "cleared_by": cleared_by,
                "note": note,
            }
        )
        _log.info("[store] REVIEW state cleared by %s: %s", cleared_by, note)

    def set_kill_switch(
        self, active: bool, set_by: str = "api", now: Optional[datetime] = None
    ) -> None:
        now = now or datetime.now(timezone.utc)
        self._state["kill_switch"] = active
        _log.info("[store] Kill switch set to %s by %s at %s", active, set_by, now.isoformat())

    def update_hedge_state(self, hedge: HedgeState, now: Optional[datetime] = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._state["hedge_state"] = {
            "is_hedged": hedge.is_hedged,
            "instrument": hedge.instrument,
            "strike": hedge.strike,
            "notional": hedge.notional,
            "premium_paid_usd": hedge.premium_paid_usd,
            "opened_at": hedge.opened_at.isoformat() if hedge.opened_at else None,
            "hedge_ratio": hedge.hedge_ratio,
            "updated_at": now.isoformat(),
        }

    def set_pending_hedge(
        self, report_id: str, asset: str, details: dict, now: Optional[datetime] = None
    ) -> None:
        """Record a HEDGE recommendation as awaiting manual approval."""
        now = now or datetime.now(timezone.utc)
        self._state["pending_hedge"] = {
            "report_id": report_id,
            "asset": asset,
            "created_at": now.isoformat(),
            **details,
        }

    def clear_pending_hedge(self) -> None:
        self._state["pending_hedge"] = None

    def log_approval(
        self,
        report_id: str,
        approved: bool,
        approved_by: str,
        note: str = "",
        now: Optional[datetime] = None,
    ) -> None:
        """
        Record an approval or rejection of a pending HEDGE.

        Append-only — this log must not be pruned (audit requirement).
        """
        now = now or datetime.now(timezone.utc)
        log = self._state.setdefault("approval_log", [])
        log.append(
            {
                "report_id": report_id,
                "approved": approved,
                "approved_by": approved_by,
                "note": note,
                "timestamp": now.isoformat(),
            }
        )
        action = "APPROVED" if approved else "REJECTED"
        _log.info("[store] Hedge %s by %s (report=%s): %s", action, approved_by, report_id, note)

    # -- Snapshot for REST API --------------------------------------------

    def state_snapshot(self) -> dict:
        """Return the full current state as a plain dict (for GET /decision/state)."""
        return {
            "kill_switch": self._state.get("kill_switch", False),
            "review_state": self._state.get("review_state", {}),
            "pending_hedge": self._state.get("pending_hedge"),
            "history_count": len(self._state.get("history", [])),
            "recent_history": self._state.get("history", [])[-5:],
            "hedge_state": self._state.get("hedge_state"),
            "approval_log_count": len(self._state.get("approval_log", [])),
            "review_clearance_log_count": len(self._state.get("review_clearance_log", [])),
        }
