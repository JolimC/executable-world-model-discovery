"""Concurrency-safe cumulative paid-model reservation and usage ledger."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from world_model_search.errors import BudgetExhaustedError, PersistenceError
from world_model_search.model.policy import CASH_VERIFICATION_LEVELS, PricePolicy
from world_model_search.persistence.manifest import utc_now
from world_model_search.serialization import JsonObject, canonical_json, sha256_text

LEDGER_SCHEMA_VERSION = 1
DUAL_LEDGER_SCHEMA_VERSION = 2
CASH_CHECKPOINT_VERSION = "provider-cash-reconciliation-v1"


def _json_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PersistenceError(f"{location} is not an integer")
    return value


@dataclass(frozen=True, slots=True)
class LedgerBalance:
    opening_nano_usd: int
    actual_nano_usd: int
    uncertain_nano_usd: int
    active_reserved_nano_usd: int

    @property
    def committed_nano_usd(self) -> int:
        return (
            self.opening_nano_usd
            + self.actual_nano_usd
            + self.uncertain_nano_usd
            + self.active_reserved_nano_usd
        )

    def to_value(self) -> JsonObject:
        return {
            "opening_nano_usd": self.opening_nano_usd,
            "actual_nano_usd": self.actual_nano_usd,
            "uncertain_nano_usd": self.uncertain_nano_usd,
            "active_reserved_nano_usd": self.active_reserved_nano_usd,
            "committed_nano_usd": self.committed_nano_usd,
        }


@dataclass(frozen=True, slots=True)
class CashBudgetBalance:
    personal_lifetime_cap_nano_usd: int
    safety_buffer_nano_usd: int
    reconciled_cash_nano_usd: int
    covered_reservation_sequence: int
    covered_published_nano_usd: int
    unreconciled_actual_nano_usd: int
    uncertain_nano_usd: int
    active_reserved_nano_usd: int
    checkpoint: JsonObject

    @property
    def cash_upper_bound_nano_usd(self) -> int:
        return (
            self.reconciled_cash_nano_usd
            + self.unreconciled_actual_nano_usd
            + self.uncertain_nano_usd
            + self.active_reserved_nano_usd
            + self.safety_buffer_nano_usd
        )

    @property
    def remaining_authorizable_nano_usd(self) -> int:
        return max(0, self.personal_lifetime_cap_nano_usd - self.cash_upper_bound_nano_usd)

    @property
    def overage_nano_usd(self) -> int:
        return max(0, self.cash_upper_bound_nano_usd - self.personal_lifetime_cap_nano_usd)

    def to_value(self) -> JsonObject:
        return {
            "enforcement_basis": "reconciled-cash-plus-unreconciled-published-v1",
            "personal_lifetime_cap_nano_usd": self.personal_lifetime_cap_nano_usd,
            "safety_buffer_nano_usd": self.safety_buffer_nano_usd,
            "reconciled_cash_nano_usd": self.reconciled_cash_nano_usd,
            "covered_reservation_sequence": self.covered_reservation_sequence,
            "covered_published_nano_usd": self.covered_published_nano_usd,
            "unreconciled_actual_nano_usd": self.unreconciled_actual_nano_usd,
            "uncertain_nano_usd": self.uncertain_nano_usd,
            "active_reserved_nano_usd": self.active_reserved_nano_usd,
            "cash_upper_bound_nano_usd": self.cash_upper_bound_nano_usd,
            "remaining_authorizable_nano_usd": self.remaining_authorizable_nano_usd,
            "overage_nano_usd": self.overage_nano_usd,
            "checkpoint": self.checkpoint,
        }


class ProjectLedger:
    def __init__(self, path: Path, policy: PricePolicy) -> None:
        self.path = path
        self.policy = policy
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self._validate_existing_identity()
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._initialize()

    def __enter__(self) -> ProjectLedger:
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def _expected_schema(self) -> int:
        return (
            DUAL_LEDGER_SCHEMA_VERSION
            if self.policy.uses_reconciled_cash_budget
            else LEDGER_SCHEMA_VERSION
        )

    def _validate_existing_identity(self) -> None:
        metadata_exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        if metadata_exists is None:
            return
        metadata = {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute("SELECT key,value FROM metadata")
        }
        if (
            metadata.get("ledger_schema_version") != str(self._expected_schema())
            or metadata.get("policy_hash") != self.policy.content_hash
        ):
            self.connection.close()
            raise PersistenceError("project ledger schema or price-policy identity mismatch")

    def _initialize(self) -> None:
        expected_schema = self._expected_schema()
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reservation (
                    reservation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    policy_hash TEXT NOT NULL,
                    reserved_nano_usd INTEGER NOT NULL CHECK (reserved_nano_usd >= 0),
                    state TEXT NOT NULL CHECK (state IN ('active','reconciled','uncertain')),
                    actual_nano_usd INTEGER NOT NULL DEFAULT 0,
                    uncertain_nano_usd INTEGER NOT NULL DEFAULT 0,
                    released_nano_usd INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS usage_record (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    reservation_id TEXT NOT NULL UNIQUE REFERENCES reservation(reservation_id),
                    record_json TEXT NOT NULL,
                    record_hash TEXT NOT NULL
                );
                """
            )
            if self.policy.uses_reconciled_cash_budget:
                self.connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS cash_checkpoint (
                        checkpoint_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        checkpoint_id TEXT NOT NULL UNIQUE,
                        covered_reservation_sequence INTEGER NOT NULL
                            CHECK (covered_reservation_sequence >= 0),
                        cumulative_billed_nano_usd INTEGER NOT NULL
                            CHECK (cumulative_billed_nano_usd >= 0),
                        covered_published_nano_usd INTEGER NOT NULL
                            CHECK (covered_published_nano_usd >= 0),
                        observed_at TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        source TEXT NOT NULL,
                        verification TEXT NOT NULL,
                        decrease_authorized INTEGER NOT NULL CHECK (decrease_authorized IN (0,1)),
                        recorded_at TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        record_hash TEXT NOT NULL
                    );
                    """
                )
            existing = self.connection.execute(
                "SELECT value FROM metadata WHERE key='ledger_schema_version'"
            ).fetchone()
            if existing is None:
                self.connection.executemany(
                    "INSERT INTO metadata VALUES (?, ?)",
                    (
                        ("ledger_schema_version", str(expected_schema)),
                        ("policy_hash", self.policy.content_hash),
                        ("opening_nano_usd", str(self.policy.opening_balance_nano_usd)),
                    ),
                )
            else:
                policy = self.connection.execute(
                    "SELECT value FROM metadata WHERE key='policy_hash'"
                ).fetchone()
                if existing["value"] != str(expected_schema) or policy is None:
                    raise PersistenceError("unsupported or corrupt project ledger")
                if policy["value"] != self.policy.content_hash:
                    raise PersistenceError("project ledger price-policy hash mismatch")

    def _sum(self, expression: str, where: str = "1", parameters: tuple[object, ...] = ()) -> int:
        row = self.connection.execute(
            f"SELECT COALESCE(SUM({expression}),0) AS amount FROM reservation WHERE {where}",
            parameters,
        ).fetchone()
        return int(row["amount"] if row is not None else 0)

    def balance(self) -> LedgerBalance:
        return LedgerBalance(
            opening_nano_usd=self.policy.opening_balance_nano_usd,
            actual_nano_usd=self._sum("actual_nano_usd"),
            uncertain_nano_usd=self._sum("uncertain_nano_usd"),
            active_reserved_nano_usd=self._sum("reserved_nano_usd", "state='active'"),
        )

    def _latest_reservation_sequence(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(rowid),0) AS sequence FROM reservation"
        ).fetchone()
        return int(row["sequence"] if row is not None else 0)

    def _reconcilable_through_sequence(self) -> int:
        rows = self.connection.execute(
            "SELECT rowid AS reservation_sequence, state FROM reservation ORDER BY rowid"
        ).fetchall()
        covered = 0
        for row in rows:
            if row["state"] != "reconciled":
                break
            covered = int(row["reservation_sequence"])
        return covered

    def _opening_checkpoint(self) -> JsonObject:
        cash = self.policy.cash_budget
        if cash is None:
            raise PersistenceError("cash reconciliation requires a dual-budget policy")
        return {
            "checkpoint_sequence": 0,
            "checkpoint_id": "opening-policy-reconciliation",
            "covered_reservation_sequence": 0,
            "cumulative_billed_nano_usd": cash.opening_reconciled_cash_nano_usd,
            "covered_published_nano_usd": cash.opening_covered_published_nano_usd,
            "observed_at": cash.opening_observed_at,
            "scope": cash.opening_scope,
            "source": cash.opening_source,
            "verification": cash.opening_verification,
            "decrease_authorized": False,
            "recorded_at": None,
            "record_hash": self.policy.content_hash,
        }

    def _checkpoint_from_row(self, row: sqlite3.Row) -> JsonObject:
        record_text = str(row["record_json"])
        record_hash = str(row["record_hash"])
        if record_hash != sha256_text(record_text) or str(row["checkpoint_id"]) != record_hash:
            raise PersistenceError("cash checkpoint record hash mismatch")
        try:
            record = json.loads(record_text)
        except json.JSONDecodeError as exc:
            raise PersistenceError("cash checkpoint record is malformed") from exc
        expected_keys = {
            "checkpoint_version",
            "cumulative_billed_nano_usd",
            "covered_reservation_sequence",
            "covered_published_nano_usd",
            "observed_at",
            "scope",
            "source",
            "verification",
            "decrease_authorized",
            "policy_hash",
        }
        if (
            not isinstance(record, dict)
            or set(record) != expected_keys
            or canonical_json(record) != record_text
            or record.get("checkpoint_version") != CASH_CHECKPOINT_VERSION
            or record.get("policy_hash") != self.policy.content_hash
        ):
            raise PersistenceError("cash checkpoint canonical record is inconsistent")
        indexed: JsonObject = {
            "cumulative_billed_nano_usd": int(row["cumulative_billed_nano_usd"]),
            "covered_reservation_sequence": int(row["covered_reservation_sequence"]),
            "covered_published_nano_usd": int(row["covered_published_nano_usd"]),
            "observed_at": str(row["observed_at"]),
            "scope": str(row["scope"]),
            "source": str(row["source"]),
            "verification": str(row["verification"]),
            "decrease_authorized": bool(row["decrease_authorized"]),
        }
        if any(record.get(key) != value for key, value in indexed.items()):
            raise PersistenceError("cash checkpoint indexed fields diverge from its record")
        return {
            "checkpoint_sequence": int(row["checkpoint_sequence"]),
            "checkpoint_id": str(row["checkpoint_id"]),
            **indexed,
            "recorded_at": str(row["recorded_at"]),
            "record_hash": record_hash,
        }

    def _latest_checkpoint(self) -> JsonObject:
        if not self.policy.uses_reconciled_cash_budget:
            raise PersistenceError("cash reconciliation requires a dual-budget policy")
        row = self.connection.execute(
            "SELECT * FROM cash_checkpoint ORDER BY checkpoint_sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return self._opening_checkpoint()
        return self._checkpoint_from_row(row)

    def _cash_balance_locked(self) -> CashBudgetBalance:
        cash = self.policy.cash_budget
        if cash is None:
            raise PersistenceError("cash balance requires a dual-budget policy")
        checkpoint = self._latest_checkpoint()
        covered = _json_int(checkpoint["covered_reservation_sequence"], "cash checkpoint coverage")
        invalid = self.connection.execute(
            "SELECT COUNT(*) AS count FROM reservation WHERE rowid <= ? AND state != 'reconciled'",
            (covered,),
        ).fetchone()
        if invalid is not None and int(invalid["count"]):
            raise PersistenceError("cash checkpoint covers unfinished or uncertain reservations")
        return CashBudgetBalance(
            personal_lifetime_cap_nano_usd=cash.personal_lifetime_cap_nano_usd,
            safety_buffer_nano_usd=cash.safety_buffer_nano_usd,
            reconciled_cash_nano_usd=_json_int(
                checkpoint["cumulative_billed_nano_usd"], "cash checkpoint billed amount"
            ),
            covered_reservation_sequence=covered,
            covered_published_nano_usd=_json_int(
                checkpoint["covered_published_nano_usd"],
                "cash checkpoint covered published amount",
            ),
            unreconciled_actual_nano_usd=self._sum(
                "actual_nano_usd", "state='reconciled' AND rowid > ?", (covered,)
            ),
            uncertain_nano_usd=self._sum(
                "uncertain_nano_usd", "state='uncertain' AND rowid > ?", (covered,)
            ),
            active_reserved_nano_usd=self._sum(
                "reserved_nano_usd", "state='active' AND rowid > ?", (covered,)
            ),
            checkpoint=checkpoint,
        )

    def cash_balance(self) -> CashBudgetBalance:
        return self._cash_balance_locked()

    def status(self) -> JsonObject:
        cash: JsonObject | None = None
        checkpoint_count = 0
        if self.policy.uses_reconciled_cash_budget:
            cash = self.cash_balance().to_value()
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM cash_checkpoint"
            ).fetchone()
            checkpoint_count = 1 + int(row["count"] if row is not None else 0)
        return {
            "ledger_schema_version": (
                DUAL_LEDGER_SCHEMA_VERSION
                if self.policy.uses_reconciled_cash_budget
                else LEDGER_SCHEMA_VERSION
            ),
            "policy_version": self.policy.policy_version,
            "policy_hash": self.policy.content_hash,
            "published_rate_balance": self.balance().to_value(),
            "cash_budget": cash,
            "latest_reservation_sequence": self._latest_reservation_sequence(),
            "reconcilable_through_sequence": self._reconcilable_through_sequence(),
            "cash_checkpoint_count_including_opening": checkpoint_count,
        }

    def cash_checkpoints(self) -> tuple[JsonObject, ...]:
        if not self.policy.uses_reconciled_cash_budget:
            raise PersistenceError("cash checkpoint history requires a dual-budget policy")
        checkpoints = [self._opening_checkpoint()]
        for row in self.connection.execute(
            "SELECT * FROM cash_checkpoint ORDER BY checkpoint_sequence"
        ):
            checkpoint = self._checkpoint_from_row(row)
            checkpoints.append(
                {
                    "checkpoint_sequence": checkpoint["checkpoint_sequence"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "recorded_at": checkpoint["recorded_at"],
                    "record": json.loads(str(row["record_json"])),
                }
            )
        return tuple(checkpoints)

    @staticmethod
    def _validate_observed_at(value: str) -> None:
        try:
            if "T" not in value:
                date.fromisoformat(value)
            else:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError
        except ValueError as exc:
            raise PersistenceError(
                "cash checkpoint observed_at must be an ISO date or offset datetime"
            ) from exc

    def append_cash_checkpoint(
        self,
        *,
        cumulative_billed_nano_usd: int,
        covered_reservation_sequence: int | None,
        observed_at: str,
        scope: str,
        source: str,
        verification: str,
        allow_decrease: bool = False,
    ) -> JsonObject:
        if not self.policy.uses_reconciled_cash_budget:
            raise PersistenceError("cash checkpoints require a dual-budget policy")
        if cumulative_billed_nano_usd < 0:
            raise PersistenceError("cumulative billed cash cannot be negative")
        if not scope.strip() or not source.strip():
            raise PersistenceError("cash checkpoint scope/source must be nonempty")
        if verification not in CASH_VERIFICATION_LEVELS:
            raise PersistenceError("cash checkpoint verification is invalid")
        self._validate_observed_at(observed_at)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            latest = self._latest_checkpoint()
            latest_covered = _json_int(
                latest["covered_reservation_sequence"], "cash checkpoint coverage"
            )
            target = (
                self._reconcilable_through_sequence()
                if covered_reservation_sequence is None
                else covered_reservation_sequence
            )
            if (
                target < 0
                or target < latest_covered
                or target > self._latest_reservation_sequence()
            ):
                raise PersistenceError("cash checkpoint coverage sequence is invalid or regressive")
            unfinished = self.connection.execute(
                "SELECT COUNT(*) AS count FROM reservation "
                "WHERE rowid <= ? AND state != 'reconciled'",
                (target,),
            ).fetchone()
            if unfinished is not None and int(unfinished["count"]):
                raise PersistenceError("cash checkpoint cannot cover active or uncertain usage")
            previous_billed = _json_int(
                latest["cumulative_billed_nano_usd"], "cash checkpoint billed amount"
            )
            if cumulative_billed_nano_usd < previous_billed and not allow_decrease:
                raise PersistenceError(
                    "cumulative billed cash decreased without explicit authorization"
                )
            covered_published = self.policy.opening_balance_nano_usd + self._sum(
                "actual_nano_usd", "state='reconciled' AND rowid <= ?", (target,)
            )
            record: JsonObject = {
                "checkpoint_version": CASH_CHECKPOINT_VERSION,
                "cumulative_billed_nano_usd": cumulative_billed_nano_usd,
                "covered_reservation_sequence": target,
                "covered_published_nano_usd": covered_published,
                "observed_at": observed_at,
                "scope": scope.strip(),
                "source": source.strip(),
                "verification": verification,
                "decrease_authorized": allow_decrease,
                "policy_hash": self.policy.content_hash,
            }
            record_text = canonical_json(record)
            record_hash = sha256_text(record_text)
            existing = self.connection.execute(
                "SELECT checkpoint_sequence FROM cash_checkpoint WHERE record_hash=?",
                (record_hash,),
            ).fetchone()
            if existing is None:
                self.connection.execute(
                    """INSERT INTO cash_checkpoint (
                        checkpoint_id, covered_reservation_sequence,
                        cumulative_billed_nano_usd, covered_published_nano_usd,
                        observed_at, scope, source, verification, decrease_authorized,
                        recorded_at, record_json, record_hash
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record_hash,
                        target,
                        cumulative_billed_nano_usd,
                        covered_published,
                        observed_at,
                        scope.strip(),
                        source.strip(),
                        verification,
                        int(allow_decrease),
                        utc_now(),
                        record_text,
                        record_hash,
                    ),
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return {
            "checkpoint": self._latest_checkpoint(),
            "cash_budget": self.cash_balance().to_value(),
        }

    def reserve(
        self,
        *,
        reservation_id: str,
        run_id: str,
        stage: str,
        request_hash: str,
        amount_nano_usd: int,
        child_cap_nano_usd: int,
    ) -> None:
        if amount_nano_usd < 0:
            raise PersistenceError("request reservation cannot be negative")
        if amount_nano_usd > self.policy.request_cap_nano_usd:
            raise BudgetExhaustedError("request reservation exceeds its hard ceiling")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            prior = self.connection.execute(
                "SELECT * FROM reservation WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if prior is not None:
                if (
                    prior["run_id"] != run_id
                    or prior["stage"] != stage
                    or prior["request_hash"] != request_hash
                    or prior["policy_hash"] != self.policy.content_hash
                    or prior["reserved_nano_usd"] != amount_nano_usd
                ):
                    raise PersistenceError("ledger reservation identity collision")
                self.connection.commit()
                return
            phase_committed = self._sum(
                "actual_nano_usd + uncertain_nano_usd + "
                "CASE WHEN state='active' THEN reserved_nano_usd ELSE 0 END"
            )
            stage_committed = self._sum(
                "actual_nano_usd + uncertain_nano_usd + "
                "CASE WHEN state='active' THEN reserved_nano_usd ELSE 0 END",
                "stage=?",
                (stage,),
            )
            child_committed = self._sum(
                "actual_nano_usd + uncertain_nano_usd + "
                "CASE WHEN state='active' THEN reserved_nano_usd ELSE 0 END",
                "run_id=?",
                (run_id,),
            )
            project_committed = self.policy.opening_balance_nano_usd + phase_committed
            checks = [
                (phase_committed + amount_nano_usd, self.policy.phase4_cap_nano_usd),
                (stage_committed + amount_nano_usd, self.policy.stage_cap(stage)),
                (child_committed + amount_nano_usd, child_cap_nano_usd),
            ]
            if self.policy.uses_reconciled_cash_budget:
                cash = self._cash_balance_locked()
                if (
                    cash.cash_upper_bound_nano_usd + amount_nano_usd
                    > cash.personal_lifetime_cap_nano_usd
                ):
                    raise BudgetExhaustedError(
                        "reservation would exceed the reconciled personal cash ceiling"
                    )
            else:
                checks.append(
                    (
                        project_committed + amount_nano_usd,
                        self.policy.project_lifetime_cap_nano_usd,
                    )
                )
            if any(value > cap for value, cap in checks):
                raise BudgetExhaustedError(
                    "hierarchical model-dollar reservation would exceed a cap"
                )
            self.connection.execute(
                """INSERT INTO reservation (
                    reservation_id, run_id, stage, request_hash, policy_hash,
                    reserved_nano_usd, state
                ) VALUES (?, ?, ?, ?, ?, ?, 'active')""",
                (
                    reservation_id,
                    run_id,
                    stage,
                    request_hash,
                    self.policy.content_hash,
                    amount_nano_usd,
                ),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def reconcile(
        self,
        *,
        reservation_id: str,
        actual_nano_usd: int,
        usage_record: JsonObject,
    ) -> tuple[int, int]:
        text = canonical_json(usage_record)
        record_hash = sha256_text(text)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT * FROM reservation WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise PersistenceError("ledger reservation is missing or already reconciled")
            if row["state"] == "reconciled":
                if int(row["actual_nano_usd"]) != actual_nano_usd:
                    raise PersistenceError("ledger reconciliation identity diverged")
                existing = self.connection.execute(
                    "SELECT record_hash FROM usage_record WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()
                if existing is None or existing["record_hash"] != record_hash:
                    raise PersistenceError("ledger reconciliation record diverged")
                self.connection.commit()
                return int(row["actual_nano_usd"]), int(row["released_nano_usd"])
            if row["state"] != "active":
                raise PersistenceError("ledger reservation is missing or already reconciled")
            reserved = int(row["reserved_nano_usd"])
            if actual_nano_usd < 0 or actual_nano_usd > reserved:
                raise PersistenceError("actual model charge exceeds its preflight reservation")
            released = reserved - actual_nano_usd
            self.connection.execute(
                """UPDATE reservation SET state='reconciled', actual_nano_usd=?,
                   released_nano_usd=? WHERE reservation_id=?""",
                (actual_nano_usd, released, reservation_id),
            )
            self.connection.execute(
                "INSERT INTO usage_record "
                "(reservation_id, record_json, record_hash) VALUES (?,?,?)",
                (reservation_id, text, record_hash),
            )
            self.connection.commit()
            return actual_nano_usd, released
        except Exception:
            self.connection.rollback()
            raise

    def mark_uncertain(self, *, reservation_id: str, failure_record: JsonObject) -> int:
        text = canonical_json(failure_record)
        record_hash = sha256_text(text)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT * FROM reservation WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise PersistenceError("uncertain reservation is missing or already finalized")
            if row["state"] == "uncertain":
                existing = self.connection.execute(
                    "SELECT record_hash FROM usage_record WHERE reservation_id=?",
                    (reservation_id,),
                ).fetchone()
                if existing is None or existing["record_hash"] != record_hash:
                    raise PersistenceError("uncertain reservation record diverged")
                self.connection.commit()
                return int(row["uncertain_nano_usd"])
            if row["state"] != "active":
                raise PersistenceError("uncertain reservation is missing or already finalized")
            amount = int(row["reserved_nano_usd"])
            self.connection.execute(
                """UPDATE reservation SET state='uncertain', uncertain_nano_usd=?
                   WHERE reservation_id=?""",
                (amount, reservation_id),
            )
            self.connection.execute(
                "INSERT INTO usage_record "
                "(reservation_id, record_json, record_hash) VALUES (?,?,?)",
                (reservation_id, text, record_hash),
            )
            self.connection.commit()
            return amount
        except Exception:
            self.connection.rollback()
            raise

    def records(self) -> tuple[sqlite3.Row, ...]:
        return tuple(self.connection.execute("SELECT * FROM reservation ORDER BY rowid").fetchall())


def rebuild_project_ledger(*, repository_root: Path, path: Path, policy: PricePolicy) -> JsonObject:
    """Reconstruct a missing ledger from verified Phase 4 databases and paid artifacts."""

    if path.exists():
        with ProjectLedger(path, policy) as existing:
            return {
                "rebuild": "not-needed-existing-verified",
                "records": len(existing.records()),
                "balance": existing.balance().to_value(),
            }
    if policy.uses_reconciled_cash_budget:
        raise PersistenceError(
            "a missing dual-budget ledger cannot be rebuilt from legacy-policy artifacts; "
            "restore its backup or initialize the versioned opening carry-forward"
        )
    temporary = path.with_name(f"{path.name}.rebuild-in-progress")
    if temporary.exists():
        raise PersistenceError("an incomplete ledger rebuild already exists")
    paid_artifacts = {
        artifact.resolve()
        for artifact in (repository_root / "artifacts").glob("**/responses/*.json")
        if artifact.is_file() and '"provider_id":"openai"' in artifact.read_text(encoding="utf-8")
    }
    referenced_artifacts: set[Path] = set()
    rebuilt_records = 0
    try:
        with ProjectLedger(temporary, policy) as ledger:
            for database_path in sorted((repository_root / "artifacts").glob("**/run.sqlite3")):
                try:
                    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
                    connection.row_factory = sqlite3.Row
                    schema = connection.execute(
                        "SELECT value FROM metadata WHERE key='database_schema_version'"
                    ).fetchone()
                except sqlite3.Error:
                    continue
                if schema is None or schema["value"] != "5":
                    connection.close()
                    continue
                run_directory = database_path.parent
                try:
                    manifest: object = json.loads(
                        (run_directory / "manifest.json").read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    connection.close()
                    raise PersistenceError(
                        "paid run manifest is unavailable during rebuild"
                    ) from exc
                if not isinstance(manifest, dict):
                    connection.close()
                    raise PersistenceError("paid run manifest is malformed during rebuild")
                budget = manifest.get("budget")
                resolved = manifest.get("resolved_configuration")
                if (
                    not isinstance(budget, dict)
                    or budget.get("price_policy_hash") != policy.content_hash
                    or not isinstance(resolved, dict)
                ):
                    connection.close()
                    raise PersistenceError("paid run policy differs during ledger rebuild")
                phase4 = resolved.get("phase4")
                child_budget = resolved.get("budget")
                if not isinstance(phase4, dict) or not isinstance(child_budget, dict):
                    connection.close()
                    raise PersistenceError("paid run spending contract is malformed")
                stage = phase4.get("stage")
                child_cap = child_budget.get("child_nano_usd_cap")
                if (
                    not isinstance(stage, str)
                    or isinstance(child_cap, bool)
                    or not isinstance(child_cap, int)
                ):
                    connection.close()
                    raise PersistenceError("paid run spending caps are malformed")
                rows = tuple(
                    connection.execute(
                        "SELECT * FROM model_request WHERE provider_id='openai' "
                        "AND reservation_id IS NOT NULL ORDER BY request_index"
                    )
                )
                state_row = connection.execute("SELECT run_id FROM run_state").fetchone()
                if state_row is None:
                    connection.close()
                    raise PersistenceError("paid run has no run identifier")
                run_id = str(state_row["run_id"])
                for row in rows:
                    reservation_id = str(row["reservation_id"])
                    reserved = int(row["reserved_nano_usd"])
                    ledger.reserve(
                        reservation_id=reservation_id,
                        run_id=run_id,
                        stage=stage,
                        request_hash=str(row["request_hash"]),
                        amount_nano_usd=reserved,
                        child_cap_nano_usd=child_cap,
                    )
                    response_name = row["response_artifact"]
                    response_hash = row["response_hash"]
                    if response_name is not None:
                        relative = Path(str(response_name))
                        if relative.is_absolute() or ".." in relative.parts:
                            raise PersistenceError("paid response path escapes its run")
                        response_path = run_directory / relative
                        response_text = response_path.read_text(encoding="utf-8").rstrip("\n")
                        if not isinstance(response_hash, str) or sha256_text(response_text) != (
                            response_hash
                        ):
                            raise PersistenceError("paid response hash mismatch during rebuild")
                        referenced_artifacts.add(response_path.resolve())
                    record: JsonObject = {
                        "reconstruction_version": 1,
                        "run_id": run_id,
                        "request_index": int(row["request_index"]),
                        "request_hash": str(row["request_hash"]),
                        "response_hash": str(response_hash) if response_hash is not None else None,
                    }
                    state = str(row["state"])
                    if state in {"usage-uncertain", "dispatched"}:
                        ledger.mark_uncertain(
                            reservation_id=reservation_id,
                            failure_record=record,
                        )
                    elif state != "pending":
                        ledger.reconcile(
                            reservation_id=reservation_id,
                            actual_nano_usd=int(row["actual_nano_usd"]),
                            usage_record=record,
                        )
                    rebuilt_records += 1
                connection.close()
            if paid_artifacts - referenced_artifacts:
                raise PersistenceError("orphan paid response artifacts prevent ledger rebuild")
            ledger.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            balance = ledger.balance().to_value()
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, path)
    except Exception:
        raise
    return {
        "rebuild": "completed-from-verified-run-records",
        "records": rebuilt_records,
        "paid_artifacts": len(paid_artifacts),
        "balance": balance,
        "policy_hash": policy.content_hash,
    }
