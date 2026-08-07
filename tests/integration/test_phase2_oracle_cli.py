from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from world_model_search.cli import main
from world_model_search.tasks import HiddenTaskStore

RULE90_TASK = "d737b0ee219de6a676c139d1"


def _write_candidate(path: Path, ast: object) -> None:
    path.write_text(
        json.dumps(
            {
                "candidate_schema_version": 1,
                "dsl_version": "binary-ca-radius1-dsl-v1",
                "ast": ast,
            }
        ),
        encoding="utf-8",
    )


def test_oracle_verify_exact_and_failed_exit_contract(tmp_path: Path, capsys) -> None:
    exact_path = tmp_path / "exact.json"
    _write_candidate(
        exact_path,
        {
            "op": "Xor",
            "left": {"op": "At", "offset": -1},
            "right": {"op": "At", "offset": 1},
        },
    )
    assert main(["oracle", "verify", "--task", RULE90_TASK, "--candidate", str(exact_path)]) == 0
    exact_output = json.loads(capsys.readouterr().out)
    assert exact_output["result"]["exact"] is True
    assert exact_output["result"]["ast_bits"] == 14
    assert set(exact_output["diagnostics"]) == {"runtime_ns"}

    wrong_path = tmp_path / "wrong.json"
    _write_candidate(wrong_path, {"op": "Const", "value": 0})
    assert main(["oracle", "verify", "--task", RULE90_TASK, "--candidate", str(wrong_path)]) == 1
    output_text = capsys.readouterr().out
    wrong_output = json.loads(output_text)
    assert wrong_output["result"]["exact"] is False
    for forbidden in (
        "reference_rule",
        "ordered_semantics",
        "semantic_hash",
        "locked_rollout",
        "internal_family",
        "seed",
    ):
        assert forbidden not in output_text


def test_invalid_candidate_is_rejected_before_hidden_task_load(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys
) -> None:
    candidate = tmp_path / "invalid.json"
    candidate.write_text(
        '{"candidate_schema_version":1,"dsl_version":"binary-ca-radius1-dsl-v1",'
        '"ast":{"op":"Const","value":true}}',
        encoding="utf-8",
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid candidate reached hidden task authority")

    monkeypatch.setattr(HiddenTaskStore, "load", forbidden)
    assert main(["oracle", "verify", "--task", RULE90_TASK, "--candidate", str(candidate)]) == 2
    assert "must be an integer" in capsys.readouterr().err
