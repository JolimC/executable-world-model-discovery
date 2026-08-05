from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from world_model_search.config import RunSettings, load_config
from world_model_search.tasks import generate_benchmark


def test_generation_is_separated_disjoint_and_deterministic(tmp_path) -> None:
    config = load_config(Path("configs/smoke.yaml"))
    config = replace(
        config,
        run=RunSettings(
            root=Path(tmp_path.name) / "runs",
            seed=config.run.seed,
            max_steps=4,
            task_id="fixture",
            split=config.run.split,
        ),
    )
    first = generate_benchmark(tmp_path, config)
    manifest = first.manifest
    tasks = manifest["tasks"]
    assert isinstance(tasks, list) and len(tasks) == 256
    assert len({task["semantic_hash"] for task in tasks}) == 256
    assert manifest["test_outcomes_accessed"] is False
    public_text = "".join(path.read_text() for path in (first.root / "public").iterdir())
    assert "reference_rule" not in public_text
    assert "semantic_hash" not in public_text
    assert "seed" not in public_text
    assert len(list((first.root / "oracle").iterdir())) == 256
    snapshot = {
        p.relative_to(first.root): p.read_bytes() for p in first.root.rglob("*") if p.is_file()
    }
    second_config = replace(config, run=replace(config.run, root=Path("other/runs")))
    second = generate_benchmark(tmp_path, second_config)
    assert snapshot == {
        p.relative_to(second.root): p.read_bytes() for p in second.root.rglob("*") if p.is_file()
    }
