import json
from pathlib import Path
from typing import Any, Dict


def save_dataset_statistics(statistics: Dict[str, Any], run_dir: Path) -> None:
    if not statistics:
        return
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "dataset_statistics.json"
    with open(path, "w") as fp:
        json.dump(statistics, fp, indent=2)
