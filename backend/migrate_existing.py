"""Adopt an existing database without recomputing rows or replacing indexes."""

import argparse
import os
from pathlib import Path


def validate_plan(diffs):
    allowed = {"raw_attributes", "topic", "style", "tone", "scenes"}
    if set(diffs) != {"creators", "videos", "video_scenes"}:
        raise RuntimeError("Expected the complete existing application schema")
    for name, diff in diffs.items():
        if not diff.exists or diff.resolution not in {"up_to_date", "update_additive"}:
            raise RuntimeError(f"Refusing incompatible migration for {name}")
        for op in diff.ops:
            if (
                name != "videos"
                or op.target != "column"
                or op.op != "alter"
                or op.name not in allowed
                or op.destructive
                or op.details.value is None
                or op.details.previous_value is None
            ):
                raise RuntimeError(f"Refusing unexpected schema operation: {op}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True, help="Explicit Pixeltable home")
    parser.add_argument("--apply", action="store_true", help="Apply the computed-expression updates")
    args = parser.parse_args()
    home = args.home.resolve()
    if not (home / "pgdata" / "PG_VERSION").is_file():
        parser.error("--home must contain an existing pgdata/PG_VERSION")

    # config loads dotenv with override=True; pin the explicit target afterwards.
    import config

    os.environ["PIXELTABLE_HOME"] = str(home)
    import app_existing

    print(f"Target: {home}; namespace: {config.APP_NAMESPACE}")
    print("Opening this home can upgrade its Pixeltable metadata. Back it up before --apply.")
    base = app_existing.TableModel
    diffs = base.get_model_diff(config.APP_NAMESPACE)
    base.diff_all(config.APP_NAMESPACE)
    validate_plan(diffs)
    if not args.apply:
        print("Plan only: no schema change was applied. Pass --apply after verifying the target and backup.")
        return
    base.update_all(config.APP_NAMESPACE, allow_destructive=False)
    remaining = base.get_model_diff(config.APP_NAMESPACE)
    if any(d.resolution != "up_to_date" for d in remaining.values()):
        raise RuntimeError("Schema reconciliation incomplete; re-run, or restore the backup")
    print("Schema adopted; existing rows and embedding indexes were not recomputed.")


if __name__ == "__main__":
    main()
