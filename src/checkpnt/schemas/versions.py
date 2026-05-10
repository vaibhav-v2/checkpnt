"""
checkpnt.schemas.versions
--------------------------
Migration chain for Checkpoint schema evolution.

Every migration is a pure function: dict -> dict.
Migrations are composable and tested independently.

When adding a new schema version:
1. Add a migration function: migrate_X_Y_to_X_Z(data: dict) -> dict
2. Register it in MIGRATION_CHAIN
3. Bump CURRENT_SCHEMA_VERSION in checkpoint.py
4. Add tests in tests/unit/test_schemas.py
"""

from __future__ import annotations
from typing import Any, Callable

MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

# Ordered list of (from_version, to_version, migration_fn) tuples.
# Migrations run in sequence. A checkpoint at v1.0 loading against SDK v2.1
# will run: 1.0→1.1, then 1.1→2.0, then 2.0→2.1.
MIGRATION_CHAIN: list[tuple[str, str, MigrationFn]] = [
    # ("1.0", "1.1", migrate_1_0_to_1_1),  # Example — uncomment when needed
]


def migrate(data: dict[str, Any], from_version: str) -> dict[str, Any]:
    """
    Run all applicable migrations on a raw checkpoint dict.
    Returns the dict at the latest schema version.
    """
    from checkpnt.schemas.checkpoint import CURRENT_SCHEMA_VERSION

    if from_version == CURRENT_SCHEMA_VERSION:
        return data  # Already current — fast path

    current = data.copy()
    current_version = from_version

    for (v_from, v_to, fn) in MIGRATION_CHAIN:
        if current_version == v_from:
            current = fn(current)
            current["schema_version"] = v_to
            current_version = v_to

        if current_version == CURRENT_SCHEMA_VERSION:
            break

    if current_version != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"No migration path from schema version '{from_version}' "
            f"to '{CURRENT_SCHEMA_VERSION}'. "
            f"Available migrations: {[(f, t) for f, t, _ in MIGRATION_CHAIN]}"
        )

    return current
