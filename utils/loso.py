from __future__ import annotations

import random


def create_loso_subject_splits(
    total_subjects: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_folds: int | None = None,
) -> list[dict[str, object]]:
    """
    Create LOSO subject splits.
    Each fold uses one test subject; validation subjects are sampled from remaining subjects.
    """
    if total_subjects < 2:
        raise ValueError(f"total_subjects must be >=2, got {total_subjects}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    fold_subjects = list(range(total_subjects))
    if num_folds is not None:
        if num_folds <= 0:
            raise ValueError(f"num_folds must be >0, got {num_folds}")
        fold_subjects = fold_subjects[: min(num_folds, total_subjects)]

    splits = []
    for fold_idx, test_subject in enumerate(fold_subjects):
        train_pool = [s for s in range(total_subjects) if s != test_subject]
        rng = random.Random(seed + fold_idx)
        rng.shuffle(train_pool)

        val_size = max(1, int(round(len(train_pool) * val_ratio)))
        if val_size >= len(train_pool):
            val_size = max(1, len(train_pool) - 1)

        val_subjects = sorted(train_pool[:val_size])
        train_subjects = sorted(train_pool[val_size:])
        if len(train_subjects) == 0:
            raise ValueError("Empty train_subjects after split.")

        splits.append(
            {
                "fold_index": fold_idx,
                "test_subject": int(test_subject),
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": [int(test_subject)],
            }
        )
    return splits
