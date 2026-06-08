from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from durable_unlearning.data.io import read_json, write_json
from durable_unlearning.data.types import ThresholdRecord


QUANTILE_TO_FIELD = {
    "q90": "tau_q90",
    "q95": "tau_q95",
    "q99": "tau_q99",
    "0.90": "tau_q90",
    "0.95": "tau_q95",
    "0.99": "tau_q99",
}


def quantile_field(quantile: Union[str, float]) -> str:
    key = f"{quantile:.2f}" if isinstance(quantile, float) else str(quantile)
    if key not in QUANTILE_TO_FIELD:
        raise ValueError(f"Unsupported threshold quantile: {quantile}")
    return QUANTILE_TO_FIELD[key]


def compute_threshold_records(
    margin_rows: Iterable[dict[str, object]],
    quantiles: tuple[float, float, float] = (0.90, 0.95, 0.99),
    tau_global: float = 0.0,
    oracle_model_hash: str = "",
) -> list[ThresholdRecord]:
    grouped: dict[str, list[float]] = defaultdict(list)
    variant_ids: dict[str, list[str]] = defaultdict(list)
    for row in margin_rows:
        item_id = str(row["item_id"])
        grouped[item_id].append(float(row["margin"]))
        if row.get("variant_id") is not None:
            variant_ids[item_id].append(str(row["variant_id"]))
    records: list[ThresholdRecord] = []
    for item_id in sorted(grouped):
        values = np.asarray(grouped[item_id], dtype=float)
        tau_q90 = max(float(np.quantile(values, quantiles[0])), tau_global)
        tau_q95 = max(float(np.quantile(values, quantiles[1])), tau_global)
        tau_q99 = max(float(np.quantile(values, quantiles[2])), tau_global)
        records.append(
            ThresholdRecord(
                item_id=item_id,
                tau_q90=tau_q90,
                tau_q95=tau_q95,
                tau_q99=tau_q99,
                tau_global=tau_global,
                selected_tau=tau_q95,
                oracle_model_hash=oracle_model_hash,
                prompt_variant_ids=sorted(set(variant_ids[item_id])),
            )
        )
    return records


def write_thresholds(path: Union[str, Path], records: list[ThresholdRecord]) -> None:
    write_json(
        path,
        {
            "records": [record.to_dict() for record in records],
            "default_quantile": "q95",
        },
    )


def load_thresholds(path: Union[str, Path], quantile: str = "q95") -> dict[str, float]:
    data = read_json(path)
    field = quantile_field(quantile)
    records = data.get("records", data if isinstance(data, list) else [])
    thresholds: dict[str, float] = {}
    for row in records:
        record = ThresholdRecord.from_dict(row)
        thresholds[record.item_id] = float(getattr(record, field))
    return thresholds
