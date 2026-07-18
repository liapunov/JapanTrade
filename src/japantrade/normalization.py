"""Composable normalization pipeline for raw Japan Customs data."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .tradefile import TradeFile


@dataclass
class NormalizationConfig:
    """Configuration for the trade file normalization pipeline."""

    chunk_size: int = 50000
    use_tqdm: bool = False
    parallel_chunks: bool = False
    max_workers: Optional[int] = None
    output_formats: Tuple[str, ...] = ("csv", "parquet")
    include_descriptions: bool = True
    convert_to_base_units: bool = False
    keep_units: Optional[Tuple[str, ...]] = None
    exclude_units: Optional[Tuple[str, ...]] = None
    warn_on_unknown: bool = True
    lookup_paths: Optional[dict] = None

    def __post_init__(self):
        self.output_formats = tuple(fmt.lower() for fmt in self.output_formats)
        if self.keep_units is not None:
            self.keep_units = tuple(unit.upper() for unit in self.keep_units)
        if self.exclude_units is not None:
            self.exclude_units = tuple(unit.upper() for unit in self.exclude_units)
        if self.lookup_paths is not None:
            self.lookup_paths = {key.lower(): Path(value) for key, value in self.lookup_paths.items()}


@dataclass
class NormalizationContext:
    """Context passed to each normalization stage."""

    kind: str
    chunk_index: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class NormalizationPipeline:
    """Run canonical and caller-supplied normalization stages."""

    def __init__(self, processor: "TradeFile", config: NormalizationConfig,
                 hooks: Optional[List[Tuple[str, Callable]]] = None):
        self.processor = processor
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._custom_steps: List[dict] = []
        self._tqdm = getattr(processor, "_tqdm", None)
        if hooks:
            for name, func in hooks:
                self.add_step(name, func)

    def add_step(self, name: str, func: Callable, position: Optional[int] = None,
                 before: Optional[str] = None, after: Optional[str] = None):
        self._custom_steps.append({
            "name": name, "func": func, "position": position,
            "before": before, "after": after,
        })

    def _default_steps(self) -> List[Tuple[str, Callable]]:
        def clean(df, context):
            cleaned = self.processor._cleanDataFile(df)
            self.processor._validate_raw_schema(cleaned.columns, context.kind)
            return cleaned, {}

        def melt_months(df, context):
            return self.processor._meltMonths(df, context.kind)

        def melt_units(df, context):
            return self.processor._meltUnits(df, context.kind)

        def normalize_units(df, context):
            return self.processor._normalize_units(df)

        def reduce_rows(df, context):
            return self.processor._reduce_rows(df), {}

        def enrich(df, context):
            return self.processor._enrich_with_lookups(df, context.kind)

        return [
            ("clean", clean), ("melt_months", melt_months),
            ("melt_units", melt_units), ("normalize_units", normalize_units),
            ("reduce", reduce_rows), ("enrich", enrich),
        ]

    def _compose_steps(self) -> List[Tuple[str, Callable]]:
        steps = list(self._default_steps())
        for custom in self._custom_steps:
            insertion = (custom["name"], custom["func"])
            if custom.get("position") is not None:
                steps.insert(custom["position"], insertion)
            elif custom.get("before"):
                index = next((i for i, (name, _) in enumerate(steps) if name == custom["before"]), len(steps))
                steps.insert(index, insertion)
            elif custom.get("after"):
                index = next((i for i, (name, _) in enumerate(steps) if name == custom["after"]), None)
                steps.insert(index + 1, insertion) if index is not None else steps.append(insertion)
            else:
                steps.append(insertion)
        return steps

    def run(self, df: pd.DataFrame, kind: str, chunk_index: Optional[int] = None):
        context = NormalizationContext(kind=kind, chunk_index=chunk_index)
        steps = self._compose_steps()
        iterator = self._tqdm(steps, desc="Normalizing chunk", unit="step") if self._tqdm else steps
        metrics: List[dict] = []
        for name, func in iterator:
            before_rows = len(df)
            started = time()
            result = func(df, context)
            df, step_metrics = result if isinstance(result, tuple) and len(result) == 2 else (result, {})
            step_metrics = step_metrics or {}
            duration = time() - started
            step_metrics.update({"duration": duration, "before_rows": before_rows, "after_rows": len(df)})
            metrics.append({"step": name, **step_metrics})
            self.logger.info("Pipeline step '%s' completed for chunk %s | rows %s -> %s | %.2fs",
                             name, chunk_index, before_rows, len(df), duration)
        return df, metrics
