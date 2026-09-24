"""Contratos serializables de los datos tecnicos."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TechnicalRecord:
    date: str
    symbol: str
    price_close: float | None
    variation_pct: float | None
    ma20: float | None
    ma50: float | None
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    support: float | None
    resistance: float | None
    trend: str
    ema50: float | None = None
    ema200: float | None = None
    raw_signal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TechnicalRecord":
        return cls(**value)


@dataclass(frozen=True)
class TechnicalDataset:
    schema_version: str
    source_file: str
    source_sha256: str
    generated_at_utc: str
    records: tuple[TechnicalRecord, ...]
    warnings: tuple[str, ...] = ()

    @property
    def dates(self) -> tuple[str, ...]:
        return tuple(sorted({record.date for record in self.records}))

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted({record.symbol for record in self.records}))

    def records_for_date(self, value: str) -> tuple[TechnicalRecord, ...]:
        return tuple(record for record in self.records if record.date == value)

    def history_for_symbol(
        self,
        symbol: str,
        *,
        through_date: str,
        limit: int = 60,
    ) -> tuple[TechnicalRecord, ...]:
        selected = sorted(
            (
                record
                for record in self.records
                if record.symbol == symbol and record.date <= through_date
            ),
            key=lambda record: record.date,
        )
        return tuple(selected[-limit:])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "file": self.source_file,
                "sha256": self.source_sha256,
            },
            "generated_at_utc": self.generated_at_utc,
            "quality": {
                "record_count": len(self.records),
                "date_count": len(self.dates),
                "symbol_count": len(self.symbols),
                "warnings": list(self.warnings),
            },
            "records": [record.to_dict() for record in self.records],
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: Path) -> "TechnicalDataset":
        value = json.loads(path.read_text(encoding="utf-8"))
        source = value["source"]
        quality = value.get("quality", {})
        return cls(
            schema_version=value["schema_version"],
            source_file=source["file"],
            source_sha256=source["sha256"],
            generated_at_utc=value["generated_at_utc"],
            records=tuple(
                TechnicalRecord.from_dict(item) for item in value["records"]
            ),
            warnings=tuple(quality.get("warnings", [])),
        )
