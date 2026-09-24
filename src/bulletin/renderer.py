"""Composicion final del boletin en paginas PNG."""

from __future__ import annotations

from pathlib import Path

from .charts import render_pages
from .models import BulletinDraft, UnifiedBulletinContext


class BulletinRenderer:
    def render(
        self,
        context: UnifiedBulletinContext,
        draft: BulletinDraft,
        output_dir: Path,
    ) -> tuple[Path, ...]:
        return render_pages(context, draft, output_dir)
