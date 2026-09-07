"""Measure time-to-first-token (TTFT), inter-token gaps, and tokens/sec."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class StreamStats:
    ttft_s: float | None = None
    total_s: float = 0.0
    tokens: int = 0
    gaps_s: list[float] = field(default_factory=list)

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.total_s if self.total_s else 0.0

    @property
    def mean_gap_ms(self) -> float:
        return 1000 * sum(self.gaps_s) / len(self.gaps_s) if self.gaps_s else 0.0

    def summary(self) -> str:
        ttft = f"{self.ttft_s * 1000:.0f} ms" if self.ttft_s is not None else "n/a"
        return (
            f"TTFT {ttft} | {self.tokens} tok in {self.total_s:.2f}s "
            f"| {self.tokens_per_s:.1f} tok/s | mean gap {self.mean_gap_ms:.0f} ms"
        )


async def instrument(
    source: AsyncIterator[str], stats: StreamStats
) -> AsyncIterator[str]:
    """Pass tokens through unchanged, filling `stats` as they flow."""
    start = time.perf_counter()
    last = start
    async for token in source:
        now = time.perf_counter()
        if stats.ttft_s is None:
            stats.ttft_s = now - start
        else:
            stats.gaps_s.append(now - last)
        last = now
        stats.tokens += 1
        yield token
    stats.total_s = time.perf_counter() - start
