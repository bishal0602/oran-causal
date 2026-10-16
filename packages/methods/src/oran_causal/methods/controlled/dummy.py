"""End-to-end test arm: iid uniform p-values (score = 1 - p), random signs, BY at q = .05.

Reads only the candidate list and the dataset identity, which seeds its stream
`SeedSequence([7899, seed, world, regime, round(1000 lambda), n])`.
"""

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from oran_causal.core.data import Dataset, EdgeResult, Result
from oran_causal.core.multiplicity import weighted_step_up

RNG_TAG = 7899


class DummyMethod:
    """Uniform-p test arm."""

    name = "dummy"
    version = "1"

    def tune(self, dev: list[Dataset], truth_free: bool = True) -> dict[str, Any]:
        """The fixed configuration `{"q": .05}`."""
        return {"q": 0.05}

    def run(self, data: Dataset, config: Mapping[str, Any] | None = None) -> Result:
        """Draws a uniform p-value and a sign per candidate and declares by BY."""
        config = dict(config or {})
        t0 = time.process_time()
        lam = round(1000 * float(data.meta.get("lam", 0.0)))
        key = [RNG_TAG, int(data.seed), int(data.world[1]), int(data.regime[1]), lam, int(data.n)]
        rng = np.random.default_rng(np.random.SeedSequence(key))
        m = len(data.candidates)
        p = rng.uniform(size=m)
        sign = rng.choice([-1, 1], size=m)
        dec = weighted_step_up(p, None, float(config.get("q", 0.05)), reshape=True)
        edges = tuple(
            EdgeResult(s, t, float(1 - p[i]), float(p[i]), int(sign[i]), bool(dec[i]))
            for i, (s, t) in enumerate(data.candidates)
        )
        return Result(self.name, self.version, edges, time.process_time() - t0, config)
