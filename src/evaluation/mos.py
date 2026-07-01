"""MOS (Mean Opinion Score) harness (required creation metric #1).

MOS is a human judgement (naturalness, 1-5). This module prepares the listening test
and aggregates results; it does not invent scores.

TODO (implementation pass):
    - build_test(clips): shuffle real + A1 + A2 clips into a blind rating sheet/form
    - aggregate(responses) -> per-system mean MOS + 95% CI
    - (optional, flagged non-course) auto_mos(wav): UTMOS/NISQA proxy to sanity-check
Deliverable: a small panel (>=10 raters where possible) + a MOS table real vs A1 vs A2.
"""


def aggregate(responses):
    """Aggregate listener ratings into per-system mean MOS + CI. TODO: implement."""
    raise NotImplementedError
