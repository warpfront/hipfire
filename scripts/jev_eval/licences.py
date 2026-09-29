"""The one licence choke point for decide calibration (bench/jev/DATA-LICENSES.md).

Human decision (2026-09-29): every temperature is fitted on licence-clear
held-out sources only. The sources below have restrictive or unclear
redistribution terms; heldout.py `run` does not collect them unless asked
explicitly (--allow-restricted), and calibrate.py refuses to fit on any row
from them. Score-type T therefore comes from synth-score alone.

Source names are the `source` field of a row: jev-bench task names
(heldout rows and frozen reported rows alike) and calibration-set names."""

RESTRICTED_SOURCES = frozenset({"ag-news", "duplicates", "yelp-stars", "offensive", "sentiment-it", "hellaswag"})


def is_restricted(source):
    return source in RESTRICTED_SOURCES


def restricted_in(sources):
    """The restricted names among `sources`, sorted."""
    return sorted(s for s in set(sources) if is_restricted(s))
