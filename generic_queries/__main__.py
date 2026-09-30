"""CLI dispatch: python -m generic_queries {retrieve,fetch,score,evaluate,sweep}."""
from __future__ import annotations

import argparse

from . import evaluate as _evaluate
from . import sweep as _sweep
from . import retrieve as _retrieve
from . import fetch as _fetch
from . import score as _score

_SUBCOMMANDS = {
    "retrieve": _retrieve,
    "fetch": _fetch,
    "score": _score,
    "evaluate": _evaluate,
    "sweep": _sweep,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="generic_queries", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, mod in _SUBCOMMANDS.items():
        sp = sub.add_parser(name, help=mod.__doc__.splitlines()[0] if mod.__doc__ else name)
        mod.add_arguments(sp)
        sp.set_defaults(_handler=mod.main)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
