"""TellTail reviewer demo — two runnable things:

    python -m demo fingerprint              # fingerprint the victim retrievers (CPU)
    python -m demo topic                    # topic-level / response-only attack demo (CPU)
    python -m demo optimize [--query-id N]  # optimize a NEW query for minilm-l6 (GPU)
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(prog="demo", description="TellTail reviewer demo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    from . import fingerprint
    fp = sub.add_parser("fingerprint", help="fingerprint the demo victim retrievers")
    fingerprint.add_arguments(fp)

    from . import topic
    tp = sub.add_parser("topic", help="topic-level / response-only attack demo")
    topic.add_arguments(tp)

    op = sub.add_parser("optimize", help="optimize a new query for minilm-l6 and test it")
    op.add_argument("--query-id", type=int, default=None)
    op.add_argument("--device", default=None)

    args = ap.parse_args()
    if args.cmd == "fingerprint":
        fingerprint.main(args)
    elif args.cmd == "topic":
        topic.main(args)
    elif args.cmd == "optimize":
        try:
            from . import optimize as demo_optimize
        except ImportError:
            print("`demo optimize` is not available yet (TASK 4).", file=sys.stderr)
            sys.exit(2)
        demo_optimize.main(args)


if __name__ == "__main__":
    main()
