#!/usr/bin/env python3
"""
Convenience entry point for CAGS image classification training and evaluation.
"""
from cags_classification import main, main_ensemble, eval_ensemble, parser

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    if args.eval_ensemble:
        eval_ensemble(args)
    elif args.ensemble is not None:
        main_ensemble(args)
    else:
        main(args)
