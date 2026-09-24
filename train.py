#!/usr/bin/env python3
"""
Convenience entry point for CIFAR-10 WideResNet training and evaluation.
"""
from cifar_competition import main, parser

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
