"""Fixture for the twin test of ops.py: how often a case's warmup and timed
rounds empty the caching allocator, which rktorch's op rounds leave warm."""

from common import GcClock, emit
from ops import case_record

import torch


def empty_cache_calls():
    calls = []
    empty_cache = torch.cuda.empty_cache
    torch.cuda.empty_cache = lambda: calls.append(None)
    try:
        case_record("add", 2, lambda: torch.ones(2) + 1, torch.device("cpu"),
                    GcClock(), warmup=1, rounds=2)
    finally:
        torch.cuda.empty_cache = empty_cache
    return len(calls)


def main():
    emit({"case": "empty-cache", "calls": empty_cache_calls()})


if __name__ == "__main__":
    main()
