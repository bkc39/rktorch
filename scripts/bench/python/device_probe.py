"""Fixture for the twin test of common.pick_device: a host with no CUDA that
advertises MPS, where the probe allocation raises, the transfer back raises,
or both succeed."""

from contextlib import contextmanager

import common

import torch


class Unmovable:
    def to(self, _device):
        raise RuntimeError("transfer failed")


@contextmanager
def advertised_mps(zeros):
    saved = (torch.cuda.is_available, torch.backends.mps.is_available, torch.zeros)
    torch.cuda.is_available = lambda: False
    torch.backends.mps.is_available = lambda: True
    torch.zeros = zeros
    try:
        yield
    finally:
        torch.cuda.is_available, torch.backends.mps.is_available, torch.zeros = saved


def picked(zeros):
    with advertised_mps(zeros):
        return common.pick_device().type


def cannot_allocate(*_size, device=None):
    raise RuntimeError("MPS backend out of memory")


def cannot_transfer(*_size, device=None):
    return Unmovable()


def allocates(*size, device=None):
    return torch.empty(size).zero_()


def main():
    common.emit({"allocation_raises": picked(cannot_allocate),
                 "transfer_raises": picked(cannot_transfer),
                 "probe_succeeds": picked(allocates)})


if __name__ == "__main__":
    main()
