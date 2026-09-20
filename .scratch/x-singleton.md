Already done, just ahead of this review: 138104d added the fast path,
which landed while the review was computing against 702d943. `image-grid`
now returns a singleton as `make_grid` does, after the one-to-three
channel expansion, the twin's fallback carries the same early return, and
the cross-test compares a seeded `[1 3 4 4]` batch against it.
