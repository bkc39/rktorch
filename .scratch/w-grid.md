Confirmed against upstream: `make_grid` has

```python
if tensor.size(0) == 1:
    return tensor.squeeze(0)
```

after the one-to-three channel expansion and before the grid is built, so
a singleton comes back unpadded and a one-channel singleton still comes
back with three channels. `image-grid` takes that path now, the twin's
fallback gained the same early return, and the cross-test compares a
seeded `[1 3 4 4]` batch against it. The docs say it too. Fixed in
e817c34.
