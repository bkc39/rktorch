Fair — comparing the values with `tol` and then the bytes exactly is
inconsistent, and a difference the first tolerates moves a byte at a
rounding boundary. The byte check is now `check-=` with a tolerance of 1,
with the length compared separately so a truncated write still fails.
Fixed in e817c34; the GPU parity run is green.
