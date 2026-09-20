Agreed — a P6 header states a width and a height and neither may be
zero, so `[3 0 W]` produced a file no reader accepts. `image/c` requires
both to be positive now, with a test for each.
