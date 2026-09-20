True — `stack` rejects the empty list, so an empty batch raised from
inside the transform. Both transforms now return the input unchanged when
the batch is empty, with a test for the `[0 3 4 4]` case. Fixed in
97d8279.
