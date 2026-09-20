This one is overtaken by 42f6a66, which landed while the review was
computing against fbf4db4: the owner's call was to restrict the dtypes,
so `image/c` now accepts only float32, float64, float16, bfloat16 and
uint8. An int64 image never reaches the range arithmetic — it is
contract blame at the boundary, with a test.

The analysis was right for the code it read, and it is the same
mechanism as the BatchNorm counter earlier in this arc: an integer
tensor combined with a floating scalar promotes and rounds. Worth
recording here in case an integer path is ever wanted; it would need
float64 arithmetic, not float32.
