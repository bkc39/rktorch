The boolean half is fixed — `image/c` refuses that dtype, since ATen has
no subtraction for it.

The int64 half I'd rather leave. `#:range` is exactly the control for
it: the manual defines the pair as "the values that map to 0 and 255", so
an int64 image holding 0 through 255 is written correctly with
`#:range '(0 255)`, and that call would stop working if the contract
were narrowed to float32/float64/uint8. The surprise you describe is not
about the dtype either — a float32 tensor holding 0 through 255 quantizes
to white under the default range in exactly the same way. uint8 is the
one dtype that carries its range in the dtype, which is why it is the one
special case.

Happy to revisit if you would rather the default range were removed than
kept, but that is a wider change than this contract.
