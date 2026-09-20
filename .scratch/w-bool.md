Agreed — ATen has no subtraction for a boolean tensor, so the contract
was admitting something the body could not handle. `image/c` now excludes
the boolean dtype, with the reason next to it and a test that the refusal
is contract blame naming `image`. Fixed in e817c34.
