Agreed, and it is the same point as the `image-grid` thread earlier —
`save_image` runs under no-grad for this reason. The quantization is
wrapped now, with a test that writing a `#:requires-grad?` image
produces the right byte count rather than raising.

Fixed in fbf4db4.
