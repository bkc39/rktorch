Agreed on both. nn.scrbl already says it in prose — "A schedule wraps an
optimizer and answers to `step!` like one: the rate for step 0 is written
at construction, and each `step!` on the schedule advances its count and
writes the rate for it" — so the struct comment and the `build` comment
only repeated the manual. Both removed in 5f93371.
