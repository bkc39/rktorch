Good catch — the stateful case is the real problem: reading the rate
advanced the callback, so a log line changed the schedule. `apply-rate!`
now stores what it writes and `scheduler-rate` returns that, which is
also what `get_last_lr()` reports. The test counts callback invocations:
construction is one, two reads of `scheduler-rate` add none, and a
`step!` adds one. Fixed in 5f93371, and the manual now says so.
