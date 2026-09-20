Agreed, same as the thread beside this one. Both contracts now ask
`(andmap positive? dims)` over the whole shape rather than the last two
dimensions, so `[N 0 H W]` gets contract blame; test added. Fixed in
bc21466.
