Agreed — the manual documents `(-> exact-nonnegative-integer? real?)` and
the code took any arity-1 procedure, so the two disagreed. `lambda-lr`
now takes the documented contract, and a callback returning `0+1i` is
caught at the boundary with the caller blamed:

```
lambda-lr: contract violation
  expected: real?
  given: 0+1i
  in: the range of
      the 2nd argument of ...
  blaming: (... scheduler-test.rkt test)
```

Fixed in 5f93371, with that case in scheduler-test.rkt.
