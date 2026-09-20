Both confirmed and fixed in bc21466.

The int64 bound is the sharper of the two: 2^63 is a power of two, so it
passes the double round-trip exactly and only failed later, in the
native conversion. The branch now asks for the signed range as well:

```racket
(and (integer? value)
     (<= (- (expt 2 63)) value (sub1 (expt 2 63)))
     (= (exact->inexact value) value))
```

and there is a `bool` case requiring 0 or 1, so that dtype is no longer
the one that silently truncates — torch makes 0.5 true there, which is
the same departure this table already makes for uint8 and int64.
