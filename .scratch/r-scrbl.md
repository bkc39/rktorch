Right, and it would have raised: `#:training?` defaults to `#f`, and the
eval path needs the running statistics. The example now carries them as
fields, which is also closer to what the layer actually holds:

```racket
(define-layer BatchNorm2d (weight bias running-mean running-var)
  #:forward ([x : image-batch/c])
  (batch-norm x #:weight weight #:bias bias
              #:running-mean running-mean #:running-var running-var))
```

Fixed in 97d8279.
