#lang racket/base

(module+ test
  (require rackunit
           (only-in threading ~>)
           (only-in "../main.rkt" * backward! grad item reshape sum tensor tensor->list zeros)
           (only-in "../nn.rkt" Parameter clip-grad-norm! clip-grad-value!)
           (only-in "private/python-env.rkt" python-check python-torch-available? tol))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (number-of v)
    (if (string? v) (string->number v) v))

  (define (parameters-with grads dtypes)
    (for/list ([g (in-list grads)] [dt (in-list dtypes)])
      (define p (Parameter (zeros (length g) #:dtype (string->symbol dt))))
      (~> (* p (tensor g #:dtype (string->symbol dt))) sum backward!)
      p))

  (define (check-grads ps want label check)
    (for ([p (in-list ps)] [w (in-list want)] [i (in-naturals)])
      (for ([x (in-list (flat (grad p)))] [y (in-list w)] [k (in-naturals)])
        (check x y (format "~a: gradient ~a[~a]" label i k)))))

  (define (check-norm-case c)
    (define label (format "G0-L7: clip_grad_norm_ ~a" (hash-ref c 'name)))
    (define ps (parameters-with (hash-ref c 'grads_in) (hash-ref c 'dtypes)))
    (define total
      (clip-grad-norm! ps (hash-ref c 'max_norm)
                       #:norm-type (number-of (hash-ref c 'norm_type))
                       #:foreach? (hash-ref c 'foreach)))
    (check-= (item total) (hash-ref c 'norm) tol (format "~a: norm" label))
    (check-grads ps (hash-ref c 'grads_out) label (lambda (x y m) (check-= x y tol m))))

  (define (check-value-case c)
    (define label (format "G0-L7: clip_grad_value_ ~a" (hash-ref c 'name)))
    (define ps (parameters-with (hash-ref c 'grads_in) (hash-ref c 'dtypes)))
    (clip-grad-value! ps (hash-ref c 'clip_value) #:foreach? (hash-ref c 'foreach))
    (check-grads ps (hash-ref c 'grads_out) label (lambda (x y m) (check-equal? x y m))))

  (cond
    [(not (python-torch-available?))
     (displayln "[clip-grad-parity-test] skipped: python3 `torch` not available")]
    [else
     (define j (python-check "clip_grad.py"))
     (for-each check-norm-case (hash-ref j 'norm))
     (for-each check-value-case (hash-ref j 'value))]))
