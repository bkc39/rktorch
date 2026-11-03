#lang racket/base

(require (only-in racket/contract/base ->) ;; noqa
         (only-in "../foreign.rkt"
                  copy! requires-grad! requires-grad? with-no-grad zeros-like)
         (only-in "../private/contract.rkt" define/checked-out)
         (only-in "buffer.rkt" Buffer Buffer?)
         (only-in "layer.rkt" layer-rebuild layer?)
         (only-in "parameter.rkt" Parameter Parameter?)
         (only-in "private/field-walk.rkt" copy-child refuse-cycles))

(define (fresh-tensor t)
  (define copy (zeros-like t))
  (with-no-grad (copy! copy t))
  (cond
    [(Parameter? t) (requires-grad! (Parameter copy) (requires-grad? t))]
    [(Buffer? t) (Buffer copy)]
    [(requires-grad? t) (requires-grad! copy)]
    [else copy]))

(define/checked-out (layer-copy m) ;; noqa
  (-> layer? layer?)
  (refuse-cycles m)
  (define memo (make-hasheq))
  (define (copy-tensor t)
    (hash-ref! memo t (lambda () (fresh-tensor t))))
  (define copy-of
    (copy-child memo
                (lambda (_name c)
                  (hash-ref! memo c (lambda () (layer-rebuild c copy-of copy-tensor))))))
  (copy-of #f m))
