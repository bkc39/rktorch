#lang racket/base

(require (only-in racket/contract/base ->) ;; noqa
         (only-in "../foreign.rkt"
                  copy! requires-grad! requires-grad? with-no-grad zeros-like)
         (only-in "../private/contract.rkt" define/checked-out)
         (only-in "buffer.rkt" Buffer Buffer?)
         (only-in "layer.rkt" layer-rebuild layer?)
         (only-in "parameter.rkt" Parameter Parameter?))

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
  (define layers (make-hasheq))
  (define tensors (make-hasheq))
  (let copy-of ([m m])
    (hash-ref! layers m
               (lambda ()
                 (layer-rebuild m
                                (lambda (_name child) (copy-of child))
                                (lambda (t)
                                  (hash-ref! tensors t
                                             (lambda () (fresh-tensor t)))))))))
