#lang racket/base

(require (only-in racket/contract/base ->) ;; noqa
         (only-in "../foreign.rkt"
                  copy! requires-grad! requires-grad? with-no-grad zeros-like)
         (only-in "../private/contract.rkt" define/checked-out)
         (only-in "buffer.rkt" Buffer Buffer?)
         (only-in "layer.rkt" layer?)
         (only-in "parameter.rkt" Parameter Parameter?)
         (only-in "private/field-walk.rkt" refuse-cycles)
         (only-in "private/layer-core.rkt" layer-rebuild-in)
         (only-in "private/memo.rkt" memo remembered remembering))

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
  (refuse-cycles 'layer-copy m)
  (define seen (memo 'layer-copy))
  (define (copy-tensor t)
    (remembered seen t (lambda () (fresh-tensor t))))
  (define (rebuilt c)
    (layer-rebuild-in seen c copy-child copy-tensor))
  (define copy-child
    (remembering seen
                 (lambda (name c)
                   (remembered seen c (lambda () (rebuilt c)) #:field name))))
  (remembered seen m (lambda () (rebuilt m))))
