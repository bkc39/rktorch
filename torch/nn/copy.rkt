#lang racket/base

(require (only-in racket/contract/base ->) ;; noqa
         (only-in "../foreign.rkt"
                  copy! requires-grad! requires-grad? with-no-grad zeros-like)
         (only-in "../private/contract.rkt" define/checked-out)
         (only-in "buffer.rkt" Buffer Buffer?)
         (only-in "layer.rkt" layer?)
         (only-in "parameter.rkt" Parameter Parameter?)
         (only-in "private/layer-core.rkt" layer-rebuild-alone layer-rebuild-in)
         (only-in "private/memo.rkt" remembered remembering with-memo))

(define (fresh-tensor t)
  (define copy (zeros-like t))
  (with-no-grad (copy! copy t))
  (cond
    [(Parameter? t) (requires-grad! (Parameter copy) (requires-grad? t))]
    [(Buffer? t) (Buffer copy)]
    [(requires-grad? t) (requires-grad! copy)]
    [else copy]))

(define ((tensor-copier seen) t)
  (remembered seen t (lambda () (fresh-tensor t))))

(define (copy-child seen child _name c)
  (layer-rebuild-in c child (tensor-copier seen)))

(define/checked-out (layer-copy m) ;; noqa
  (-> layer? layer?)
  (with-memo (seen 'layer-copy)
    (layer-rebuild-alone m (remembering seen copy-child) (tensor-copier seen))))
