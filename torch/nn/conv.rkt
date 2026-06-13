#lang racket/base

;; Conv/pool layers as gen:module values, mirroring nn.Conv2d / nn.MaxPool2d /
;; nn.Flatten. conv2d holds weight+bias and matches nn.Conv2d.reset_parameters
;; (kaiming-uniform a=sqrt5 for the weight, then bias uniform on +/-1/sqrt(fan_in),
;; in that order) so a shared seed yields parameters bit-identical to PyTorch.
;; max-pool2d/flatten are stateless config holders whose forward defers to the
;; functional ops in the facade; they exist as modules so `sequential` can hold
;; them.

;; The functional ops are renamed f:* so the layer constructors below can keep
;; the PyTorch names (nn.Conv2d vs F.conv2d) without shadowing themselves.
(require (only-in "../foreign.rkt"
                  [conv2d f:conv2d]
                  [flatten f:flatten]
                  [max-pool2d f:max-pool2d])
         (only-in "../foreign/size.rkt" ->2d)
         (only-in "init.rkt" fan-in kaiming-uniform uniform-init)
         (only-in "module.rkt" define-module))

;; conv2d?/max-pool2d?/flatten? are produced by the define-module expansions
;; below (the conv2d%/... structs), invisible to raco review without expansion.
(provide conv2d
         conv2d? ;; noqa
         max-pool2d
         max-pool2d? ;; noqa
         flatten
         flatten? ;; noqa
         )

;; ------------------------------------------------------------------- conv2d

;; kernel-size/stride/padding arrive already normalized to [h w] lists from the
;; smart constructor, so the weight shape and fan-in are straightforward.
(define-module conv2d% (in-channels out-channels kernel-size stride padding)
  #:params ([weight (kaiming-uniform (list out-channels in-channels
                                            (car kernel-size) (cadr kernel-size)))]
            [bias (let ([bound (/ 1.0 (sqrt (fan-in (list out-channels in-channels
                                                          (car kernel-size)
                                                          (cadr kernel-size)))))])
                    (uniform-init (list out-channels) (- bound) bound))])
  #:forward (x)
  (f:conv2d x weight #:bias bias #:stride stride #:padding padding))

(define (conv2d in-channels out-channels kernel-size
                #:stride [stride 1]
                #:padding [padding 0])
  (conv2d% in-channels out-channels
           (->2d kernel-size) (->2d stride) (->2d padding)))

(define conv2d? conv2d%?)

;; ---------------------------------------------------------------- max-pool2d

;; Stateless; stride #f means "default to kernel-size", matching nn.MaxPool2d.
(define-module max-pool2d% (kernel-size stride padding)
  #:forward (x)
  (f:max-pool2d x kernel-size #:stride stride #:padding padding))

(define (max-pool2d kernel-size #:stride [stride #f] #:padding [padding 0])
  (max-pool2d% kernel-size stride padding))

(define max-pool2d? max-pool2d%?)

;; ------------------------------------------------------------------- flatten

;; nn.Flatten defaults to start_dim=1, keeping the batch dim.
(define-module flatten% (start-dim end-dim)
  #:forward (x)
  (f:flatten x start-dim end-dim))

(define (flatten #:start-dim [start-dim 1] #:end-dim [end-dim -1])
  (flatten% start-dim end-dim))

(define flatten? flatten%?)
