#lang racket/base

;; Conv/pool layers as gen:module values, mirroring nn.Conv2d / nn.MaxPool2d /
;; nn.Flatten. conv2d holds weight+bias and matches nn.Conv2d.reset_parameters
;; (kaiming-uniform a=sqrt5 for the weight, then bias uniform on +/-1/sqrt(fan_in),
;; in that order) so a shared seed yields parameters bit-identical to PyTorch.
;; max-pool2d/flatten are stateless config holders whose forward defers to the
;; functional ops in the facade; they exist as modules so `sequential` can hold
;; them.

;; The layer constructors are PascalCase (Conv2d/MaxPool2d/Flatten), mirroring
;; the torch.nn.* classes; the lowercase functional ops they call (conv2d/
;; max-pool2d/flatten on `torch`) are imported directly — the differing case
;; means no shadowing, and `(require torch torch/nn)` stays collision-free (#11).
(require (only-in "../foreign.rkt" conv2d flatten max-pool2d)
         (only-in "../foreign/size.rkt" ->2d)
         (only-in "init.rkt" fan-in kaiming-uniform uniform-init)
         (only-in "module.rkt" define-module))

;; Constructors are PascalCase, predicates lowercase (the linear.rkt
;; convention). Each layer is a single define-module form — keyword defaults
;; and size coercion live in the macro's ctor formals + #:coerce (#10), no
;; `%` struct or hand-written smart constructor. The definitions are macro
;; expansions, invisible to raco review; predicates export via rename-out.
(provide Conv2d
         (rename-out [Conv2d? conv2d?]) ;; noqa
         MaxPool2d
         (rename-out [MaxPool2d? max-pool2d?]) ;; noqa
         Flatten
         (rename-out [Flatten? flatten?]) ;; noqa
         )

;; ------------------------------------------------------------------- conv2d

(define-module Conv2d (in-channels out-channels kernel-size
                       #:stride [stride 1]
                       #:padding [padding 0])
  #:coerce ([kernel-size (->2d kernel-size)]
            [stride (->2d stride)]
            [padding (->2d padding)])
  #:params ([weight (kaiming-uniform (list out-channels in-channels
                                            (car kernel-size) (cadr kernel-size)))]
            [bias (let ([bound (/ 1.0 (sqrt (fan-in (list out-channels in-channels
                                                          (car kernel-size)
                                                          (cadr kernel-size)))))])
                    (uniform-init (list out-channels) (- bound) bound))])
  #:forward (x)
  (conv2d x weight #:bias bias #:stride stride #:padding padding))

;; ---------------------------------------------------------------- max-pool2d

;; Stateless; stride #f means "default to kernel-size", matching nn.MaxPool2d
;; (the functional max-pool2d handles the #f and the size normalization).
(define-module MaxPool2d (kernel-size
                          #:stride [stride #f]
                          #:padding [padding 0])
  #:forward (x)
  (max-pool2d x kernel-size #:stride stride #:padding padding))

;; ------------------------------------------------------------------- flatten

;; nn.Flatten defaults to start_dim=1, keeping the batch dim.
(define-module Flatten (#:start-dim [start-dim 1] #:end-dim [end-dim -1])
  #:forward (x)
  (flatten x start-dim end-dim))
