#lang racket/base

;; Contract combinators for the public facade. foreign.rkt imports these
;; and applies them in its contract-out provide; keeping the definitions
;; here keeps the facade a pure surface description.

(require (only-in racket/contract/base
                  ->
                  ->i
                  any/c
                  case->
                  list/c
                  listof
                  none/c
                  or/c
                  unsupplied-arg?)
         (only-in "device-type.rkt" device?)
         (only-in "structs.rkt" tensor?))

(provide dims-rest/c
         index/c
         device/c
         tensor-or-real/c
         pool-size/c
         pos-size/c
         nonneg-size/c
         binary-arith/c
         unary-numeric/c
         log/c
         reduce-or-variadic/c
         argmax/c
         compare/c
         flatten/c
         arange/c)

;; Shape arguments for the variadic creation ops.
(define dims-rest/c (listof exact-nonnegative-integer?))

;; reshape/view accept -1 ("infer this dimension"), so plain integers.
(define index/c exact-integer?)

(define tensor-or-real/c (or/c tensor? real?))

;; A device argument: a first-class device struct (device-type.rkt), or a
;; legacy form — 'cpu, 'cuda (ordinal 0 shorthand), (list 'cuda n) — all
;; accepted everywhere, the PyTorch pattern of device objects beside
;; strings. Queries (default-device/tensor-device) return device structs.
(define device/c
  (or/c device? 'cpu 'cuda (list/c 'cuda exact-nonnegative-integer?)))

;; Binary arithmetic: a real is welcome on either side, but at least one
;; argument must be a tensor — (add 1 2) is a caller error and should get
;; contract blame, not a runtime error from the dispatcher.
(define binary-arith/c
  (->i ([a (or/c tensor? real?)]
        [b (a) (if (tensor? a) (or/c tensor? real?) tensor?)])
       [result tensor?]))

;; Shadow-dispatch unaries (exp sqrt tanh): tensors produce tensors,
;; numbers defer to racket/base and produce numbers.
(define unary-numeric/c
  (->i ([v (or/c tensor? number?)])
       [result (v) (if (tensor? v) tensor? number?)]))

;; log: racket/base's optional base argument only makes sense for numbers;
;; (log some-tensor 2) is blamed at the boundary.
(define log/c
  (->i ([v (or/c tensor? number?)])
       ([base (v) (if (tensor? v) none/c number?)])
       [result (v) (if (tensor? v) tensor? number?)]))

;; max/min: a single tensor reduces; reals behave like racket/base's
;; variadic max/min. Extra arguments after a tensor are blamed.
(define reduce-or-variadic/c
  (->i ([v (or/c tensor? real?)])
       #:rest [rest (v) (if (tensor? v) null? (listof real?))]
       [result (v) (if (tensor? v) tensor? real?)]))

;; argmax: tensor form takes an optional dim + #:keepdim; procedure form is
;; racket/list's (argmax proc lst) and the list is mandatory there.
(define argmax/c
  (->i ([v (or/c tensor? procedure?)])
       ([dim (v) (if (tensor? v) index/c list?)]
        #:keepdim [keepdim (v) (if (tensor? v) boolean? none/c)])
       #:pre/name (v dim) "a list argument is required with a procedure"
       (or (tensor? v) (not (unsupplied-arg? dim)))
       [result any/c]))

;; A conv/pool size argument: an int (broadcast to a square) or an [h w] list.
;; list/c, not listof: ATen rejects non-2-element lists, so blame at the
;; facade boundary rather than deep inside at::conv2d.
(define pool-size/c (or/c index/c (list/c index/c index/c)))

;; Stricter size args for the nn layer constructors (nn.Conv2d/MaxPool2d):
;; kernel/stride are positive, padding may be 0. (pool-size/c above is the
;; looser functional surface where -1-style sentinels can flow through.)
(define pos-size/c
  (or/c exact-positive-integer?
        (list/c exact-positive-integer? exact-positive-integer?)))
(define nonneg-size/c
  (or/c exact-nonnegative-integer?
        (list/c exact-nonnegative-integer? exact-nonnegative-integer?)))

;; eq/ne/lt/le/gt/ge: tensor lhs, tensor-or-real rhs, tensor (float mask) out.
(define compare/c (-> tensor? tensor-or-real/c tensor?))

;; flatten shadow-dispatches: a tensor collapses dims (optional start/end);
;; anything else defers to racket/list's flatten, which takes its one value.
(define flatten/c
  (->i ([v any/c])
       ([start (v) (if (tensor? v) index/c none/c)]
        [end (v) (if (tensor? v) index/c none/c)])
       [result any/c]))

;; arange mirrors torch.arange's three arities.
(define arange/c
  (case-> (-> real? tensor?)
          (-> real? real? tensor?)
          (-> real? real? real? tensor?)))
