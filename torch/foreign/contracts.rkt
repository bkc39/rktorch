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

(define dims-rest/c (listof exact-nonnegative-integer?))

;; reshape/view accept -1 ("infer this dimension"), so plain integers.
(define index/c exact-integer?)

(define tensor-or-real/c (or/c tensor? real?))

;; A device struct or a legacy form — 'cpu, 'cuda (ordinal 0 shorthand),
;; (list 'cuda n); queries return device structs.
(define device/c
  (or/c device? 'cpu 'cuda (list/c 'cuda exact-nonnegative-integer?)))

;; At least one argument must be a tensor — (add 1 2) should get contract
;; blame, not a runtime error from the dispatcher.
(define binary-arith/c
  (->i ([a (or/c tensor? real?)]
        [b (a) (if (tensor? a) (or/c tensor? real?) tensor?)])
       [result tensor?]))

;; Shadow-dispatch unaries: tensors produce tensors, numbers numbers.
(define unary-numeric/c
  (->i ([v (or/c tensor? number?)])
       [result (v) (if (tensor? v) tensor? number?)]))

;; The optional base argument only makes sense for numbers; (log t 2) is
;; blamed at the boundary.
(define log/c
  (->i ([v (or/c tensor? number?)])
       ([base (v) (if (tensor? v) none/c number?)])
       [result (v) (if (tensor? v) tensor? number?)]))

;; A single tensor reduces; reals behave like racket/base's variadic
;; max/min. Extra arguments after a tensor are blamed.
(define reduce-or-variadic/c
  (->i ([v (or/c tensor? real?)])
       #:rest [rest (v) (if (tensor? v) null? (listof real?))]
       [result (v) (if (tensor? v) tensor? real?)]))

;; Tensor form takes an optional dim + #:keepdim; procedure form is
;; racket/list's (argmax proc lst), where the list is mandatory.
(define argmax/c
  (->i ([v (or/c tensor? procedure?)])
       ([dim (v) (if (tensor? v) index/c list?)]
        #:keepdim [keepdim (v) (if (tensor? v) boolean? none/c)])
       #:pre/name (v dim) "a list argument is required with a procedure"
       (or (tensor? v) (not (unsupplied-arg? dim)))
       [result any/c]))

;; An int (broadcast to a square) or an [h w] list. list/c, not listof:
;; ATen rejects non-2-element lists, so blame at the facade boundary.
(define pool-size/c (or/c index/c (list/c index/c index/c)))

;; Stricter size args for the nn layer constructors: kernel/stride are
;; positive, padding may be 0 (pool-size/c is the looser surface).
(define pos-size/c
  (or/c exact-positive-integer?
        (list/c exact-positive-integer? exact-positive-integer?)))
(define nonneg-size/c
  (or/c exact-nonnegative-integer?
        (list/c exact-nonnegative-integer? exact-nonnegative-integer?)))

(define compare/c (-> tensor? tensor-or-real/c tensor?))

;; A tensor collapses dims (optional start/end); anything else defers to
;; racket/list's flatten, which takes its one value.
(define flatten/c
  (->i ([v any/c])
       ([start (v) (if (tensor? v) index/c none/c)]
        [end (v) (if (tensor? v) index/c none/c)])
       [result any/c]))

(define arange/c
  (case-> (-> real? tensor?)
          (-> real? real? tensor?)
          (-> real? real? real? tensor?)))
