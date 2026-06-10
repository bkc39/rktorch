#lang racket/base

;; raco review does surface-level linting without macro expansion, so it cannot
;; see that every `contract-out` entry below re-exports an imported identifier —
;; it would report each as "provided but not defined".  This is a pure
;; re-export facade with nothing else to lint.
#|review: ignore|#

;; Facade for the safe, contracted FFI layer.  `(require torchrkt/foreign)`
;; exposes the low-level surface; all implementation lives in foreign/*.  A
;; `tensor` is a wrapper struct over a cpointer whose underlying handle is
;; GC-reclaimed.
;;
;; `(require (submod torchrkt/foreign unsafe))` additionally exposes
;; `tensor-free!` for callers that need deterministic release.  It is
;; idempotent: a second call hits the cpointer tag guard and raises
;; `exn:fail:contract` instead of double-freeing.
;;
;; Contracts are applied here, at the facade boundary, so this file is the
;; single authoritative description of the public surface.

(require ffi/vector
         racket/contract
         "foreign/structs.rkt"
         "foreign/ops.rkt"
         "foreign/tensor-ops.rkt"
         "foreign/autograd-ops.rkt")

;; with-no-grad is a macro (a dynamic-extent form), so it bypasses
;; contract-out; its expansion bottoms out in the contracted procedures.
(provide with-no-grad)

;; Shape arguments and the variadic creation ops.
(define dims-rest/c (listof exact-nonnegative-integer?))
;; reshape/view accept -1 ("infer this dimension"), so plain integers.
(define index/c exact-integer?)
(define tensor-or-real/c (or/c tensor? real?))

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

(provide
 (contract-out
  [torch-version (-> string?)]
  [manual-seed! (-> exact-nonnegative-integer? void?)]
  [randn (->* () #:rest dims-rest/c tensor?)]
  [rand (->* () #:rest dims-rest/c tensor?)]
  [uniform! (-> tensor? real? real? void?)]
  [tensor? (-> any/c boolean?)]
  [tensor-shape (-> tensor? (listof exact-nonnegative-integer?))]
  [tensor-numel (-> tensor? exact-nonnegative-integer?)]
  [tensor->vector (-> tensor? f32vector?)]
  [tensor->list (-> tensor? (listof real?))]
  ;; tensor->repr: the PyTorch `repr` text (what the REPL prints);
  ;; tensor->string: ATen's C++ `operator<<` text.
  [tensor->repr (-> tensor? string?)]
  [tensor->string (-> tensor? string?)]
  ;; creation
  [zeros (->* () #:rest dims-rest/c tensor?)]
  [ones (->* () #:rest dims-rest/c tensor?)]
  [full (->* (real?) #:rest dims-rest/c tensor?)]
  [arange (case-> (-> real? tensor?)
                  (-> real? real? tensor?)
                  (-> real? real? real? tensor?))]
  [eye (->* (exact-nonnegative-integer?)
            (exact-nonnegative-integer?)
            tensor?)]
  [tensor (-> (or/c real? list?) tensor?)]
  ;; shape
  [reshape (->* (tensor?) #:rest (listof index/c) tensor?)]
  [view (->* (tensor?) #:rest (listof index/c) tensor?)]
  [transpose (-> tensor? index/c index/c tensor?)]
  [permute (->* (tensor?) #:rest (listof index/c) tensor?)]
  [squeeze (->* (tensor?) (index/c) tensor?)]
  [unsqueeze (-> tensor? index/c tensor?)]
  [cat (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  [stack (->* ((non-empty-listof tensor?)) (index/c) tensor?)]
  ;; elementwise (binary ops take a real on either side, tensor required
  ;; on at least one)
  [add binary-arith/c]
  [sub binary-arith/c]
  [mul binary-arith/c]
  [div binary-arith/c]
  [pow (-> tensor? tensor-or-real/c tensor?)]
  [neg (-> tensor? tensor?)]
  [relu (-> tensor? tensor?)]
  [sigmoid (-> tensor? tensor?)]
  ;; exp/log/sqrt/tanh/max/min shadow racket/base: tensors hit libtorch,
  ;; anything else defers to the racket/base function, so requiring
  ;; torchrkt never breaks numeric code.
  [exp unary-numeric/c]
  [log log/c]
  [sqrt unary-numeric/c]
  [tanh unary-numeric/c]
  [max reduce-or-variadic/c]
  [min reduce-or-variadic/c]
  ;; reductions
  [sum (-> tensor? tensor?)]
  [mean (-> tensor? tensor?)]
  ;; argmax shadows racket/list's: (argmax proc lst) delegates to it.
  [argmax argmax/c]
  [softmax (-> tensor? index/c tensor?)]
  [log-softmax (-> tensor? index/c tensor?)]
  ;; linalg
  [matmul (-> tensor? tensor? tensor?)]
  [mm (-> tensor? tensor? tensor?)]
  [mv (-> tensor? tensor? tensor?)]
  [dot (-> tensor? tensor? tensor?)]
  ;; out-marshalling
  [item (-> tensor? real?)]
  [to-dtype (-> tensor? (or/c 'float32 'float64 'int64) tensor?)]
  ;; autograd
  [requires-grad! (->* (tensor?) (boolean?) tensor?)]
  [requires-grad? (-> tensor? boolean?)]
  [backward! (-> tensor? void?)]
  [grad (-> tensor? tensor?)]
  [has-grad? (-> tensor? boolean?)]
  [maybe-grad (-> tensor? (or/c tensor? #f))]
  [detach (-> tensor? tensor?)]
  [grad-enabled? (-> boolean?)]
  [call-with-no-grad (-> (-> any) any)]
  ;; in-place update primitives (use under with-no-grad, like torch.optim)
  [sub! (->* (tensor? tensor?) (real?) void?)]
  [zero! (-> tensor? void?)]
  [mul! (-> tensor? real? void?)]
  [zero-grad! (-> tensor? void?)]))

;; Explicit-free helper for deterministic release.  See the file comment.
(module+ unsafe
  (provide
   (contract-out
    [tensor-free! (-> tensor? void?)])))
