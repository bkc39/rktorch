#lang racket/base

;; What each validation strategy costs.  Every strategy is a submodule so
;; that a call from `main` crosses a module boundary and a submodule's own
;; `intra-*` loop does not; flattening them would measure nothing.
;; Every callee is imported from its defining module, never from the torch
;; facade, which contracts them -- otherwise the bare row already pays a
;; crossing and the column measures a second copy of the same contract.
;; Run:  racket scripts/bench-contract-overhead.rkt

(module loops racket/base
  (provide define-intra-loops)
  ;; The four intra loops differ only in which wrapper they call, and each
  ;; must stay shaped exactly like its cross counterpart in `main`.
  (define-syntax-rule
    (define-intra-loops [intra-scale scale] [intra-add add-wrap]
                        [intra-heads split-heads] [intra-shape shape-wrap])
    (begin
      (define (intra-scale reps x k)
        (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (scale x k))))
      (define (intra-add reps a b)
        (for ([_ (in-range reps)]) (add-wrap a b)))
      (define (intra-heads reps n h)
        (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (split-heads n h))))
      (define (intra-shape reps ts _unused)
        (for/fold ([acc 0]) ([i (in-range reps)])
          (+ acc (length (shape-wrap (vector-ref ts (bitwise-and i 7))))))))))

(module bare racket/base
  (require (submod ".." loops))
  (provide scale add-wrap split-heads shape-wrap
           intra-scale intra-add intra-heads intra-shape)
  (require (only-in (file "../torch/foreign/ops.rkt") tensor-shape)
           (only-in (file "../torch/foreign/tensor-ops.rkt") add))

  (define (scale x k) (* x k))
  (define (add-wrap a b) (add a b))
  (define (split-heads n h) (quotient n h))


  (define (shape-wrap t) (tensor-shape t))
  (define-intra-loops [intra-scale scale] [intra-add add-wrap]
                      [intra-heads split-heads] [intra-shape shape-wrap]))

(module guards racket/base
  (require (submod ".." loops))
  (provide scale add-wrap split-heads shape-wrap
           intra-scale intra-add intra-heads intra-shape)
  (require           (only-in (file "../torch/foreign/ops.rkt") tensor-shape)
           (only-in (file "../torch/foreign/structs.rkt") tensor?)
           (only-in (file "../torch/foreign/tensor-ops.rkt") add))

  (define (scale x k)
    (unless (real? x) (error 'scale "x must be real: ~e" x))
    (unless (real? k) (error 'scale "k must be real: ~e" k))
    (* x k))
  (define (add-wrap a b)
    (unless (tensor? a) (error 'add-wrap "a must be a tensor: ~e" a))
    (unless (tensor? b) (error 'add-wrap "b must be a tensor: ~e" b))
    (add a b))
  (define (split-heads n h)
    (unless (exact-positive-integer? n) (error 'split-heads "n: ~e" n))
    (unless (exact-positive-integer? h) (error 'split-heads "h: ~e" h))
    (unless (zero? (remainder n h)) (error 'split-heads "~a % ~a" n h))
    (quotient n h))


  (define (shape-wrap t)
    (unless (tensor? t) (error 'shape-wrap "t must be a tensor: ~e" t))
    (tensor-shape t))
  (define-intra-loops [intra-scale scale] [intra-add add-wrap]
                      [intra-heads split-heads] [intra-shape shape-wrap]))

(module defcontract racket/base
  (require (submod ".." loops))
  (provide scale add-wrap split-heads shape-wrap
           intra-scale intra-add intra-heads intra-shape)
  ;; define/contract is not in racket/contract/base; raco review's
  ;; "prefer base" warning here cannot be satisfied
  (require (only-in racket/contract define/contract)
           (only-in racket/contract/base -> ->i listof)
                     (only-in (file "../torch/foreign/ops.rkt") tensor-shape)
           (only-in (file "../torch/foreign/structs.rkt") tensor?)
           (only-in (file "../torch/foreign/tensor-ops.rkt") add))

  (define/contract (scale x k) (-> real? real? real?) (* x k))
  (define/contract (add-wrap a b) (-> tensor? tensor? tensor?) (add a b))
  (define/contract (split-heads n h)
    (->i ([n exact-positive-integer?] [h exact-positive-integer?])
         #:pre (n h) (zero? (remainder n h))
         [result exact-positive-integer?])
    (quotient n h))


  (define/contract (shape-wrap t)
    (-> tensor? (listof exact-nonnegative-integer?))
    (tensor-shape t))
  (define-intra-loops [intra-scale scale] [intra-add add-wrap]
                      [intra-heads split-heads] [intra-shape shape-wrap]))

(module boundary racket/base
  (require (submod ".." loops))
  (provide intra-scale intra-add intra-heads intra-shape)
  (require (only-in racket/contract/base -> ->i listof)
           (only-in (file "../torch/foreign/ops.rkt") tensor-shape)
           (only-in (file "../torch/foreign/structs.rkt") tensor?)
           (only-in (file "../torch/foreign/tensor-ops.rkt") add)
           (only-in (file "../torch/private/contract.rkt") define/contract-out))

  (define/contract-out (scale x k) (-> real? real? real?) (* x k))
  (define/contract-out (add-wrap a b) (-> tensor? tensor? tensor?) (add a b))
  (define/contract-out (split-heads n h)
    (->i ([n exact-positive-integer?] [h exact-positive-integer?])
         #:pre (n h) (zero? (remainder n h))
         [result exact-positive-integer?])
    (quotient n h))


  (define/contract-out (shape-wrap t)
    (-> tensor? (listof exact-nonnegative-integer?))
    (tensor-shape t))
  (define-intra-loops [intra-scale scale] [intra-add add-wrap]
                      [intra-heads split-heads] [intra-shape shape-wrap]))

(module+ main
  (require racket/format
           (only-in torch manual-seed! randn reclaim-native-memory!
                    [add torch-add])
           (only-in (file "../torch/foreign/tensor-ops.rkt") [add raw-add])
           (only-in torch/audio/data load-audio-fixture)
           (only-in torch/audio/functional log-mel-spectrogram)
           (prefix-in bare: (submod ".." bare))
           (prefix-in guards: (submod ".." guards))
           (prefix-in dc: (submod ".." defcontract))
           (prefix-in bo: (submod ".." boundary)))

  ;; The cross loops must mirror the intra ones exactly: same accumulate,
  ;; same arguments-as-parameters.  Literal arguments let the compiler
  ;; constant-fold an uncontracted callee away entirely.
  (define-syntax-rule (scale-loop f)
    (lambda (reps x k)
      (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (f x k)))))
  (define-syntax-rule (heads-loop f)
    (lambda (reps n h)
      (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (f n h)))))
  (define-syntax-rule (add-loop f)
    (lambda (reps a b) (for ([_ (in-range reps)]) (f a b))))
  ;; a pure accessor on a loop-invariant tensor is hoisted out of the loop
  ;; entirely -- the argument has to vary or this measures nothing
  (define-syntax-rule (shape-loop f)
    (lambda (reps ts _unused)
      (for/fold ([acc 0]) ([i (in-range reps)])
        (+ acc (length (f (vector-ref ts (bitwise-and i 7))))))))

  (define (settle-jit! thunk) (thunk))

  (define (best-ms thunk #:runs [runs 7])
    (settle-jit! thunk)
    (for/fold ([best +inf.0]) ([_ (in-range runs)])
      ;; a plain collect does not wait on the finalizer queue, so native
      ;; frees can land after t0 and be charged to the wrong strategy
      (reclaim-native-memory!)
      (define t0 (current-inexact-milliseconds))
      (thunk)
      (min best (- (current-inexact-milliseconds) t0))))

  (define (row label reps thunk baseline)
    (define ns/call (/ (* (best-ms thunk) 1e6) reps))
    (printf "  ~a ~a ns/call~a\n"
            (~a label #:min-width 24)
            (~a (~r ns/call #:precision '(= 1)) #:min-width 9 #:align 'right)
            (if baseline
                (format "   ~ax" (~r (/ ns/call baseline) #:precision '(= 2)))
                "   (baseline)"))
    ns/call)

  (define strategies '("bare           " "unless+error   "
                       "define/contract" "contract-out   "))

  (define (workload name reps intra cross a b)
    (printf "\n~a  (~a reps, warmed, best of 7)\n" name reps)
    ;; every call site must be settled before the first timed row: the
    ;; one measured first would otherwise carry the cold start
    (for ([v (in-list (list intra cross))] [_ (in-naturals)])
      (for ([f (in-vector v)]) (f (quotient reps 10) a b)))
    (define base-intra
      (row (string-append (car strategies) " intra") reps
           (lambda () ((vector-ref intra 0) reps a b)) #f))
    (define base-cross
      (row (string-append (car strategies) " cross") reps
           (lambda () ((vector-ref cross 0) reps a b)) #f))
    (for ([s (in-list strategies)]
          [i (in-naturals)]
          #:unless (zero? i))
      (row (string-append s " intra") reps (lambda () ((vector-ref intra i) reps a b)) base-intra)
      (row (string-append s " cross") reps (lambda () ((vector-ref cross i) reps a b)) base-cross)))

  (displayln "== bench-contract-overhead (#96) ==")
  (manual-seed! 0)

  (workload "flat contract, non-tensor call" 1000000
            (vector bare:intra-scale guards:intra-scale
                    dc:intra-scale bo:intra-scale)
            (vector (scale-loop bare:scale) (scale-loop guards:scale)
                    (scale-loop dc:scale) (scale-loop bo:scale))
            1.5 2.0)

  (workload "->i with #:pre vs its hand guard" 1000000
            (vector bare:intra-heads guards:intra-heads
                    dc:intra-heads bo:intra-heads)
            (vector (heads-loop bare:split-heads) (heads-loop guards:split-heads)
                    (heads-loop dc:split-heads) (heads-loop bo:split-heads))
            32 4)

  (workload "tensor? contract on an 8x8 add" 50000
            (vector bare:intra-add guards:intra-add dc:intra-add bo:intra-add)
            (vector (add-loop bare:add-wrap) (add-loop guards:add-wrap)
                    (add-loop dc:add-wrap) (add-loop bo:add-wrap))
            (randn 8 8) (randn 8 8))

  (workload "tensor? contract on a cached shape read" 1000000
            (vector bare:intra-shape guards:intra-shape
                    dc:intra-shape bo:intra-shape)
            (vector (shape-loop bare:shape-wrap) (shape-loop guards:shape-wrap)
                    (shape-loop dc:shape-wrap) (shape-loop bo:shape-wrap))
            (build-vector 8 (lambda (_) (randn 8 8))) #f)

  ;; the production surface, not a stand-in: torch's add carries
  ;; binary-arith/c, a dependent ->i whose admissible type for b turns on
  ;; whether a is a tensor
  (displayln "\nthe shipped contract on add: facade vs raw callee")
  (let* ([u (randn 8 8)]
         [v (randn 8 8)]
         [n 50000]
         [raw-loop (lambda () (for ([_ (in-range n)]) (raw-add u v)))]
         [fac-loop (lambda () (for ([_ (in-range n)]) (torch-add u v)))]
         [_settled (begin (settle-jit! raw-loop) (settle-jit! fac-loop))]
         [raw (/ (* (best-ms raw-loop) 1e6) n)]
         [fac (/ (* (best-ms fac-loop) 1e6) n)])
    (printf "  raw tensor-ops add        ~a ns/call\n"
            (~r raw #:precision '(= 1)))
    (printf "  torch add (binary-arith/c) ~a ns/call   ~ax\n"
            (~r fac #:precision '(= 1))
            (~r (/ fac raw) #:precision '(= 2))))

  (displayln "\nreal pipeline: log-mel-spectrogram over the speech fixture")
  (define-values (samples rate) (load-audio-fixture))
  (define ms (best-ms (lambda () (log-mel-spectrogram samples #:sample-rate rate))))
  (printf "  ~a ms/call\n" (~r ms #:precision '(= 2))))
