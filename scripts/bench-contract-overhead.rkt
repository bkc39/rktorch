#lang racket/base

;; Which validation strategy costs what, for #96.  Each strategy is a
;; submodule, so a call from `main` really does cross a module boundary
;; while the submodule's own `intra-*` loop really does not -- that
;; difference is the whole question.
;;
;; Run:  racket scripts/bench-contract-overhead.rkt

(module bare racket/base
  (provide scale add-wrap split-heads intra-scale intra-add intra-heads)
  (require (only-in torch add))

  (define (scale x k) (* x k))
  (define (add-wrap a b) (add a b))
  (define (split-heads n h) (quotient n h))

  (define (intra-scale reps x k)
    (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (scale x k))))
  (define (intra-add reps a b)
    (for ([_ (in-range reps)]) (add-wrap a b)))
  (define (intra-heads reps n h)
    (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (split-heads n h)))))

(module guards racket/base
  (provide scale add-wrap split-heads intra-scale intra-add intra-heads)
  (require (only-in torch add tensor?))

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

  (define (intra-scale reps x k)
    (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (scale x k))))
  (define (intra-add reps a b)
    (for ([_ (in-range reps)]) (add-wrap a b)))
  (define (intra-heads reps n h)
    (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (split-heads n h)))))

(module defcontract racket/base
  (provide scale add-wrap split-heads intra-scale intra-add intra-heads)
  ;; define/contract is not in racket/contract/base; raco review's
  ;; "prefer base" warning here cannot be satisfied
  (require (only-in racket/contract define/contract)
           (only-in racket/contract/base -> ->i)
           (only-in torch add tensor?))

  (define/contract (scale x k) (-> real? real? real?) (* x k))
  (define/contract (add-wrap a b) (-> tensor? tensor? tensor?) (add a b))
  (define/contract (split-heads n h)
    (->i ([n exact-positive-integer?] [h exact-positive-integer?])
         #:pre (n h) (zero? (remainder n h))
         [result exact-positive-integer?])
    (quotient n h))

  (define (intra-scale reps x k)
    (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (scale x k))))
  (define (intra-add reps a b)
    (for ([_ (in-range reps)]) (add-wrap a b)))
  (define (intra-heads reps n h)
    (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (split-heads n h)))))

(module boundary racket/base
  (provide intra-scale intra-add intra-heads)
  (require (only-in racket/contract/base -> ->i)
           (only-in torch add tensor?)
           (only-in (file "../torch/private/contract.rkt") define/contract-out))

  (define/contract-out (scale x k) (-> real? real? real?) (* x k))
  (define/contract-out (add-wrap a b) (-> tensor? tensor? tensor?) (add a b))
  (define/contract-out (split-heads n h)
    (->i ([n exact-positive-integer?] [h exact-positive-integer?])
         #:pre (n h) (zero? (remainder n h))
         [result exact-positive-integer?])
    (quotient n h))

  (define (intra-scale reps x k)
    (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (scale x k))))
  (define (intra-add reps a b)
    (for ([_ (in-range reps)]) (add-wrap a b)))
  (define (intra-heads reps n h)
    (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (split-heads n h)))))

(module+ main
  (require racket/format
           (only-in torch manual-seed! randn)
           (only-in torch/audio/data load-audio-fixture)
           (only-in torch/audio/functional log-mel-spectrogram)
           (prefix-in bare: (submod ".." bare))
           (prefix-in guards: (submod ".." guards))
           (prefix-in dc: (submod ".." defcontract))
           (prefix-in bo: (submod ".." boundary)))

  ;; The cross loops must mirror the intra ones exactly: same accumulate,
  ;; same arguments-as-parameters.  Literal arguments let the compiler
  ;; constant-fold an uncontracted callee away entirely, which made the
  ;; baseline read 1.8 ns/call -- no call at all.
  (define-syntax-rule (scale-loop f)
    (lambda (reps x k)
      (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (f x k)))))
  (define-syntax-rule (heads-loop f)
    (lambda (reps n h)
      (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (f n h)))))
  (define-syntax-rule (add-loop f)
    (lambda (reps a b) (for ([_ (in-range reps)]) (f a b))))

  (define (best-ms thunk #:runs [runs 7])
    (thunk)                             ; warm up before the first timed run
    (for/fold ([best +inf.0]) ([_ (in-range runs)])
      (collect-garbage)
      (collect-garbage)
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
    ;; settle every call site before the first timed row -- otherwise
    ;; whichever strategy is measured first pays the cold-start and the
    ;; baseline reads slower than the guards it is supposed to beat
    (for ([v (in-list (list intra cross))] [_ (in-naturals)])
      (for ([f (in-vector v)]) (f (quotient reps 10) a b)))
    (define base
      (row (string-append (car strategies) " intra") reps
           (lambda () ((vector-ref intra 0) reps a b)) #f))
    (for ([s (in-list strategies)] [i (in-naturals)])
      (unless (zero? i)
        (row (string-append s " intra") reps
             (lambda () ((vector-ref intra i) reps a b)) base))
      (row (string-append s " cross") reps
           (lambda () ((vector-ref cross i) reps a b)) base)))

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

  (displayln "\nreal pipeline: log-mel-spectrogram over the speech fixture")
  (define-values (samples rate) (load-audio-fixture))
  (define ms (best-ms (lambda () (log-mel-spectrogram samples #:sample-rate rate))))
  (printf "  ~a ms/call\n" (~r ms #:precision '(= 2))))
