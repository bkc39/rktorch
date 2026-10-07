#lang racket/base

;; Each strategy is a submodule, so a call from `contract-cases` crosses a
;; module boundary and a submodule's own intra loop does not. Callees come
;; from their defining modules, never the torch facade, which contracts them.

(module loops racket/base
  (require (for-syntax racket/base)
           ;; whole-module: the pattern's syntax classes live at phase 1
           syntax/parse/define)
  (provide intra-loops
           scale-loop heads-loop add-loop shape-loop)

  ;; a literal argument lets the compiler fold an uncontracted callee away,
  ;; and a loop-invariant accessor is hoisted: arguments arrive as parameters
  ;; and the shape loop varies its tensor
  (define-syntax-parse-rule (scale-loop f:id)
    (lambda (reps x k)
      (for/fold ([acc 0.0]) ([_ (in-range reps)]) (+ acc (f x k)))))
  (define-syntax-parse-rule (heads-loop f:id)
    (lambda (reps n h)
      (for/fold ([acc 0]) ([_ (in-range reps)]) (+ acc (f n h)))))
  (define-syntax-parse-rule (add-loop f:id)
    (lambda (reps a b) (for ([_ (in-range reps)]) (f a b))))
  (define-syntax-parse-rule (shape-loop f:id)
    (lambda (reps ts _unused)
      (for/fold ([acc 0]) ([i (in-range reps)])
        (+ acc (length (f (vector-ref ts (bitwise-and i 7))))))))

  (define-syntax-parse-rule (intra-loops scale:id add-wrap:id split-heads:id
                                         shape-wrap:id)
    (vector (scale-loop scale) (heads-loop split-heads) (add-loop add-wrap)
            (shape-loop shape-wrap))))

(module bare racket/base
  (require (only-in torch/foreign/ops tensor-shape)
           (only-in torch/foreign/tensor-ops add)
           (submod ".." loops))
  (provide intra scale add-wrap split-heads shape-wrap)
  (define (scale x k) (* x k))
  (define (add-wrap a b) (add a b))
  (define (split-heads n h) (quotient n h))
  (define (shape-wrap t) (tensor-shape t))
  (define intra (intra-loops scale add-wrap split-heads shape-wrap)))

(module guards racket/base
  (require (only-in torch/foreign/ops tensor-shape)
           (only-in torch/foreign/structs tensor?)
           (only-in torch/foreign/tensor-ops add)
           (submod ".." loops))
  (provide intra scale add-wrap split-heads shape-wrap)
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
  (define intra (intra-loops scale add-wrap split-heads shape-wrap)))

(module defcontract racket/base
  ;; define/contract is not in racket/contract/base
  (require (only-in racket/contract define/contract)
           (only-in racket/contract/base -> ->i listof)
           (only-in torch/foreign/ops tensor-shape)
           (only-in torch/foreign/structs tensor?)
           (only-in torch/foreign/tensor-ops add)
           (submod ".." loops))
  (provide intra scale add-wrap split-heads shape-wrap)
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
  (define intra (intra-loops scale add-wrap split-heads shape-wrap)))

(module boundary racket/base
  (require (only-in racket/contract/base -> ->i listof)
           (only-in torch/foreign/ops tensor-shape)
           (only-in torch/foreign/structs tensor?)
           (only-in torch/foreign/tensor-ops add)
           (only-in torch/private/contract define/contract-out)
           (submod ".." loops))
  (provide intra)
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
  (define intra (intra-loops scale add-wrap split-heads shape-wrap)))

(require (only-in torch randn reclaim-native-memory!)
         (only-in torch/audio/data load-audio-fixture)
         (only-in torch/audio/functional log-mel-spectrogram)
         (prefix-in bare: (submod "." bare))
         (prefix-in bo: (submod "." boundary))
         (prefix-in dc: (submod "." defcontract))
         (prefix-in guards: (submod "." guards))
         (submod "." loops)
         (only-in "harness.rkt" bench-case))

(provide contract-workloads
         pipeline-case)

(define strategies '(bare unless+error define/contract contract-out))

(define (workload name reps intra cross a b)
  (cons name
        (for*/list ([i (in-range (length strategies))]
                    [where (in-list '(intra cross))])
          (define f (vector-ref (if (eq? where 'intra) intra cross) i))
          (bench-case (string->symbol (format "~a/~a" (list-ref strategies i) where))
                      (lambda (n) (f n a b))
                      #:reps reps
                      #:before reclaim-native-memory!))))

(define (intra k)
  (for/vector ([loops (in-list (list bare:intra guards:intra dc:intra bo:intra))])
    (vector-ref loops k)))

(define (contract-workloads #:scale [scale 1])
  (define (n reps) (max 1 (inexact->exact (round (* reps scale)))))
  (list
   (workload 'flat-contract (n 1000000) (intra 0)
             (vector (scale-loop bare:scale) (scale-loop guards:scale)
                     (scale-loop dc:scale) (scale-loop bo:scale))
             1.5 2.0)
   (workload 'dependent-pre (n 1000000) (intra 1)
             (vector (heads-loop bare:split-heads) (heads-loop guards:split-heads)
                     (heads-loop dc:split-heads) (heads-loop bo:split-heads))
             32 4)
   (workload 'tensor-add-8x8 (n 50000) (intra 2)
             (vector (add-loop bare:add-wrap) (add-loop guards:add-wrap)
                     (add-loop dc:add-wrap) (add-loop bo:add-wrap))
             (randn 8 8) (randn 8 8))
   (workload 'cached-shape-read (n 1000000) (intra 3)
             (vector (shape-loop bare:shape-wrap) (shape-loop guards:shape-wrap)
                     (shape-loop dc:shape-wrap) (shape-loop bo:shape-wrap))
             (build-vector 8 (lambda (_) (randn 8 8))) #f)))

(define (pipeline-case)
  (define-values (samples rate) (load-audio-fixture))
  (bench-case 'log-mel-spectrogram
              (lambda (_n) (log-mel-spectrogram samples #:sample-rate rate))
              #:before reclaim-native-memory!))
