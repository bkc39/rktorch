#lang racket/base

(module loops racket/base
  (require (for-syntax racket/base syntax/parse)
           (only-in syntax/parse/define define-syntax-parse-rule))
  (provide intra-loops
           scale-loop heads-loop add-loop shape-loop)

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
    (hasheq 'scale (scale-loop scale) 'heads (heads-loop split-heads)
            'add (add-loop add-wrap) 'shape (shape-loop shape-wrap))))

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
  (define (scale x k) ;; noqa
    (unless (real? x) (error 'scale "x must be real: ~e" x))
    (unless (real? k) (error 'scale "k must be real: ~e" k))
    (* x k))
  (define (add-wrap a b) ;; noqa
    (unless (tensor? a) (error 'add-wrap "a must be a tensor: ~e" a))
    (unless (tensor? b) (error 'add-wrap "b must be a tensor: ~e" b))
    (add a b))
  (define (split-heads n h) ;; noqa
    (unless (exact-positive-integer? n) (error 'split-heads "n: ~e" n))
    (unless (exact-positive-integer? h) (error 'split-heads "h: ~e" h))
    (unless (zero? (remainder n h)) (error 'split-heads "~a % ~a" n h))
    (quotient n h))
  (define (shape-wrap t) ;; noqa
    (unless (tensor? t) (error 'shape-wrap "t must be a tensor: ~e" t))
    (tensor-shape t))
  (define intra (intra-loops scale add-wrap split-heads shape-wrap)))

(module defcontract racket/base
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

(require (for-syntax racket/base racket/syntax syntax/parse)
         (only-in racket/match match-lambda)
         (only-in racket/syntax format-symbol)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in torch device dtype numel randn reclaim-native-memory! shape)
         (prefix-in long:
                    (only-in torch tensor-device tensor-dtype tensor-numel tensor-shape))
         (only-in torch/audio/data load-audio-fixture)
         (only-in torch/audio/functional log-mel-spectrogram)
         (prefix-in bare: (submod "." bare))
         (prefix-in bo: (submod "." boundary))
         (prefix-in dc: (submod "." defcontract))
         (prefix-in guards: (submod "." guards))
         (submod "." loops)
         (only-in "harness.rkt" bench-case))

(provide contract-workloads
         getter-cases
         pipeline-case)

(define strategies '(bare unless+error define/contract contract-out))

(define-syntax-parse-rule (cross-loops loop:id f:id)
  #:with (g ...) (for/list ([prefix (in-list '(bare: guards: dc: bo:))])
                   (format-id #'f "~a~a" prefix #'f))
  (list (loop g) ...))

(define (intra key)
  (for/list ([loops (in-list (list bare:intra guards:intra dc:intra bo:intra))])
    (hash-ref loops key)))

(define (workload name reps intra cross a b)
  (cons name
        (for*/list ([(strategy intra-f cross-f) (in-parallel strategies intra cross)]
                    [(where f) (in-parallel '(intra cross) (list intra-f cross-f))])
          (bench-case (format-symbol "~a/~a" strategy where)
                      (lambda (n) (f n a b))
                      #:reps reps
                      #:before reclaim-native-memory!))))

(define (contract-workloads #:scale [scale 1])
  (define (n reps) (max 1 (inexact->exact (round (* reps scale)))))
  (list
   (workload 'flat-contract (n 1000000) (intra 'scale) (cross-loops scale-loop scale)
             1.5 2.0)
   (workload 'dependent-pre (n 1000000) (intra 'heads) (cross-loops heads-loop split-heads)
             32 4)
   (workload 'tensor-add-8x8 (n 50000) (intra 'add) (cross-loops add-loop add-wrap)
             (randn 8 8) (randn 8 8))
   (workload 'cached-shape-read (n 1000000) (intra 'shape) (cross-loops shape-loop shape-wrap)
             (build-vector 8 (lambda (_) (randn 8 8))) #f)))

(define (getter-cases #:reps [reps 200000])
  (define t (randn 8 8))
  (map (match-lambda
         [(cons name getter)
          (bench-case name
                      (lambda (n)
                        (for ([_ (in-range n)])
                          (getter t)))
                      #:reps reps)])
       (list (cons 'device device)
             (cons 'tensor-device long:tensor-device)
             (cons 'dtype dtype)
             (cons 'tensor-dtype long:tensor-dtype)
             (cons 'shape shape)
             (cons 'tensor-shape long:tensor-shape)
             (cons 'numel numel)
             (cons 'tensor-numel long:tensor-numel))))

(define (pipeline-case #:reps [reps 1])
  (define-values (samples rate) (load-audio-fixture))
  (bench-case 'log-mel-spectrogram
              (lambda (n)
                (for ([_ (in-range n)])
                  (log-mel-spectrogram samples #:sample-rate rate)))
              #:reps reps
              #:before reclaim-native-memory!))
