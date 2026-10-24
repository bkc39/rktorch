#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/contract exn:fail:contract:blame?)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    + > ~> arange cuda-available? cuda-device cuda-memory-stats
                    cuda-reset-peak-stats! current-generator device dtype
                    make-generator mps-available? mps-device native-memory-use
                    randn reclaim-native-memory! reshape tensor tensor->list
                    tensor-shape to to-dtype transpose zeros)
           (only-in "../vision/transforms.rkt" random-crop)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/reference-crop.rkt" reference-crop)
           (only-in "private/within.rkt" bounded-test))

  (define devices
    (append (list 'cpu)
            (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define (mps? on)
    (and (not (eq? on 'cpu)) (equal? on (mps-device))))

  (define (batches on)
    (define all
      (list (~> (* 2 3 5 7) arange (reshape 2 3 5 7) (to on))
            (~> (* 2 3 5 7) arange (reshape 2 3 5 7) (to-dtype 'uint8) (to on))
            (~> (randn 3 2 5 4) (> 0) (to on))
            (~> (randn 4 3 6 5) (to-dtype 'float16) (to on))
            (~> (randn 4 3 6 5) (to-dtype 'bfloat16) (to on))
            (to (randn 9 1 6 4 #:dtype 'float64) on)
            (to (randn 128 3 32 32) on)
            (~> (for/list ([k (in-range (* 2 2 3 4))])
                  (vector-ref (vector +nan.0 +inf.0 -inf.0 -0.0 1.5) (modulo k 5)))
                tensor (reshape 2 2 3 4) (to on))
            (~> (randn 3 2 5 7) (to on) (transpose 2 3))))
    (if (mps? on)
        (filter (lambda (x) (~> x dtype (eq? 'float64) not)) all)
        all))

  (define (agree? x padding seed)
    (define ours (random-crop x #:padding padding
                              #:generator (make-generator seed)))
    (define theirs (reference-crop x #:padding padding
                                   #:generator (make-generator seed)))
    (and (equal? (tensor-shape ours) (tensor-shape theirs))
         (eq? (dtype ours) (dtype theirs))
         (equal? (device ours) (device x))
         (equal? (tensor->list ours) (tensor->list theirs))))

  (bounded-test "random-crop equals the per-image reference for a generator"
    (for* ([on (in-list devices)]
           [x (in-list (batches on))]
           [padding (in-list '(0 1 4 9))]
           [seed (in-list '(0 7 2026))])
      (check-true (agree? x padding seed)
                  (format "~a ~a ~a padding ~a seed ~a"
                          on (dtype x) (tensor-shape x) padding seed))))

  (bounded-test "random-crop draws from current-generator as the reference does"
    (for ([on (in-list devices)])
      (define x (to (randn 16 3 8 8) on))
      (define (twice crop)
        (parameterize ([current-generator (make-generator 13)])
          (define first (crop x #:padding 4))
          (list (tensor->list first)
                (tensor->list (crop x #:padding 4)))))
      (check-equal? (twice random-crop) (twice reference-crop)
                    (format "~a" on))))

  (define (op-calls-of thunk)
    (define here (current-thread))
    (define crossings 0)
    (define frees 0)
    (define tensors 0)
    (parameterize ([native-collect-at-troughs #f])
      (with-crossing-hook (lambda ()
                            (when (eq? (current-thread) here)
                              (set! crossings (add1 crossings))))
        (with-race-hook (lambda (name _subject)
                          (case (and (eq? (current-thread) here) name)
                            [(ledger-entry-added) (set! tensors (add1 tensors))]
                            [(free-released finalizer-releasing)
                             (set! frees (add1 frees))]
                            [else (void)]))
          (thunk))))
    (- crossings frees (* 2 tensors)))

  (define (crop-op-calls n on)
    (define x (to (randn n 3 32 32) on))
    (random-crop x #:generator (make-generator 1))
    (op-calls-of (lambda () (random-crop x #:generator (make-generator 1)))))

  (bounded-test "random-crop makes as many native op calls at 128 as at 2"
    (for ([on (in-list devices)])
      (define small (crop-op-calls 2 on))
      (define large (crop-op-calls 128 on))
      (check-equal? large small
                    (format "~a: ~a at 2, ~a at 128" on small large))))

  (define (bytes-on on)
    (for/sum ([entry (in-list (native-memory-use))]
              #:when (equal? (car entry) on))
      (cdr entry)))

  (bounded-test "random-crop allocates the crop and indices of one row or column per image"
    (define-values (n c h w padding) (values 32 3 32 32 4))
    (define x (randn n c h w))
    (define on (device x))
    (define out (* 4 n c h w))
    (define indices (* 8 (+ (* 16 n) (* 2 c) (* 4 h) (* 4 w) (* 4 n h) (* 4 n w))))
    (define budget (+ out indices))
    (random-crop x #:padding padding #:generator (make-generator 1))
    (reclaim-native-memory!)
    (define before (bytes-on on))
    (define kept (random-crop x #:padding padding
                              #:generator (make-generator 1)))
    (define grown (- (bytes-on on) before))
    (check-equal? (tensor-shape kept) (list n c h w))
    (check-true (<= grown budget)
                (format "~a bytes over a budget of ~a" grown budget)))

  (when (cuda-available?)
    (bounded-test "random-crop on CUDA peaks at the crop and one intermediate of its size"
      (define-values (n c h w) (values 32 3 32 32))
      (define on (cuda-device))
      (define x (to (randn n c h w) on))
      (define out (* 4 n c h w))
      (define (stat key) (cdr (assq key (cuda-memory-stats on))))
      (random-crop x #:generator (make-generator 1))
      (cuda-reset-peak-stats! on)
      (define before (stat 'allocated))
      (define kept (random-crop x #:generator (make-generator 1)))
      (define peak (- (stat 'peak-allocated) before))
      (check-equal? (tensor-shape kept) (list n c h w))
      (check-true (<= peak (* 5/2 out))
                  (format "~a bytes at the peak for a ~a-byte crop" peak out))))

  (bounded-test "random-crop keeps master's draws for a seed"
    (define x (~> 36 arange (+ 1) (reshape 4 1 3 3) (to-dtype 'int64)))
    (define golden
      '(0 0 0 2 3 0 5 6 0 14 15 0 17 18 0 0 0 0
        23 24 0 26 27 0 0 0 0 0 0 0 29 30 0 32 33 0))
    (for ([crop (in-list (list random-crop reference-crop))])
      (check-equal? (tensor->list
                     (crop x #:padding 1 #:generator (make-generator 0)))
                    golden)))

  (bounded-test "random-crop takes a batch without images or channels"
    (for ([on (in-list devices)])
      (define empty (to (zeros 0 0 4 4) on))
      (check-eq? (random-crop empty) empty (format "~a" on))
      (check-equal? (~> (zeros 2 0 4 4) (to on) random-crop tensor-shape)
                    '(2 0 4 4)
                    (format "~a" on))))

  (bounded-test "random-crop blames its caller for a batch without pixels"
    (for ([x (in-list (list (zeros 2 3 0 4) (zeros 2 3 4 0) (zeros 0 3 0 4)))])
      (check-exn (lambda (e)
                   (and (exn:fail:contract:blame? e)
                        (regexp-match? #rx"blaming: [^\n]*random-crop-test"
                                       (exn-message e))))
                 (lambda () (random-crop x))
                 (format "~a" (tensor-shape x))))))
