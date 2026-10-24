#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    ~> arange cuda-available? cuda-device current-generator
                    device dtype make-generator randn reshape tensor->list
                    tensor-shape to to-dtype)
           (only-in "../vision/transforms.rkt" random-crop)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/reference-crop.rkt" reference-crop)
           (only-in "private/within.rkt" bounded-test))

  (define devices
    (if (cuda-available?) (list 'cpu (cuda-device)) (list 'cpu)))

  (define (batches on)
    (list (~> (* 2 3 5 7) arange (reshape 2 3 5 7) (to on))
          (~> (* 2 3 5 7) arange (reshape 2 3 5 7) (to-dtype 'uint8) (to on))
          (to (randn 9 1 6 4 #:dtype 'float64) on)
          (to (randn 128 3 32 32) on)))

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
      (check-equal? large small (format "~a: ~a at 2, ~a at 128" on small large))
      (check-true (<= small 60) (format "~a: ~a op calls" on small)))))
