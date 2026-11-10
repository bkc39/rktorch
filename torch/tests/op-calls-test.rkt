#lang racket/base

(module+ test
  (require rackunit
           (only-in "private/wedge.rkt" exit-status-of))

  (define (with-one-reaper thunk)
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"RKTORCH_REAPERS" #"1")
    (parameterize ([current-environment-variables env])
      (thunk)))

  (define (holding-the-reaper . body)
    `(begin
       (require (only-in torch make-generator zeros)
                (only-in torch/foreign/raw/cas cas-claim!)
                (only-in torch/foreign/raw/fault with-crossing-hook)
                (only-in torch/foreign/raw/pressure-settings native-collect-at-troughs)
                (only-in torch/foreign/raw/race-points race-hook)
                (only-in torch/runtime/reaper
                         reap-after-collection! reap-when-dead! reaped wake-reapers!)
                (only-in (submod torch/runtime/reaper private)
                         reaper-statistics reaper-threads)
                (only-in torch/tests/private/op-calls crossings-of op-calls-of)
                (only-in torch/tests/private/race-harness with-race-hook)
                (only-in torch/tests/private/wedge call-with-process-deadline))
       (define holding (box #f))
       (define go (box #f))
       (set-box! race-hook
                 (lambda (name subject)
                   (when (and (eq? name 'reaper-taken)
                              (memq subject (reaper-threads))
                              (cas-claim! holding #f subject #:site 'gate))
                     (let spin () (unless (unbox go) (spin))))))
       (define kind (reaped 'held (lambda (_v) (void))))
       (define (released) (hash-ref (reaper-statistics) 'released))
       (define (settle!)
         (collect-garbage)
         (let reap ([seen (released)])
           (reap-after-collection!)
           (unless (= seen (released)) (reap (released)))))
       (call-with-process-deadline
        50
        (lambda ()
          (let hold ()
            (unless (unbox holding)
              (reap-when-dead! kind (make-vector 1 0))
              (collect-garbage)
              (wake-reapers!)
              (sleep 0.01)
              (hold)))
          ,@body))))

  (define dead-generator-program
    (holding-the-reaper
     '(define (op)
        (reap-after-collection!)
        (zeros 1))
     '(op-calls-of op)
     '(settle!)
     '(define alone (op-calls-of op))
     '(settle!)
     '(make-generator 1)
     '(collect-garbage)
     '(define before (released))
     '(define with-a-dead-generator (op-calls-of op))
     '(define freed (- (released) before))
     '(set-box! go #t)
     '(eprintf "op alone ~a, with a dead generator freed in its count ~a (~a released)\n"
               alone with-a-dead-generator freed)
     '(exit (if (and (= with-a-dead-generator alone) (= freed 1)) 0 1))))

  (define dead-tensors 20)

  (define dead-tensors-program
    (holding-the-reaper
     '(define (op)
        (zeros 1))
     '(define (every-crossing-of thunk)
        (define here (current-thread))
        (define crossings 0)
        (define frees 0)
        (parameterize ([native-collect-at-troughs #f])
          (with-crossing-hook (lambda ()
                                (when (eq? (current-thread) here)
                                  (set! crossings (add1 crossings))))
            (with-race-hook (lambda (name _subject)
                              (when (and (eq? name 'free-released) (eq? (current-thread) here))
                                (set! frees (add1 frees))))
              (thunk))))
        (list crossings frees))
     `(define (with-dead-tensors measure)
        (settle!)
        (for ([_ (in-range ,dead-tensors)])
          (zeros 1))
        (collect-garbage)
        (define before (released))
        (define counted (measure op))
        (list counted (- (released) before)))
     '(void (crossings-of op) (every-crossing-of op))
     '(settle!)
     '(define alone (crossings-of op))
     '(settle!)
     '(define every-alone (car (every-crossing-of op)))
     '(define shared (with-dead-tensors crossings-of))
     '(define old (with-dead-tensors every-crossing-of))
     '(set-box! go #t)
     '(eprintf "op alone ~a, every crossing ~a; reaping dead tensors inline ~a, every crossing ~a\n"
               alone every-alone shared old)
     `(exit (if (and (equal? shared (list alone ,dead-tensors))
                     (equal? old (list (list (+ every-alone ,dead-tensors) 0) ,dead-tensors)))
                0
                1))))

  (test-case "op-calls-of counts no release a reaper batch on its thread makes"
    (check-equal? (with-one-reaper (lambda () (exit-status-of dead-generator-program #:within 60)))
                  0))

  (test-case "crossings-of counts none of the dead tensors an op reaps inline after a collection"
    (check-equal? (with-one-reaper (lambda () (exit-status-of dead-tensors-program #:within 60)))
                  0)))
