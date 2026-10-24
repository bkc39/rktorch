#lang racket/base

(module+ test
  (require (only-in racket/list last)
           (only-in racket/match match-define)
           (only-in rackunit
                    check-eq? check-equal? check-false check-true)
           (only-in (submod "../foreign.rkt" unsafe) to!)
           (only-in (submod "../nn/batch-norm.rkt" private) counter-one)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    accelerator-if-available copy! cpu-device device
                    device-type dtype item randn tensor to with-default-device)
           (only-in "../nn.rkt"
                    BatchNorm1d BatchNorm2d buffers in-eval-mode layer-copy
                    parameters state-dict)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/within.rkt" bounded-test))

  (define (tally thunk)
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
    (values crossings frees tensors))

  (define (op-calls-of thunk)
    (define-values (crossings frees tensors) (tally thunk))
    (- crossings frees (* 2 tensors)))

  (define (crossings-of thunk)
    (define-values (crossings frees _tensors) (tally thunk))
    (- crossings frees))

  (define norms
    (list (list "BatchNorm2d" BatchNorm2d (lambda () (randn 4 3 5 5)))
          (list "BatchNorm1d" BatchNorm1d (lambda () (randn 8 3)))))

  (define (counter bn)
    (last (buffers bn)))

  (define (cache bn)
    (counter-one (counter bn)))

  (struct injected (k))

  (define (faulted-at? k thunk)
    (define here (current-thread))
    (define n 0)
    (with-handlers ([injected? (lambda (_) #t)])
      (with-crossing-hook (lambda ()
                            (when (eq? (current-thread) here)
                              (set! n (add1 n))
                              (when (= n k)
                                (raise (injected k)))))
        (thunk))
      #f))

  (define (repaired-after-every-fault make batch target)
    (let loop ([k 1])
      (define bn (make 3))
      (define x (to (batch) target))
      (bn (batch))
      (cond
        [(faulted-at? k (lambda () (to bn target) (bn x)))
         (to bn target)
         (define before (item (counter bn)))
         (bn x)
         (check-equal? (item (counter bn)) (add1 before)
                       (format "a fault at crossing ~a" k))
         (loop (add1 k))]
        [else (sub1 k)])))

  (define (placed t)
    (list (device-type (device t)) (dtype t)))

  (for ([norm (in-list norms)])
    (match-define (list name make batch) norm)

    (bounded-test (format "~a: a training forward counts in one op call" name)
      (define bn (make 3))
      (define x (batch))
      (bn x)
      (define training (op-calls-of (lambda () (bn x))))
      (define evaluating (in-eval-mode bn (op-calls-of (lambda () (bn x)))))
      (check-equal? (- training evaluating) 1))

    (bounded-test (format "~a: the counter stays int64 past 2^24" name)
      (define bn (make 3))
      (define x (batch))
      (for ([start (list (expt 2 24) (expt 2 53))])
        (copy! (counter bn) (tensor start))
        (for ([_ (in-range 3)])
          (bn x))
        (check-equal? (dtype (counter bn)) 'int64)
        (check-equal? (item (counter bn)) (+ start 3)))
      (check-equal? (item (cache bn)) 1))

    (bounded-test (format "~a: eval forwards leave the counter" name)
      (define bn (make 3))
      (define x (batch))
      (bn x)
      (in-eval-mode bn
        (for ([_ (in-range 2)])
          (bn x)))
      (check-equal? (item (counter bn)) 1))

    (bounded-test (format "~a: every momentum advances the counter" name)
      (for ([momentum (in-list '(0.0 0.1 0.5 1.0))])
        (define bn (make 3 #:momentum momentum))
        (define x (batch))
        (bn x)
        (bn x)
        (check-equal? (item (counter bn)) 2 (format "momentum ~a" momentum))))

    (bounded-test (format "~a: the state dict holds the counter, not its one"
                          name)
      (define bn (make 3))
      (bn (batch))
      (check-equal? (map car (state-dict bn))
                    '("weight" "bias" "running-mean" "running-var"
                      "num-batches-tracked"))
      (check-equal? (placed (cache bn)) '(cpu int64))
      (check-false (memq (cache bn) (map cdr (state-dict bn)))))

    (bounded-test (format "~a: a copy counts with its own one" name)
      (define bn (make 3))
      (bn (batch))
      (define c (layer-copy bn))
      (c (batch))
      (check-equal? (map item (list (counter c) (counter bn))) '(2 1)))

    (bounded-test (format "~a: a no-op move reads no placements" name)
      (define bn (make 3))
      (bn (batch))
      (define tensors (append (parameters bn) (buffers bn)))
      (define each (crossings-of (lambda () (to! (car tensors) (cpu-device)))))
      (check-equal? (crossings-of (lambda () (to bn (cpu-device))))
                    (* (length tensors) each)))

    (bounded-test (format "~a: on the cpu a retry after any fault counts exactly"
                          name)
      (check-true (positive? (repaired-after-every-fault make batch 'float64))))

    (bounded-test (format "~a: a dtype move leaves the counter and its one int64"
                          name)
      (define bn (make 3))
      (to bn 'float64)
      (bn (to (batch) 'float64))
      (check-equal? (placed (counter bn)) '(cpu int64))
      (check-equal? (placed (cache bn)) '(cpu int64))
      (check-equal? (item (counter bn)) 1))

    (define accel (accelerator-if-available))
    (unless (eq? (device-type accel) 'cpu)
      (define there (device-type accel))
      (bounded-test (format "~a: a move to ~a and back takes the one along"
                            name there)
        (define bn (make 3))
        (define x (batch))
        (to bn accel)
        (bn (to x accel))
        (check-equal? (placed (cache bn)) (list there 'int64))
        (to bn 'cpu)
        (bn x)
        (check-equal? (placed (cache bn)) '(cpu int64))
        (check-equal? (item (counter bn)) 2))

      (bounded-test (format "~a: a retry mends a fault moving to ~a or forward"
                            name there)
        (check-true (positive? (repaired-after-every-fault make batch accel))))

      (bounded-test (format "~a: on ~a the counter is exact past 2^24 and 2^53"
                            name there)
        (define bn (to (make 3) accel))
        (define x (to (batch) accel))
        (for ([start (list (expt 2 24) (expt 2 53))])
          (copy! (counter bn) (tensor start))
          (for ([_ (in-range 3)])
            (bn x))
          (check-equal? (placed (counter bn)) (list there 'int64))
          (check-equal? (item (counter bn)) (+ start 3))))

      (bounded-test (format "~a: one built on ~a follows a move to the cpu"
                            name there)
        (define bn (with-default-device accel (make 3)))
        (bn (to (batch) accel))
        (check-equal? (placed (cache bn)) (list there 'int64))
        (check-eq? (to bn 'cpu) bn)
        (bn (batch))
        (check-equal? (placed (cache bn)) '(cpu int64))
        (check-equal? (item (counter bn)) 2)))))
