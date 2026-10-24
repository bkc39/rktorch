#lang racket/base

(module+ test
  (require (only-in racket/list last)
           (only-in racket/match match-define)
           (only-in rackunit check-eq? check-equal? check-false check-not-eq?)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    accelerator-if-available copy! device device-type dtype
                    item randn tensor to with-default-device)
           (only-in "../nn.rkt"
                    BatchNorm1d BatchNorm2d Buffer? Parameter? buffers
                    in-eval-mode layer-copy layer-rebuild state-dict)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/within.rkt" bounded-test))

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

  (define norms
    (list (list "BatchNorm2d" BatchNorm2d (lambda () (randn 4 3 5 5)))
          (list "BatchNorm1d" BatchNorm1d (lambda () (randn 8 3)))))

  (define (counter bn)
    (last (buffers bn)))

  (define (cache bn)
    (define seen '())
    (layer-rebuild bn
                   (lambda (_name child) child)
                   (lambda (t)
                     (set! seen (cons t seen))
                     t))
    (for/first ([t (in-list seen)]
                #:unless (or (Buffer? t) (Parameter? t)))
      t))

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
      (copy! (counter bn) (tensor (expt 2 24)))
      (for ([_ (in-range 3)])
        (bn x))
      (check-equal? (dtype (counter bn)) 'int64)
      (check-equal? (item (counter bn)) (+ (expt 2 24) 3))
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
      (check-equal? (map car (state-dict bn))
                    '("weight" "bias" "running-mean" "running-var"
                      "num-batches-tracked"))
      (check-equal? (placed (cache bn)) '(cpu int64))
      (check-false (memq (cache bn) (map cdr (state-dict bn)))))

    (bounded-test (format "~a: a copy counts with its own one" name)
      (define bn (make 3))
      (define c (layer-copy bn))
      (check-not-eq? (cache c) (cache bn))
      (c (batch))
      (check-equal? (map item (list (counter c) (counter bn))) '(1 0)))

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
        (check-equal? (placed (cache bn)) (list there 'int64))
        (bn (to x accel))
        (to bn 'cpu)
        (check-equal? (placed (cache bn)) '(cpu int64))
        (bn x)
        (check-equal? (item (counter bn)) 2))

      (bounded-test (format "~a: one built on ~a follows a move to the cpu"
                            name there)
        (define bn (with-default-device accel (make 3)))
        (check-equal? (placed (cache bn)) (list there 'int64))
        (check-eq? (to bn 'cpu) bn)
        (bn (batch))
        (check-equal? (placed (cache bn)) '(cpu int64))
        (check-equal? (item (counter bn)) 1)))))
