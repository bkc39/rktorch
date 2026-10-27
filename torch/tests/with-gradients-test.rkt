#lang racket/base

(module+ test
  (require rackunit
           (only-in threading ~>)
           (only-in "../foreign/autograd-ops.rkt" with-gradient with-gradients)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/memory.rkt" adopted-bytes)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    backward! grad manual-seed! randn reshape sum tensor->list)
           (only-in "../nn.rkt" Linear parameters)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/within.rkt" bounded-test))

  (define (flat t) (~> t (reshape -1) tensor->list))

  (define (crossings-of thunk)
    (define here (current-thread))
    (define n 0)
    (parameterize ([native-collect-at-troughs #f])
      (with-crossing-hook (lambda ()
                            (when (eq? (current-thread) here)
                              (set! n (add1 n))))
        (thunk)))
    n)

  (define (balance-of thunk)
    (define here (current-thread))
    (define added 0)
    (define freed 0)
    (with-race-hook (lambda (name _subject)
                      (case (and (eq? (current-thread) here) name)
                        [(ledger-entry-added) (set! added (add1 added))]
                        [(free-released) (set! freed (add1 freed))]
                        [else (void)]))
      (with-handlers ([exn:fail? void]) (thunk)))
    (- added freed))

  (define (layers depth #:through [through depth])
    (manual-seed! 0)
    (define ls (for/list ([_ (in-range depth)]) (Linear 3 3)))
    (~> (for/fold ([h (randn 2 3)]) ([l (in-list ls)] [_ (in-range through)])
          (l h))
        sum backward!)
    ls)

  (define (params-of ls) (apply append (map parameters ls)))

  (bounded-test "a fetch and its release cost one crossing each at any depth"
    (for ([depth (in-list '(1 4))])
      (define ps (params-of (layers depth)))
      (check-equal? (crossings-of (lambda () (with-gradients ([qs gs] ps) (void)))) 2)
      (check-equal? (crossings-of (lambda () (with-gradients (gs ps) (void)))) 2)))

  (bounded-test "pairs each parameter that has a gradient with it, in order"
    (define ls (layers 3 #:through 2))
    (define ps (params-of ls))
    (with-gradients ([qs gs] ps)
      (check-equal? qs (params-of (list (car ls) (cadr ls))))
      (check-equal? (map flat gs) (map (lambda (q) (flat (grad q))) qs))))

  (bounded-test "adopts what the one-at-a-time path adopts"
    (define batched
      (with-gradients ([qs gs] (params-of (layers 2))) (map adopted-bytes gs)))
    (define one-by-one
      (for/list ([p (in-list (params-of (layers 2)))])
        (define seen #f)
        (with-gradient (g p) (set! seen (adopted-bytes g)))
        seen))
    (check-equal? batched one-by-one))

  (bounded-test "frees every handle it accounts, also when the body raises"
    (define ps (params-of (layers 3)))
    (check-equal? (balance-of (lambda () (with-gradients ([qs gs] ps) (void)))) 0)
    (check-equal? (balance-of (lambda () (with-gradients ([qs gs] ps) (error 'body))))
                  0)))
