#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../foreign/autograd-ops.rkt" with-gradient with-gradients)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/memory.rkt" adopted-bytes)
           (only-in "../foreign/structs.rkt" tensor-handle)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    backward! grad manual-seed! randn reshape sum tensor->list)
           (only-in "../nn.rkt" Linear parameters)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/unseen-gradient.rkt" with-unseen-gradient)
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
    (match-define (list front middle _) ls)
    (define ps (params-of ls))
    (with-gradients ([qs gs] ps)
      (check-equal? qs (params-of (list front middle)))
      (check-equal? (map flat gs) (map (lambda (q) (flat (grad q))) qs))))

  (define (adopted g)
    (adopted-bytes (tensor-handle g)))

  (define (with-gradients-unseen thunk)
    (with-unseen-gradient (box (* 1024 1024)) thunk))

  (bounded-test "adopts what the one-at-a-time path adopts"
    (define batched
      (with-gradients-unseen
       (lambda ()
         (with-gradients (gs (params-of (layers 2)))
           (map adopted gs)))))
    (define one-by-one
      (with-gradients-unseen
       (lambda ()
         (for/list ([p (in-list (params-of (layers 2)))])
           (with-gradient (g p) (adopted g))))))
    (check-true (ormap positive? batched) "the shadow charged the gradients")
    (check-equal? batched one-by-one))

  (bounded-test "frees every handle it accounts, also when the body raises"
    (define ps (params-of (layers 3)))
    (check-equal? (balance-of (lambda () (with-gradients ([qs gs] ps) (void)))) 0)
    (check-equal? (balance-of (lambda () (with-gradients ([qs gs] ps) (error 'body))))
                  0)))
