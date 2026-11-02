#lang racket/base

(module+ test
  (require (only-in ffi/unsafe _fun _int _int64 _list)
           (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../foreign/autograd-ops.rkt" with-gradient with-gradients)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/gradients.rkt" fetched-gradient gradients-allocator)
           (only-in "../foreign/raw/memory.rkt" _tr-device-type adopted-bytes free-handles!)
           (only-in "../foreign/raw/syntax.rkt" _Tensor _Tensor/null define-torch)
           (only-in "../foreign/structs.rkt" tensor-handle)
           (only-in "../generated.rkt" embedding)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    backward! grad manual-seed! randn requires-grad! reshape sum tensor
                    tensor->list)
           (only-in "../nn.rkt" Linear parameters sgd step!)
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
                  0))

  (define-torch tensors-grad/unwrapped
    (_fun (ts : (_list i _Tensor))
          (n : _int64 = (length ts))
          (grads : (_list o _Tensor/null n))
          (nbytes : (_list o _int64 n))
          (types : (_list o _tr-device-type n))
          (indices : (_list o _int64 n))
          -> (rc : _int)
          -> (and (zero? rc) (map fetched-gradient grads nbytes types indices)))
    #:c-id tr_tensors_grad)

  ;; 2^60 floats: a deterministic host OOM, through the RNG's no-retry path
  (define (fail-with-oom!)
    (with-handlers ([exn:fail? void])
      (randn 1152921504606846976))
    #f)

  (bounded-test "a fetch that fails with an OOM collects and runs again"
    (define ps (params-of (layers 2)))
    (define calls 0)
    (define (fails-once ts)
      (set! calls (add1 calls))
      (if (= calls 1) (fail-with-oom!) (tensors-grad/unwrapped ts)))
    (define handles ((gradients-allocator fails-once) ps))
    (check-equal? calls 2)
    (check-equal? (length handles) (length ps))
    (free-handles! (filter values handles)))

  (define (sparse-step foreach?)
    (manual-seed! 0)
    (define w (requires-grad! (randn 3 2)))
    (~> (embedding w (tensor '(0 2) #:dtype 'int64) -1 #f #t) sum backward!)
    (step! (sgd (list w) #:lr 0.5 #:foreach? foreach?))
    (flat w))

  (bounded-test "a step takes a sparse gradient as the one-at-a-time path does"
    (check-equal? (sparse-step #t) (sparse-step #f))))

