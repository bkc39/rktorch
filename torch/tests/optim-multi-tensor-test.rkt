#lang racket/base

(module+ test
  (require (only-in racket/contract exn:fail:contract:blame?)
           (only-in racket/list partition)
           (only-in racket/match match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../foreign.rkt" accelerator-if-available device-type)
           (only-in "../foreign/raw/fault.rkt" with-crossing-hook)
           (only-in "../foreign/raw/pressure-settings.rkt"
                    native-collect-at-troughs)
           (only-in "../main.rkt"
                    backward! dtype grad manual-seed! randn reshape shape sum
                    tensor->list to zeros)
           (only-in "../nn.rkt"
                    Linear Sequential adam adam? adamw adamw? ema ema-update!
                    parameters rmsprop set-learning-rate! sgd step!
                    zero-grads!)
           (only-in (submod "../nn/optim.rkt" private) optimizer-state)
           (only-in "../vision/resnet.rkt" ResNet)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/within.rkt" bounded-test))

  (struct config (name foreach-bound fused-bound optimizer options))

  (define configs
    (list (config "sgd" 6 6 sgd (hash '#:lr 0.1))
          (config "sgd, Nesterov momentum and weight decay" 11 6 sgd
                  (hash '#:lr 0.1 '#:momentum 0.9 '#:nesterov? #t
                        '#:weight-decay 5e-4))
          (config "adam" 13 7 adam (hash))
          (config "adam, weight decay" 15 7 adam
                  (hash '#:lr 0.01 '#:weight-decay 1e-2))
          (config "adamw" 14 7 adamw (hash '#:lr 0.01))
          (config "rmsprop" 11 #f rmsprop (hash '#:lr 0.01))
          (config "rmsprop, momentum and weight decay" 15 #f rmsprop
                  (hash '#:lr 0.01 '#:momentum 0.5 '#:weight-decay 1e-3))))

  (define (mode-options mode)
    (match mode
      ['loop (hash '#:foreach? #f)]
      ['foreach (hash '#:foreach? #t)]
      ['fused (hash '#:fused? #t)]))

  (define (build c mode ps)
    (match-define (config _ _ _ optimizer options) c)
    (define all (for/fold ([all options]) ([(k v) (in-hash (mode-options mode))])
                  (hash-set all k v)))
    (define kws (sort (hash-keys all) keyword<?))
    (keyword-apply optimizer kws (map (lambda (k) (hash-ref all k)) kws) (list ps)))

  (define (fast-modes c)
    (if (config-fused-bound c) '(foreach fused) '(foreach)))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (close? a b)
    (<= (abs (- a b)) (~> (abs b) (* 1e-5) (+ 1e-6))))

  (define (check-close xss yss what)
    (check-equal? (map length xss) (map length yss) what)
    (for ([xs (in-list xss)] [ys (in-list yss)] [k (in-naturals)])
      (check-true (andmap close? xs ys) (format "~a: parameter ~a" what k))))

  (define (trained c mode #:move-to [move-to #f] #:lr-after-1 [lr #f])
    (manual-seed! 0)
    (define front (Linear 4 8))
    (define back (Linear 8 3))
    (define ps (append (parameters front) (parameters back)))
    (define opt (build c mode ps))
    (for/fold ([x (randn 5 4)]) ([n (in-range 1 4)])
      (zero-grads! opt)
      (define h (front x))
      (backward! (sum (if (= n 2) h (back h))))
      (step! opt)
      (when (and lr (= n 1))
        (set-learning-rate! opt lr))
      (cond
        [(and move-to (= n 1))
         (apply to front move-to)
         (apply to back move-to)
         (apply to x move-to)]
        [else x]))
    (map flat ps))

  (define (trained-resnet c mode)
    (manual-seed! 0)
    (define net (ResNet #:base 4))
    (define opt (build c mode (parameters net)))
    (define x (randn 2 3 32 32))
    (for ([_ (in-range 3)])
      (zero-grads! opt)
      (~> x net sum backward!)
      (step! opt))
    (map flat (parameters net)))

  (define accelerator (accelerator-if-available))

  (define accelerator-target
    (if (eq? (device-type accelerator) 'cpu)
        (list accelerator 'float64)
        (list accelerator)))

  (define (crossings-of thunk)
    (define here (current-thread))
    (define n 0)
    (parameterize ([native-collect-at-troughs #f])
      (with-crossing-hook (lambda ()
                            (when (eq? (current-thread) here)
                              (set! n (add1 n))))
        (thunk)))
    n)

  (define (owned-balance thunk)
    (define here (current-thread))
    (define added 0)
    (define freed 0)
    (parameterize ([native-collect-at-troughs #f])
      (with-race-hook (lambda (name _subject)
                        (case (and (eq? (current-thread) here) name)
                          [(ledger-entry-added) (set! added (add1 added))]
                          [(free-released) (set! freed (add1 freed))]
                          [else (void)]))
        (thunk)))
    (- added freed))

  (define (steady c mode depth measure)
    (manual-seed! 0)
    (define net
      (apply Sequential (for/list ([_ (in-range depth)]) (Linear 3 3))))
    (define opt (build c mode (parameters net)))
    (define x (randn 2 3))
    (for/last ([_ (in-range 4)])
      (zero-grads! opt)
      (~> x net sum backward!)
      (measure (lambda () (step! opt)))))

  (for ([c (in-list configs)])
    (match-define (config name foreach-bound fused-bound _ _) c)
    (bounded-test
     (format "~a: the foreach step is the loop's, bit for bit, a gradient missing on step 2"
             name)
     (check-equal? (trained c 'foreach) (trained c 'loop)))
    (bounded-test (format "~a: on 09's ResNet the foreach step is the loop's" name)
      (check-equal? (trained-resnet c 'foreach)
                    (trained-resnet c 'loop)))
    (bounded-test (format "~a: a rate set between steps reaches the foreach step" name)
      (check-equal? (trained c 'foreach #:lr-after-1 0.003)
                    (trained c 'loop #:lr-after-1 0.003)))
    (bounded-test (format "~a: foreach state follows a move to float64" name)
      (check-equal? (trained c 'foreach #:move-to '(float64))
                    (trained c 'loop #:move-to '(float64))))
    (for ([mode (in-list (fast-modes c))])
      (define bound (if (eq? mode 'foreach) foreach-bound fused-bound))
      (bounded-test (format "~a, ~a: within tolerance of the loop on the CPU" name mode)
        (check-close (trained c mode) (trained c 'loop) name))
      (bounded-test (format "~a, ~a: within tolerance after a move to ~a"
                            name mode (device-type accelerator))
        (check-close (trained c mode #:move-to accelerator-target)
                     (trained c 'loop #:move-to accelerator-target)
                     name))
      (bounded-test
       (format "~a, ~a: a steady step makes at most ~a crossings at any depth"
               name mode bound)
       (define one (steady c mode 1 crossings-of))
       (check-equal? (steady c mode 2 crossings-of) one)
       (check <= one bound))
      (bounded-test (format "~a, ~a: a steady step frees every tensor it accounts"
                            name mode)
        (check-equal? (steady c mode 2 owned-balance) 0))
      (bounded-test (format "~a, ~a: a step leaves every gradient as it was" name mode)
        (manual-seed! 0)
        (define net (Sequential (Linear 4 8) (Linear 8 3)))
        (define opt (build c mode (parameters net)))
        (define (gradients) (map (lambda (p) (flat (grad p))) (parameters net)))
        (for ([_ (in-range 2)])
          (zero-grads! opt)
          (~> (randn 5 4) net sum backward!)
          (define before (gradients))
          (step! opt)
          (check-equal? (gradients) before)))))

  (bounded-test "fused adam keeps its step counts float32 across a move to float64"
    (manual-seed! 0)
    (define net (Linear 4 3))
    (define opt (adam (parameters net) #:fused? #t))
    (~> (randn 2 4) net sum backward!)
    (step! opt)
    (to net 'float64)
    (zero-grads! opt)
    (~> (randn 2 4 #:dtype 'float64) net sum backward!)
    (step! opt)
    (define-values (counts moments)
      (partition (lambda (s) (null? (shape s))) (optimizer-state opt)))
    (check-equal? (map dtype counts) '(float32 float32))
    (check-true (andmap (lambda (s) (eq? (dtype s) 'float64)) moments)))

  (define (mixed-step-crossings #:move-after-step? move-after-step?)
    (manual-seed! 0)
    (define front (Linear 3 3))
    (define back (Linear 3 3))
    (unless move-after-step?
      (to back 'float64))
    (define opt (adam (append (parameters front) (parameters back))))
    (define (backward-both!)
      (zero-grads! opt)
      (~> (randn 2 3) front sum backward!)
      (~> (randn 2 3 #:dtype (dtype (car (parameters back)))) back sum backward!))
    (for ([n (in-range 4)])
      (backward-both!)
      (step! opt)
      (when (and move-after-step? (zero? n))
        (to back 'float64)))
    (backward-both!)
    (crossings-of (lambda () (step! opt))))

  (bounded-test "a move of one layer regroups the parameters a foreach step updates"
    (check-equal? (mixed-step-crossings #:move-after-step? #t)
                  (mixed-step-crossings #:move-after-step? #f)))

  (define (violation thunk)
    (with-handlers ([exn:fail:contract:blame? exn-message])
      (thunk)
      #f))

  (bounded-test "asking for fused and foreach together is the caller's violation"
    (define m (violation (lambda ()
                           (~> (Linear 2 2)
                               parameters
                               (adam #:foreach? #t #:fused? #t)))))
    (check-regexp-match #rx"fused and foreach cannot both be requested" m)
    (check-regexp-match #rx"blaming: [^\n]*optim-multi-tensor-test" m))

  (bounded-test "a fused step over integer parameters is the caller's violation"
    (define m (violation (lambda ()
                           (sgd (list (zeros 2 #:dtype 'int64)) #:lr 0.1
                                #:fused? #t))))
    (check-regexp-match #rx"floating-tensor/c" m)
    (check-regexp-match #rx"blaming: [^\n]*optim-multi-tensor-test" m))

  (bounded-test "rmsprop has no fused kernel to ask for"
    (check-exn exn:fail:contract?
               (lambda () (rmsprop (parameters (Linear 2 2)) #:fused? #t))))

  (bounded-test "an adamw is an adam, not the other way round"
    (check-true (adam? (adamw (parameters (Linear 2 2)))))
    (check-false (adamw? (adam (parameters (Linear 2 2))))))

  (define (ema-update-crossings depth)
    (manual-seed! 0)
    (define (net)
      (apply Sequential (for/list ([_ (in-range depth)]) (Linear 3 3))))
    (define avg (ema (net) (net) #:decay 0.9))
    (ema-update! avg)
    (crossings-of (lambda () (ema-update! avg))))

  (bounded-test "an EMA of layers without parameters updates without a native call"
    (define avg (ema (Sequential) (Sequential)))
    (ema-update! avg)
    (check-equal? (crossings-of (lambda () (ema-update! avg))) 0))

  (bounded-test "an EMA update makes at most 4 crossings at any depth"
    (define one (ema-update-crossings 1))
    (check-equal? (ema-update-crossings 2) one)
    (check <= one 4)))
