#lang racket/base

(require (only-in racket/list append* last)
         (only-in racket/path file-name-from-path)
         rackunit
         (only-in "../../data/loader.rkt" dataloader tensor-dataset)
         (only-in "../../main.rkt"
                  * + arange copy! generator-state make-generator manual-seed! ones randn rng-state
                  tensor->list with-no-grad zeros)
         (only-in "../../nn.rkt"
                  Dropout Linear Sequential adamw cosine-annealing-lr ema ema-n-averaged
                  ema-update! grad-scaler-state-dict linear-lr mse-loss named-parameters
                  optimizer-state-dict scheduler-state-dict sequential-lr set-ema-n-averaged!
                  state-dict weight-decay-param-groups)
         (only-in "../../private/util.rkt" with-temporary-directory)
         "../../train.rkt"
         (only-in "../../train/private/types.rkt" callback-hook)
         (only-in "comparable.rkt" comparable host-bytes)
         (only-in "trainer-kit.rkt"
                  apply-keywords collecting-hparams collecting-logger collecting-steps))

(provide check-resume
         fixture-fit-args
         fixture-world
         (struct-out world)
         every-epoch-saver)

(struct world (trainer train val callbacks generators fingerprints))

(define-trainer Resumable (net temperature seen)
  (define-init (s)
    (set! net (Sequential (Linear 4 8) (Dropout 0.5) (Linear 8 1)))
    (set! temperature 1.0)
    (set! seen s))
  (define-hparams temperature)
  (define-optimizers
    (define opt (adamw (weight-decay-param-groups net #:weight-decay 0.1) #:lr 0.01))
    (define warmup (linear-lr opt #:start-factor 0.1 #:total-iters 2))
    (values opt
            (scheduler-config (sequential-lr opt (list warmup (cosine-annealing-lr opt #:t-max 6))
                                             #:milestones '(2))
                              #:interval 'step)))
  (define-training-step (x y i)
    (set-box! seen (cons (tensor->list i) (unbox seen)))
    (define loss (* (mse-loss (net x) y) temperature))
    (log! 'train-loss loss)
    loss)
  (define-validation-step (x y _i)
    (log! 'val-loss (mse-loss (net x) y))))

(define (counting-callback)
  (define count 0)
  (define total (zeros 1))
  (lambda-callback #:state-key "counting"
                   #:on-train-batch-end (lambda (_r . _)
                                          (set! count (add1 count))
                                          (set! total (+ total (ones 1))))
                   #:state-dict (lambda (_r) (hash "count" count "total" total))
                   #:load-state! (lambda (_r state)
                                   (set! count (hash-ref state "count"))
                                   (set! total (hash-ref state "total")))))

(define-callback averaging (average e)
  (define-init () (set! average #f) (set! e #f))
  (define-state-key "averaging")
  (define-hook (setup _stage)
    (set! average (Sequential (Linear 4 8) (Dropout 0.5) (Linear 8 1)))
    (set! e (ema (run-trainer run) average #:decay 0.9)))
  (define-hook (on-train-batch-end _loss _extra . _batch) (ema-update! e))
  (define-hook (state-dict)
    (hash "n" (ema-n-averaged e)
          "average" (for/hash ([entry (in-list (named-parameters average))])
                      (values (car entry) (cdr entry)))))
  (define-hook (load-state! state)
    (with-no-grad
      (for ([entry (in-list (named-parameters average))])
        (copy! (cdr entry) (hash-ref (hash-ref state "average") (car entry)))))
    (set-ema-n-averaged! e (hash-ref state "n"))))

(define (data n generator)
  (tensor-dataset (randn n 4 #:generator generator)
                  (randn n 1 #:generator generator)
                  (arange n)))

(define fixture-fit-args
  '((#:accumulate-grad-batches . 2) (#:gradient-clip-val . 0.5)))

(define (fixture-world dir #:generator? [generator? #f] #:workers [workers 0])
  (define seen (box '()))
  (define g (and generator? (make-generator 11)))
  (define source (make-generator 3))
  (world (Resumable seen)
         (dataloader (data 16 source) #:batch-size 2 #:shuffle? #t #:generator g
                     #:workers workers)
         (dataloader (data 8 source) #:batch-size 4)
         (list (hparam-scheduler 'temperature (lambda (t) (/ 1.0 (add1 t))))
               (counting-callback)
               (averaging)
               (model-checkpoint #:dirpath dir #:monitor 'val-loss #:save-top-k 2))
         (if g (list g) '())
         seen))

(define (every-epoch-saver dir)
  (model-checkpoint #:dirpath dir #:filename "split-{epoch}-{step}" #:save-top-k -1))

(define ((every-steps-saver n) dir)
  (model-checkpoint #:dirpath dir #:filename "split-{epoch}-{step}" #:save-top-k -1
                    #:every-n-train-steps n))

(define (monitor-checkpoint w)
  (for/first ([cb (in-list (world-callbacks w))] #:when (model-checkpoint? cb)) cb))

(define (callback-states w r)
  (for/list ([cb (in-list (world-callbacks w))])
    (define hook (callback-hook cb 'state-dict))
    (and hook (comparable (hook cb r)))))

(define (snapshot w r)
  (define tr (world-trainer w))
  (hasheq 'parameters (comparable (state-dict tr))
          'optimizers (for/list ([opt (in-list (run-optimizers r))])
                        (comparable (optimizer-state-dict opt tr)))
          'schedulers (for/list ([c (in-list (run-schedulers r))])
                        (scheduler-state-dict (scheduler-config-scheduler c)))
          'scaler (and (run-grad-scaler r) (grad-scaler-state-dict (run-grad-scaler r)))
          'callbacks (callback-states w r)
          'cpu-rng (host-bytes (rng-state))
          'racket-rng (pseudo-random-generator->vector (current-pseudo-random-generator))
          'loader-generators (map (lambda (g) (host-bytes (generator-state g)))
                                  (world-generators w))
          'epoch (run-epoch r)
          'global-step (run-global-step r)
          'best-k (for/hash ([(path score) (in-hash (model-checkpoint-best-k-models
                                                     (monitor-checkpoint w)))])
                    (values (path->string (file-name-from-path path)) score))))

(struct leg (world run logger saver))

(define (seed! n)
  (manual-seed! n)
  (random-seed n))

(define (fit-leg make-world dir fit-args #:seed seed #:ckpt-path [ckpt-path #f] #:bounds bounds
                 #:saver make-saver)
  (seed! seed)
  (define w (make-world dir))
  (define l (collecting-logger))
  (define saver (make-saver dir))
  (define r
    (apply-keywords fit
                    (append bounds
                            fit-args
                            (list (cons '#:val (world-val w))
                                  (cons '#:logger l)
                                  (cons '#:callbacks (cons saver (world-callbacks w)))
                                  (cons '#:ckpt-path ckpt-path)))
                    (world-trainer w)
                    (world-train w)))
  (leg w r l saver))

(define (fingerprints l)
  (reverse (unbox (world-fingerprints (leg-world l)))))

(define (check-resume make-world #:split split #:total n #:fit-args [fit-args '()]
                      #:split-step [split-step #f])
  (define splits (if (list? split) split (list split)))
  (define make-saver (if split-step (every-steps-saver split-step) every-epoch-saver))
  (with-temporary-directory (root #:template "resume-~a")
    (define straight (fit-leg make-world (build-path root "straight") fit-args #:saver make-saver
                              #:seed 0 #:bounds (list (cons '#:max-epochs n))))
    (define split-dir (build-path root "split"))
    (define first-leg
      (fit-leg make-world split-dir fit-args #:seed 0 #:saver make-saver
               #:bounds (if split-step
                            (list (cons '#:max-epochs n) (cons '#:max-steps split-step))
                            (list (cons '#:max-epochs (car splits))))))
    (define legs
      (for/fold ([legs (list first-leg)] #:result (reverse legs))
                ([epochs (in-list (append (if split-step '() (cdr splits)) (list n)))])
        (define previous (car legs))
        (cons (fit-leg make-world split-dir fit-args #:seed 12345 #:saver make-saver
                       #:ckpt-path (model-checkpoint-best-model-path (leg-saver previous))
                       #:bounds (list (cons '#:max-epochs epochs)))
              legs)))
    (define resumed (last legs))
    (define expected (snapshot (leg-world straight) (leg-run straight)))
    (define actual (snapshot (leg-world resumed) (leg-run resumed)))
    (for ([(key v) (in-hash expected)])
      (check-equal? (hash-ref actual key) v (format "resume law: ~a" key)))
    (check-equal? (append* (map (lambda (l) (collecting-steps (leg-logger l))) legs))
                  (collecting-steps (leg-logger straight))
                  "resume law: the logged rows")
    (check-equal? (collecting-hparams (leg-logger first-leg))
                  (collecting-hparams (leg-logger straight))
                  "resume law: the hyperparameter row")
    (check-equal? (append* (map fingerprints legs)) (fingerprints straight)
                  "resume law: the batches drawn")))
