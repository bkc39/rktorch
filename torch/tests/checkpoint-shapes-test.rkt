#lang racket/base

(module+ test
  (require (only-in racket/contract/base flat-contract-predicate)
           (only-in racket/match match-define)
           rackunit
           (only-in "../main.rkt" ones tensor zeros)
           (only-in "../nn.rkt"
                    Linear grad-scaler-state/c load-optimizer-state! optimizer-state-dict
                    optimizer-state/c parameters scheduler-state/c sgd)
           "../train.rkt"
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/within.rkt" within))

  (define blames-this-test? (blames-caller? "checkpoint-shapes-test"))

  (define (holds c) (flat-contract-predicate c))

  (define optimizer-state? (holds optimizer-state/c))
  (define scheduler-state? (holds scheduler-state/c))
  (define grad-scaler-state? (holds grad-scaler-state/c))
  (define checkpoint-value? (holds checkpoint-value/c))
  (define checkpoint-filename? (holds checkpoint-filename/c))

  (define sgd-group (hash "params" '("weight" "bias") "lr" 0.1 "momentum" 0.9
                          "nesterov" #f "weight_decay" 0))

  (test-case "optimizer-state/c: torch's two keys, slots by parameter name"
    (check-true (optimizer-state? (hash "state" (hash) "param_groups" (list sgd-group))))
    (check-true (optimizer-state?
                 (hash "state" (hash "weight" (hash "exp_avg" (zeros 2) "exp_avg_sq" (zeros 2)
                                                    "step" 3))
                       "param_groups" (list (hash "params" '("weight") "lr" 0.01
                                                  "betas" '(0.9 0.999) "eps" 1e-8
                                                  "weight_decay" 0.01 "initial_lr" 0.01)))))
    (check-false (optimizer-state? (make-hash (list (cons "state" (hash))
                                                    (list "param_groups" sgd-group))))
                 "mutable")
    (check-false (optimizer-state? (hasheq "state" (hash) "param_groups" (list sgd-group)))
                 "not equal?-based")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups" '())) "no group")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups" (list sgd-group) "x" 1)))
    (check-false (optimizer-state? (hash "param_groups" (list sgd-group))))
    (check-false (optimizer-state? (hash "state" (hash "weight" (hash "step" (zeros 1)))
                                         "param_groups" (list sgd-group)))
                 "an integer step")
    (check-false (optimizer-state? (hash "state" (hash "weight" (hash "exp_avg" 1.0))
                                         "param_groups" (list sgd-group)))
                 "a tensor slot")
    (check-false (optimizer-state? (hash "state" (hash "weight" (hash "velocity" (zeros 1)))
                                         "param_groups" (list sgd-group)))
                 "a torch slot name")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups"
                                         (list (hash "lr" 0.1))))
                 "a group without params")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups"
                                         (list (hash-set sgd-group "nesterov" 1))))
                 "a boolean nesterov")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups"
                                         (list (hash-set sgd-group "betas" '(0.9)))))
                 "two betas")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups"
                                         (list (hash-set sgd-group "dampening" 0))))
                 "a fixed torch flag")
    (check-false (optimizer-state? (hash "state" (hasheq "weight" (hash "step" 1))
                                         "param_groups" (list sgd-group)))
                 "parameter states keyed by eq?")
    (check-false (optimizer-state? (hash "state" (hash "weight" (hasheq "step" 1))
                                         "param_groups" (list sgd-group)))
                 "slots keyed by eq?")
    (check-false (optimizer-state? (hash "state" (hash "weight" (make-hash (list (cons "step" 1))))
                                         "param_groups" (list sgd-group)))
                 "mutable slots")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups"
                                         (list (make-hash (hash->list sgd-group)))))
                 "a mutable group")
    (check-false (optimizer-state? (hash "state" (hash) "param_groups"
                                         (list (for/hasheqv ([(k v) (in-hash sgd-group)])
                                                 (values k v)))))
                 "a group keyed by eqv?"))

  (test-case "the state shapes take exact rationals, and hold only checkpoint values"
    (define exact-group (hash-set* sgd-group "lr" 1/10 "momentum" 9/10 "initial_lr" 1/10))
    (define accepted
      (list (list optimizer-state? (hash "state" (hash) "param_groups" (list exact-group)))
            (list scheduler-state? (hash "last_epoch" 0 "base_lrs" '(1/10) "_last_lr" '(1/20)))
            (list scheduler-state? (hash "last_epoch" 0 "base_lrs" '(0.1) "_last_lr" '(0.1)
                                         "T_cur" 1 "T_i" 3))
            (list grad-scaler-state? (hash "scale" 65536.0 "growth_factor" 2.0
                                           "backoff_factor" 0.5 "growth_interval" 2000
                                           "_growth_tracker" 0))))
    (for ([a (in-list accepted)])
      (match-define (list shape? v) a)
      (check-true (shape? v) (format "~e" v))
      (check-true (checkpoint-value? v) (format "~e" v)))
    (check-false (scheduler-state?
                  (hash "last_epoch" 0 "_last_lr" '(0.1)
                        "_schedulers" (list (hasheq "last_epoch" 0 "base_lrs" '(0.1)
                                                    "_last_lr" '(0.1)))))
                 "an inner schedule keyed by eq?"))

  (test-case "scheduler-state/c: last_epoch, base_lrs, _last_lr, restarts and inner schedules"
    (define plain (hash "last_epoch" 3 "base_lrs" '(0.1) "_last_lr" '(0.05)))
    (check-true (scheduler-state? plain))
    (check-true (scheduler-state? (hash-set* plain "T_cur" 1 "T_i" 6)))
    (define composite (hash "last_epoch" 3 "_last_lr" '(0.05) "_schedulers" (list plain plain)))
    (check-true (scheduler-state? composite) "chained and sequential keep no base_lrs, as torch's")
    (check-false (scheduler-state? (hash-set composite "base_lrs" '(0.1))))
    (check-false (scheduler-state? (hash-set plain "T_cur" 1)) "T_cur without T_i")
    (check-false (scheduler-state? (hash-set* plain "T_cur" 1 "T_i" 0)))
    (for ([t-cur (in-list '(1/2 1.5 2.0 -1))])
      (check-false (scheduler-state? (hash-set* plain "T_cur" t-cur "T_i" 6))
                   (format "T_cur ~a is not an exact count" t-cur)))
    (check-false (scheduler-state? (hash-set plain "last_epoch" 1.0)))
    (check-false (scheduler-state? (hash-remove plain "base_lrs")))
    (check-false (scheduler-state? (hash-set plain "_last_lr" '(a))))
    (check-false (scheduler-state? (hash-set plain "step_size" 2)) "constructor arguments")
    (check-false (scheduler-state? (hash-set composite "_schedulers" (list (hash)))))
    (check-false (scheduler-state? (hash-copy plain)) "mutable"))

  (test-case "grad-scaler-state/c: empty, or torch's five keys in range"
    (define st (hash "scale" 65536.0 "growth_factor" 2.0 "backoff_factor" 0.5
                     "growth_interval" 2000 "_growth_tracker" 0))
    (check-true (grad-scaler-state? (hash)))
    (check-true (grad-scaler-state? st))
    (check-false (grad-scaler-state? (make-hash)) "mutable")
    (check-false (grad-scaler-state? (hash-remove st "_growth_tracker")))
    (check-false (grad-scaler-state? (hash-set st "extra" 1)))
    (check-true (grad-scaler-state? (hash-set st "scale" 0.0)) "an underflowed scale")
    (check-false (grad-scaler-state? (hash-set st "scale" -1.0)))
    (check-false (grad-scaler-state? (hash-set st "scale" 65536)) "a flonum scale")
    (check-false (grad-scaler-state? (hash-set st "growth_factor" 1.0)))
    (check-false (grad-scaler-state? (hash-set st "backoff_factor" 1.0)))
    (check-false (grad-scaler-state? (hash-set st "growth_interval" 0)))
    (check-false (grad-scaler-state? (hash-set st "_growth_tracker" -1))))

  (test-case "checkpoint-value/c: the values a checkpoint holds"
    (for ([v (list #t #f 0 (expt 2 70) -3 1.5 +inf.0 -inf.0 +nan.0 -0.0 "" "π" 'sym '()
                   '(1 . 2) (list 1 "a" 'b) (zeros 0 3) (tensor 1.0) (ones 2 #:dtype 'uint8)
                   (ones 2 #:dtype 'bool) (ones 2 #:dtype 'bfloat16) (ones 2 #:dtype 'int64)
                   1/2 -7/3 (hash "lr" 1/10)
                   (hash "a" 1) (hasheq 'a (list (zeros 1))) (hasheqv 1 2.0)
                   (make-hash (list (cons "k" (hash 1 'x)))) (hash '(1 2) "list key"))])
      (check-true (checkpoint-value? v) (format "~e" v)))
    (for ([v (list (lambda () 1) (vector 1) #"bytes" #\c (void) (string->uninterned-symbol "u")
                   (hash "$tensor" 1) (hash "ok" (hash "$float" 1.0)) (list 1 (vector))
                   (cons 1 (lambda () 1)) (hash (vector) 1))])
      (check-false (checkpoint-value? v) (format "~e" v)))
    (define cyclic (make-hasheq))
    (hash-set! cyclic 'self cyclic)
    (check-false (checkpoint-value? cyclic) "a hash that holds itself"))

  (test-case "checkpoint-value/c: only eq?, eqv? and equal? hashes that hold their keys strongly"
    (for ([v (list (hashalw "a" 1) (make-hashalw) (make-weak-hash) (make-weak-hasheq)
                   (make-ephemeron-hash) (make-ephemeron-hasheqv) (hash "k" (make-weak-hash)))])
      (check-false (checkpoint-value? v) (format "~e" v))))

  (test-case "checkpoint-value/c: a cycle through pairs is refused, shared structure is walked once"
    (define loop (read (open-input-string "#0=(1 . #0#)")))
    (define car-loop (read (open-input-string "#0=(#0# . 1)")))
    (define shared-dag (for/fold ([v '(1)]) ([_ (in-range 60)]) (cons v v)))
    (within 10 (lambda ()
                 (check-false (checkpoint-value? loop) "a list that is its own tail")
                 (check-false (checkpoint-value? car-loop) "a pair that is its own car")
                 (check-false (checkpoint-value? (hash "k" loop)) "a cycle inside a hash")
                 (check-false (checkpoint-value? (let ([bad (list (vector))]) (cons bad bad)))
                              "a refused node in two places")
                 (check-true (checkpoint-value? shared-dag) "2^60 paths, 61 nodes")
                 (check-true (checkpoint-value? (let ([h (hash "x" 1.0)]) (list h h h)))
                             "one hash in three places"))))

  (test-case "checkpoint-filename/c: text with {name}, {name:.Nf} and {name:0Nd}"
    (for ([f '("{epoch}-{step}" "{epoch}-{val-loss:.2f}" "best" "{epoch:02d}" "split-{epoch}"
               "epoch={epoch}-validation_loss={val_loss:.3f}")])
      (check-true (checkpoint-filename? f) f))
    (for ([f '("" "{" "}" "{}" "{epoch" "{epoch:.2}" "{epoch:2d}" "{epoch:x}" "{a{b}}")])
      (check-false (checkpoint-filename? f) f))
    (check-false (checkpoint-filename? 'epoch)))

  (test-case "save-top-k/c and ckpt-path/c"
    (define top-k? (holds save-top-k/c))
    (define ckpt-path? (holds ckpt-path/c))
    (check-true (andmap top-k? '(-1 0 1 5)))
    (check-false (ormap top-k? '(-2 1.0 #f)))
    (check-true (andmap ckpt-path? (list #f "a.ckpt" (string->path "a.ckpt") 'last 'best)))
    (check-false (ormap ckpt-path? '(3 latest #t ""))))

  (test-case "optimizer-state-dict's precondition: the model names every parameter"
    (define net (Linear 2 1))
    (define opt (sgd (parameters net) #:lr 0.1))
    (check-pred optimizer-state? (optimizer-state-dict opt net)
                "parameters of the model reach the body")
    (check-exn blames-this-test? (lambda () (optimizer-state-dict opt (Linear 2 1))))
    (check-exn blames-this-test?
               (lambda () (load-optimizer-state! opt (Linear 2 1)
                                                 (hash "state" (hash)
                                                       "param_groups" (list sgd-group)))))
    (check-exn blames-this-test?
               (lambda () (load-optimizer-state! opt net (hash "state" (hash)))))))
