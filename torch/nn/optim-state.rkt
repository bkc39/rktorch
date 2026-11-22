#lang racket/base

(require (only-in racket/contract/base
                  ->i contract? list/c listof non-empty-listof rename-contract)
         (only-in racket/dict in-dict)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/match match match* match-lambda)
         (only-in racket/set list->set set=?)
         (only-in "../foreign.rkt"
                  copy! device dtype full item tensor-shape tensor? with-no-grad zeros-like)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt" layer? named-parameters)
         (only-in "optim.rkt" adam? optimizer-parameters optimizer-root optimizer-store optimizer?
                  rmsprop? sgd?)
         (only-in (submod "optim.rkt" private) optimizer-mode tracked-state tracked?)
         (only-in "optim-groups.rkt" store-groups store-initial-lrs store-load!)
         (only-in (submod "param-group.rkt" private) param-group-settings param-group-tensors)
         (only-in "private/state-shape.rkt" record/c state-real/c string-keyed/c))

(define slots/c
  (record/c (hash "momentum_buffer" tensor? "exp_avg" tensor? "exp_avg_sq" tensor?
                  "square_avg" tensor? "step" exact-nonnegative-integer?)
            #:required '()))

(define param-group/c
  (record/c (hash "params" (listof string?) "nesterov" boolean?
                  "betas" (list/c state-real/c state-real/c)
                  "lr" state-real/c "momentum" state-real/c "weight_decay" state-real/c
                  "eps" state-real/c "alpha" state-real/c "initial_lr" state-real/c)
            #:required '("params")))

(define (parameters-named? opt model)
  (define named
    (for/hasheq ([entry (in-list (named-parameters model))])
      (values (cdr entry) #t)))
  (for/and ([p (in-list (optimizer-parameters opt))])
    (hash-ref named p #f)))

(define/contract-out optimizer-state/c ;; noqa
  contract?
  (rename-contract (record/c (hash "state" (string-keyed/c slots/c)
                                  "param_groups" (non-empty-listof param-group/c)))
                   'optimizer-state/c))

;; the slot tables an optimizer keeps per parameter, under torch's names; the
;; scratch tables of the foreach updates are workspace, not state
(define slot-tables
  '((momentum-buffer . "momentum_buffer") (exp-avg . "exp_avg") (exp-avg-sq . "exp_avg_sq")
    (square-avg . "square_avg") (step . "step")))

(define (kind-of root)
  (cond
    [(sgd? root) 'sgd]
    [(adam? root) 'adam]
    [(rmsprop? root) 'rmsprop]
    [else #f]))

(define (group-keys kind)
  (match kind
    ['sgd '("lr" "momentum" "nesterov" "weight_decay")]
    ['adam '("lr" "betas" "eps" "weight_decay")]
    ['rmsprop '("lr" "alpha" "eps" "weight_decay" "momentum")]
    [#f '()]))

(define (expected-slots kind group)
  (define momentum? (positive? (hash-ref group "momentum" 0)))
  (match kind
    ['sgd (if momentum? '("momentum_buffer") '())]
    ['adam '("exp_avg" "exp_avg_sq" "step")]
    ['rmsprop (if momentum? '("momentum_buffer" "square_avg") '("square_avg"))]
    [#f '()]))

(define (torch-settings h)
  (for/fold ([out (hash)]) ([(k v) (in-hash h)])
    (match k
      ['nesterov? (hash-set out "nesterov" v)]
      ['weight-decay (hash-set out "weight_decay" v)]
      [(or 'beta1 'beta2) (hash-set out "betas" (list (hash-ref h 'beta1) (hash-ref h 'beta2)))]
      [_ (hash-set out (symbol->string k) v)])))

(define (rktorch-settings group)
  (for/fold ([h (hasheq)])
            ([(k v) (in-hash group)] #:unless (member k '("params" "initial_lr")))
    (match* (k v)
      [("nesterov" _) (hash-set h 'nesterov? v)]
      [("weight_decay" _) (hash-set h 'weight-decay v)]
      [("betas" (list b1 b2)) (hash-set* h 'beta1 b1 'beta2 b2)]
      [(_ _) (hash-set h (string->symbol k) v)])))

(define (parameter-names model)
  (for/hasheq ([entry (in-list (named-parameters model))])
    (values (cdr entry) (car entry))))

(define (parameters-by-name model)
  (for/hash ([entry (in-list (named-parameters model))])
    (values (car entry) (cdr entry))))

(define (slot-value v)
  (if (tensor? v) (inexact->exact (item v)) v))

(define (parameter-slots state p)
  (for*/hash ([(table name) (in-dict slot-tables)]
              [v (in-value (hash-ref (hash-ref state table #hasheq()) p #f))]
              #:when v)
    (values name (if (equal? name "step") (slot-value v) v))))

(define (group-state g initial names)
  (define h (torch-settings (param-group-settings g)))
  (define with-params (hash-set h "params" (map (lambda (p) (hash-ref names p))
                                                (param-group-tensors g))))
  (if initial (hash-set with-params "initial_lr" initial) with-params))

(define/contract-out (optimizer-state-dict opt model) ;; noqa
  (->i ([opt optimizer?] [model layer?])
       #:pre/name (opt model) "every parameter of the optimizer is a named parameter of the model"
       (parameters-named? opt model)
       [_ optimizer-state/c])
  (define root (optimizer-root opt))
  (define names (parameter-names model))
  (define st (optimizer-store opt))
  (hash "state"
        (if (tracked? root)
            (for*/hash ([p (in-list (optimizer-parameters opt))]
                        [slots (in-value (parameter-slots (tracked-state root) p))]
                        #:unless (hash-empty? slots))
              (values (hash-ref names p) slots))
            (hash))
        "param_groups"
        (for/list ([g (in-list (store-groups st))] [initial (in-list (store-initial-lrs st))])
          (group-state g initial names))))

(define (difference xs ys)
  (filter (lambda (x) (not (member x ys))) xs))

(define (group-problems kind groups saved names)
  (for*/list ([(g i) (in-indexed groups)]
              [h (in-value (list-ref saved i))]
              [ours (in-value (map (lambda (p) (hash-ref names p)) (param-group-tensors g)))]
              [problem (in-list (append (names-problems i ours (hash-ref h "params"))
                                        (hyperparameter-problems kind i h)))])
    problem))

(define (names-problems i ours theirs)
  (append
   (match (difference ours theirs)
     ['() '()]
     [missing (list (cons "missing parameters" (format "group ~a: ~s" i missing)))])
   (match (difference theirs ours)
     ['() '()]
     [extra (list (cons "unexpected parameters" (format "group ~a: ~s" i extra)))])
   (if (= (length theirs) (length (remove-duplicates theirs)))
       '()
       (list (cons "duplicated parameters" (format "group ~a: ~s" i theirs))))))

(define (keys-fit? kind h)
  (set=? (list->set (difference (hash-keys h) '("params" "initial_lr")))
         (list->set (group-keys kind))))

(define (hyperparameter-problems kind i h)
  (define (bad key why)
    (cons "hyperparameter" (format "group ~a: ~a ~e ~a" i key (hash-ref h key) why)))
  (cond
    [(not (keys-fit? kind h))
     (list (cons "hyperparameters"
                 (format "group ~a has ~s, the optimizer takes ~s" i
                         (sort (difference (hash-keys h) '("params")) string<?)
                         (group-keys kind))))]
    [else
     (append
      (for/list ([key (in-list '("lr" "momentum" "weight_decay" "alpha" "initial_lr"))]
                 #:when (negative? (hash-ref h key 0)))
        (bad key "is negative"))
      (if (and (eq? kind 'rmsprop) (negative? (hash-ref h "eps")))
          (list (bad "eps" "is negative"))
          '())
      (if (and (hash-ref h "nesterov" #f) (not (positive? (hash-ref h "momentum"))))
          (list (bad "nesterov" "needs a momentum"))
          '()))]))

(define floating-dtypes '(float32 float64 float16 bfloat16))

(define (slot-problems kind name slots p group)
  (define expected (expected-slots kind group))
  (cond
    [(not (set=? (list->set (hash-keys slots)) (list->set expected)))
     (list (cons "slots" (format "~a has ~s, the optimizer keeps ~s" name
                                 (sort (hash-keys slots) string<?) expected)))]
    [else
     (for*/list ([(slot v) (in-hash slots)]
                 #:when (tensor? v)
                 [why (in-value (cond
                                  [(not (equal? (tensor-shape v) (tensor-shape p)))
                                   (format "shape ~s, not the parameter's ~s"
                                           (tensor-shape v) (tensor-shape p))]
                                  [(not (memq (dtype v) floating-dtypes))
                                   (format "dtype ~a, not a floating one" (dtype v))]
                                  [else #f]))]
                 #:when why)
       (cons "slot" (format "~a.~a: ~a" name slot why)))]))

(define (state-problems kind groups saved-groups saved names)
  (define owner
    (for*/hash ([(g i) (in-indexed groups)] [p (in-list (param-group-tensors g))])
      (values (hash-ref names p) (cons p i))))
  (for*/list ([(name slots) (in-hash saved)]
              [problem (in-list
                        (match (hash-ref owner name #f)
                          [#f (list (cons "state" (format "~s is not a parameter of the optimizer"
                                                          name)))]
                          [(cons p i)
                           (define group (list-ref saved-groups i))
                           (if (keys-fit? kind group)
                               (slot-problems kind name slots p group)
                               '())]))])
    problem))

(define (load-problems kind groups saved names)
  (define saved-groups (hash-ref saved "param_groups"))
  (cond
    [(not (= (length groups) (length saved-groups)))
     (list (cons "param_groups" (format "~a in the state, ~a in the optimizer"
                                        (length saved-groups) (length groups))))]
    [else
     (append (group-problems kind groups saved-groups names)
             (state-problems kind groups saved-groups (hash-ref saved "state") names))]))

(define (placed-slot fused? p v)
  (match v
    [(? tensor?)
     (define fresh (zeros-like p))
     (copy! fresh v)
     fresh]
    [n (if fused? (full (exact->inexact n) #:device (device p) #:dtype 'float32) n)]))

(define (install-slots! root kind saved by-name)
  (define state (tracked-state root))
  (define fused? (and (eq? kind 'adam) (eq? (optimizer-mode root) 'fused)))
  (hash-clear! state)
  (with-no-grad
    (for* ([(name slots) (in-hash saved)]
           [p (in-value (hash-ref by-name name))]
           [(table torch-name) (in-dict slot-tables)]
           [v (in-value (hash-ref slots torch-name #f))]
           #:when v)
      (hash-set! (hash-ref! state table make-hasheq) p (placed-slot fused? p v)))))

(define/contract-out (load-optimizer-state! opt model state) ;; noqa
  (->i ([opt optimizer?] [model layer?] [state optimizer-state/c])
       #:pre/name (opt model) "every parameter of the optimizer is a named parameter of the model"
       (parameters-named? opt model)
       [_ void?])
  (define root (optimizer-root opt))
  (define kind (kind-of root))
  (define st (optimizer-store opt))
  (define groups (store-groups st))
  (define problems (load-problems kind groups state (parameter-names model)))
  (unless (null? problems)
    (apply raise-arguments-error 'load-optimizer-state! "the state does not fit the optimizer"
           (append-map (match-lambda [(cons label why) (list label why)]) problems)))
  (define saved-groups (hash-ref state "param_groups"))
  (store-load! st (map rktorch-settings saved-groups)
               (map (lambda (h) (hash-ref h "initial_lr" #f)) saved-groups))
  (when (tracked? root)
    (install-slots! root kind (hash-ref state "state") (parameters-by-name model))))
