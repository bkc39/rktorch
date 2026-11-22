#lang racket/base

(require (only-in racket/contract/base
                  ->i contract? list/c listof non-empty-listof rename-contract)
         (only-in racket/dict in-dict)
         (only-in racket/list append* append-map check-duplicates)
         (only-in racket/match match match* match-define match-lambda)
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

(define (kept-slots kind)
  (match kind
    ['sgd (values '() '("momentum_buffer"))]
    ['adam (values '("exp_avg" "exp_avg_sq" "step") '())]
    ['rmsprop (values '("square_avg") '("momentum_buffer"))]
    [#f (values '() '())]))

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
    (match-define (cons name p) entry)
    (values p name)))

(define (copied like v)
  (define fresh (zeros-like like))
  (copy! fresh v)
  fresh)

(define (saved-slot name v)
  (cond
    [(not (tensor? v)) v]
    [(equal? name "step") (inexact->exact (item v))]
    [else (copied v v)]))

(define (parameter-slots state p)
  (for*/hash ([(table name) (in-dict slot-tables)]
              [v (in-value (hash-ref (hash-ref state table #hasheq()) p #f))]
              #:when v)
    (values name (saved-slot name v))))

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

(define (names-problems i ours theirs)
  (define (problem label names)
    (list (cons label (format "group ~a: ~s" i names))))
  (append
   (match (remove* theirs ours)
     ['() '()]
     [missing (problem "missing parameters" missing)])
   (match (remove* ours theirs)
     ['() '()]
     [extra (problem "unexpected parameters" extra)])
   (if (check-duplicates theirs) (problem "duplicated parameters" theirs) '())))

(define (keys-fit? kind h)
  (equal? (sort (remove* '("params" "initial_lr") (hash-keys h)) string<?)
          (sort (group-keys kind) string<?)))

(define (hyperparameter-defect kind h key)
  (match key
    [(or "lr" "momentum" "weight_decay" "alpha" "initial_lr")
     (and (negative? (hash-ref h key)) "is negative")]
    ["eps" (and (eq? kind 'rmsprop) (negative? (hash-ref h key)) "is negative")]
    ["nesterov" (and (hash-ref h key) (not (positive? (hash-ref h "momentum"))) "needs a momentum")]
    [_ #f]))

(define (hyperparameter-problems kind i h)
  (if (keys-fit? kind h)
      (for*/list ([key (in-list (sort (hash-keys h) string<?))]
                  [why (in-value (hyperparameter-defect kind h key))]
                  #:when why)
        (cons "hyperparameter" (format "group ~a: ~a ~e ~a" i key (hash-ref h key) why)))
      (list (cons "hyperparameters"
                  (format "group ~a has ~s, the optimizer takes ~s" i
                          (sort (remove* '("params") (hash-keys h)) string<?)
                          (group-keys kind))))))

(define (group-problems kind groups saved names)
  (append*
   (for/list ([g (in-list groups)] [h (in-list saved)] [i (in-naturals)])
     (append (names-problems i (map (lambda (p) (hash-ref names p)) (param-group-tensors g))
                             (hash-ref h "params"))
             (hyperparameter-problems kind i h)))))

(define (floating? t)
  (and (memq (dtype t) '(float32 float64 float16 bfloat16)) #t))

(define (slot-defect p v)
  (and (not (equal? (tensor-shape v) (tensor-shape p)))
       (format "shape ~s, not the parameter's ~s" (tensor-shape v) (tensor-shape p))))

(define (slot-problems kind name slots p)
  (define-values (required optional) (kept-slots kind))
  (define held (hash-keys slots))
  (if (and (null? (remove* held required)) (null? (remove* (append required optional) held)))
      (for*/list ([(slot v) (in-hash slots)]
                  #:when (tensor? v)
                  [why (in-value (slot-defect p v))]
                  #:when why)
        (cons "slot" (format "~a.~a: ~a" name slot why)))
      (list (cons "slots" (format "~a has ~s, the optimizer needs ~s and may keep ~s"
                                  name (sort held string<?) required optional)))))

(define (state-problems kind saved owned)
  (for*/list ([(name slots) (in-hash saved)]
              [problem (in-list
                        (match (hash-ref owned name #f)
                          [#f (list (cons "state" (format "~s is not a parameter of the optimizer"
                                                          name)))]
                          [p (slot-problems kind name slots p)]))])
    problem))

(define (load-problems kind groups saved names owned)
  (define saved-groups (hash-ref saved "param_groups"))
  (if (= (length groups) (length saved-groups))
      (append (group-problems kind groups saved-groups names)
              (state-problems kind (hash-ref saved "state") owned))
      (list (cons "param_groups" (format "~a in the state, ~a in the optimizer"
                                         (length saved-groups) (length groups))))))

(define (placed-slot fused? p v)
  (match v
    [(? tensor?) (if (floating? p) (copied p v) (copied (zeros-like p #:dtype (dtype v)) v))]
    [n (if fused? (full (exact->inexact n) #:device (device p) #:dtype 'float32) n)]))

(define (loaded-tables fused? saved owned)
  (define tables (make-hasheq))
  (with-no-grad
    (for* ([(name slots) (in-hash saved)]
           [p (in-value (hash-ref owned name))]
           [(table torch-name) (in-dict slot-tables)]
           [v (in-value (hash-ref slots torch-name #f))]
           #:when v)
      (hash-set! (hash-ref! tables table make-hasheq) p (placed-slot fused? p v))))
  tables)

(define (install-tables! state tables)
  (hash-clear! state)
  (for ([(table slots) (in-hash tables)])
    (hash-set! state table slots)))

(define/contract-out (load-optimizer-state! opt model state) ;; noqa
  (->i ([opt optimizer?] [model layer?] [state optimizer-state/c])
       #:pre/name (opt model) "every parameter of the optimizer is a named parameter of the model"
       (parameters-named? opt model)
       [_ void?])
  (define root (optimizer-root opt))
  (define kind (kind-of root))
  (define st (optimizer-store opt))
  (define groups (store-groups st))
  (define names (parameter-names model))
  (define owned
    (for*/hash ([g (in-list groups)] [p (in-list (param-group-tensors g))])
      (values (hash-ref names p) p)))
  (define problems (load-problems kind groups state names owned))
  (unless (null? problems)
    (apply raise-arguments-error 'load-optimizer-state! "the state does not fit the optimizer"
           (append-map (match-lambda [(cons label why) (list label why)]) problems)))
  (define saved-groups (hash-ref state "param_groups"))
  (define settings (map rktorch-settings saved-groups))
  (define initial-lrs (map (lambda (h) (hash-ref h "initial_lr" #f)) saved-groups))
  (define tables
    (and (tracked? root)
         (loaded-tables (and (eq? kind 'adam) (eq? (optimizer-mode root) 'fused))
                        (hash-ref state "state") owned)))
  (parameterize-break #f
    (store-load! st settings initial-lrs)
    (when tables
      (install-tables! (tracked-state root) tables))))
