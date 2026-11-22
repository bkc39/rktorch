#lang racket/base

(module+ test
  (require (only-in racket/match match match* match-define)
           rackunit
           (only-in "../main.rkt"
                    * + - abs backward! copy! item max sum with-no-grad)
           (only-in "../nn.rkt"
                    Linear adam adamw in-named-parameters load-optimizer-state! named-parameters
                    optimizer-state-dict param-group param-group-set! rmsprop sgd step! zero-grads!)
           (only-in "private/optim-oracle.rkt" call-with-optim-oracle oracle-json oracle-tensors)
           (only-in "private/trainer-kit.rkt" apply-keywords))

  (define cases
    '("sgd" "sgd-momentum" "sgd-nesterov" "adam" "adamw" "rmsprop" "rmsprop-momentum"
      "sgd-momentum-fall" "sgd-momentum-rise" "rmsprop-momentum-fall"))

  (define makers (hash "sgd" sgd "adam" adam "adamw" adamw "rmsprop" rmsprop))

  (define (keywords kwargs)
    (for/fold ([kws '()]) ([(key v) (in-hash kwargs)] #:unless (eq? key 'params))
      (match key
        ['lr (cons (cons '#:lr v) kws)]
        ['momentum (cons (cons '#:momentum v) kws)]
        ['weight_decay (cons (cons '#:weight-decay v) kws)]
        ['nesterov (cons (cons '#:nesterov? v) kws)]
        ['alpha (cons (cons '#:alpha v) kws)]
        ['eps (cons (cons '#:eps v) kws)]
        ['betas
         (match-define (list beta1 beta2) v)
         (list* (cons '#:beta1 beta1) (cons '#:beta2 beta2) kws)])))

  (define (build j net)
    (define params (make-immutable-hash (named-parameters net)))
    (define groups
      (for/list ([grp (in-list (hash-ref j 'groups))])
        (apply-keywords param-group (keywords grp)
                        (for/list ([n (in-list (hash-ref grp 'params))]) (hash-ref params n)))))
    (apply-keywords (hash-ref makers (hash-ref j 'optimizer)) (keywords (hash-ref j 'kwargs))
                    groups))

  (define (load-parameters! net tensors prefix)
    (with-no-grad
      (for ([(name p) (in-named-parameters net)])
        (copy! p (hash-ref tensors (string-append prefix name))))))

  (define (gradient-steps! opt net j tensors from to)
    (define momentum-after
      (for/hash ([change (in-list (hash-ref j 'momentum_after))])
        (match-define (list k m) change)
        (values k m)))
    (for ([k (in-range from to)])
      (zero-grads! opt)
      (backward! (for/fold ([loss #f]) ([e (in-list (named-parameters net))])
                   (define grad (hash-ref tensors (format "grad.~a.~a" k (car e))))
                   (define term (sum (* (cdr e) grad)))
                   (if loss (+ loss term) term)))
      (step! opt)
      (match (hash-ref momentum-after (add1 k) #f)
        [#f (void)]
        [m (for ([i (in-range (length (hash-ref j 'groups)))])
             (param-group-set! opt i 'momentum (exact->inexact m)))])))

  (define (gap a b) (item (max (abs (- a b)))))

  (define (keys h) (sort (hash-keys h) string<?))

  (define (same-setting? mine theirs)
    (match* (mine theirs)
      [((? real?) (? real?)) (= mine theirs)]
      [((list a ...) (list b ...)) (and (= (length a) (length b)) (andmap same-setting? a b))]
      [(_ _) (equal? mine theirs)]))

  (define fixed-flags
    (hash 'dampening 0 'maximize #f 'amsgrad #f 'centered #f 'capturable #f
          'differentiable #f 'foreach 'null 'fused 'null))

  (define (fixed-value optimizer key)
    (if (eq? key 'decoupled_weight_decay)
        (equal? optimizer "adamw")
        (hash-ref fixed-flags key)))

  (define (torch-only? key)
    (or (eq? key 'decoupled_weight_decay) (hash-has-key? fixed-flags key)))

  (define (group-state optimizer grp)
    (for ([(key v) (in-hash grp)] #:when (torch-only? key))
      (unless (equal? v (fixed-value optimizer key))
        (error 'optim-state-dict-parity-test "torch's ~a is ~e, not rktorch's fixed value" key v)))
    (for/hash ([(key v) (in-hash grp)] #:unless (torch-only? key))
      (values (symbol->string key) v)))

  (define (rktorch-state j tensors)
    (define optimizer (hash-ref j 'optimizer))
    (define sd (hash-ref j 'state_dict))
    (hash "state"
          (for/hash ([(name slots) (in-hash (hash-ref sd 'state))])
            (values (symbol->string name)
                    (for/hash ([(slot v) (in-hash slots)]
                               #:unless (and (equal? optimizer "rmsprop") (eq? slot 'step)))
                      (values (symbol->string slot) (if (string? v) (hash-ref tensors v) v)))))
          "param_groups"
          (for/list ([grp (in-list (hash-ref sd 'param_groups))])
            (group-state optimizer grp))))

  (call-with-optim-oracle
   'optim-state-dict-parity-test
   (lambda (dir)
     (test-case "T3-L29: the same steps on the same gradients give torch's state"
       (for ([case (in-list cases)])
         (define j (oracle-json dir case))
         (define tensors (oracle-tensors dir case))
         (define net (Linear 3 2))
         (load-parameters! net tensors "initial.")
         (define opt (build j net))
         (gradient-steps! opt net j tensors 0 3)
         (define ours (optimizer-state-dict opt net))
         (define torch (rktorch-state j tensors))
         (check-equal? (keys (hash-ref ours "state")) (keys (hash-ref torch "state"))
                       (format "~a: the parameters with state" case))
         (for* ([(name slots) (in-hash (hash-ref torch "state"))]
                [mine-slots (in-value (hash-ref (hash-ref ours "state") name (hash)))])
           (check-equal? (keys mine-slots) (keys slots) (format "~a: ~a's slots" case name))
           (for ([(slot v) (in-hash slots)] #:when (hash-has-key? mine-slots slot))
             (define mine (hash-ref mine-slots slot))
             (define where (format "~a: ~a.~a" case name slot))
             (if (exact-integer? v)
                 (check-equal? mine v where)
                 (check-true (< (gap mine v) 1e-6) where))))
         (check-equal? (length (hash-ref ours "param_groups"))
                       (length (hash-ref torch "param_groups"))
                       (format "~a: the group count" case))
         (for ([mine (in-list (hash-ref ours "param_groups"))]
               [theirs (in-list (hash-ref torch "param_groups"))])
           (check-equal? (keys mine) (keys theirs) (format "~a: the group keys" case))
           (for ([(key v) (in-hash theirs)] #:when (hash-has-key? mine key))
             (check-true (same-setting? (hash-ref mine key) v) (format "~a: group ~a" case key))))))

     (test-case "T3-L29: torch's state loaded into rktorch steps as torch does"
       (for ([case (in-list cases)])
         (define j (oracle-json dir case))
         (define tensors (oracle-tensors dir case))
         (define net (Linear 3 2))
         (load-parameters! net tensors "after3.")
         (define opt (build j net))
         (load-optimizer-state! opt net (rktorch-state j tensors))
         (gradient-steps! opt net j tensors 3 5)
         (for ([e (in-list (named-parameters net))])
           (check-true (< (gap (cdr e) (hash-ref tensors (string-append "after5." (car e)))) 1e-6)
                       (format "~a: ~a after two more steps" case (car e)))))))))
