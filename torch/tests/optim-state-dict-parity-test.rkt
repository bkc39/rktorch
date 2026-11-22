#lang racket/base

(module+ test
  (require (only-in racket/match match match*)
           rackunit
           (only-in "../main.rkt"
                    * + - abs backward! copy! item max sum with-no-grad)
           (only-in "../nn.rkt"
                    Linear adam adamw load-optimizer-state! named-parameters
                    optimizer-state-dict param-group rmsprop sgd step! zero-grads!)
           (only-in "private/optim-oracle.rkt" call-with-optim-oracle oracle-json oracle-tensors)
           (only-in "private/trainer-kit.rkt" apply-keywords))

  (define cases '("sgd" "sgd-momentum" "sgd-nesterov" "adam" "adamw" "rmsprop" "rmsprop-momentum"))

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
        ['betas (list* (cons '#:beta1 (car v)) (cons '#:beta2 (cadr v)) kws)])))

  (define (build j net)
    (define params (for/hash ([e (in-list (named-parameters net))]) (values (car e) (cdr e))))
    (define groups
      (for/list ([grp (in-list (hash-ref j 'groups))])
        (apply-keywords param-group (keywords grp)
                        (for/list ([n (in-list (hash-ref grp 'params))]) (hash-ref params n)))))
    (apply-keywords (hash-ref makers (hash-ref j 'optimizer)) (keywords (hash-ref j 'kwargs))
                    groups))

  (define (load-parameters! net tensors prefix)
    (with-no-grad
      (for ([e (in-list (named-parameters net))])
        (copy! (cdr e) (hash-ref tensors (string-append prefix (car e)))))))

  (define (gradient-steps! opt net tensors from to)
    (for ([k (in-range from to)])
      (zero-grads! opt)
      (backward! (for/fold ([loss #f]) ([e (in-list (named-parameters net))])
                   (define term (sum (* (cdr e) (hash-ref tensors (format "grad.~a.~a" k (car e))))))
                   (if loss (+ loss term) term)))
      (step! opt)))

  (define (gap a b) (item (max (abs (- a b)))))

  ;; torch's defaults are Python ints (`weight_decay: float = 0`), rktorch's store flonums
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
         (gradient-steps! opt net tensors 0 3)
         (define ours (optimizer-state-dict opt net))
         (define torch (rktorch-state j tensors))
         (for* ([(name slots) (in-hash (hash-ref torch "state"))]
                [(slot v) (in-hash slots)])
           (define mine (hash-ref (hash-ref (hash-ref ours "state") name) slot))
           (define where (format "~a: ~a.~a" case name slot))
           (if (exact-integer? v)
               (check-equal? mine v where)
               (check-true (< (gap mine v) 1e-6) where)))
         (for ([mine (in-list (hash-ref ours "param_groups"))]
               [theirs (in-list (hash-ref torch "param_groups"))])
           (for ([(key v) (in-hash theirs)])
             (check-true (same-setting? (hash-ref mine key) v) (format "~a: group ~a" case key))))))

     (test-case "T3-L29: torch's state loaded into rktorch steps as torch does"
       (for ([case (in-list cases)])
         (define j (oracle-json dir case))
         (define tensors (oracle-tensors dir case))
         (define net (Linear 3 2))
         (load-parameters! net tensors "after3.")
         (define opt (build j net))
         (load-optimizer-state! opt net (rktorch-state j tensors))
         (gradient-steps! opt net tensors 3 5)
         (for ([e (in-list (named-parameters net))])
           (check-true (< (gap (cdr e) (hash-ref tensors (string-append "after5." (car e)))) 1e-6)
                       (format "~a: ~a after two more steps" case (car e)))))))))
