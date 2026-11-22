#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt" backward! dtype grad-enabled? make-generator randn to zeros)
           (only-in "../nn.rkt"
                    Linear adam adamw cosine-annealing-lr layer-copy load-optimizer-state! mse-loss
                    optimizer-state-dict param-group-set! parameters rmsprop sgd step!
                    weight-decay-param-groups zero-grads!)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/comparable.rkt" comparable layer-bytes)
           (only-in "private/trainer-kit.rkt" apply-keywords))

  (define blames-this-test? (blames-caller? "optim-state-dict-test"))

  (define g (make-generator 5))
  (define x (randn 4 3 #:generator g))
  (define y (randn 4 2 #:generator g))

  (define (fresh-net) (Linear 3 2))

  (define cases
    `(("sgd" ,sgd ((#:lr . 0.1)) (loop foreach fused))
      ("sgd momentum 0.9" ,sgd ((#:lr . 0.1) (#:momentum . 0.9) (#:weight-decay . 0.01))
                          (loop foreach fused))
      ("sgd nesterov" ,sgd ((#:lr . 0.1) (#:momentum . 0.9) (#:nesterov? . #t))
                      (loop foreach fused))
      ("adam" ,adam ((#:lr . 0.01) (#:beta2 . 0.99) (#:weight-decay . 0.01)) (loop foreach fused))
      ("adamw" ,adamw ((#:lr . 0.01)) (loop foreach fused))
      ("rmsprop" ,rmsprop ((#:lr . 0.01) (#:alpha . 0.9)) (loop foreach))
      ("rmsprop momentum 0.9" ,rmsprop ((#:lr . 0.01) (#:alpha . 0.9) (#:momentum . 0.9))
                              (loop foreach))))

  (define (mode-keywords mode)
    (case mode
      [(loop) '((#:foreach? . #f))]
      [(foreach) '()]
      [(fused) '((#:fused? . #t))]))

  (define (build make kws mode net)
    (apply-keywords make (append kws (mode-keywords mode)) (parameters net)))

  (define (steps! opt net n)
    (for ([_ (in-range n)])
      (zero-grads! opt)
      (backward! (mse-loss (net x) y))
      (step! opt)))

  (test-case "T3-L10: 3 steps, dict, load into a fresh optimizer, 2 more steps on both"
    (for* ([c (in-list cases)] [mode (in-list (list-ref c 3))])
      (match-define (list name make kws _modes) c)
      (define label (format "~a, ~a" name mode))
      (define net (fresh-net))
      (define opt (build make kws mode net))
      (steps! opt net 3)
      (define saved (optimizer-state-dict opt net))
      (define twin (layer-copy net))
      (define twin-opt (build make kws mode twin))
      (load-optimizer-state! twin-opt twin saved)
      (check-equal? (comparable (optimizer-state-dict twin-opt twin)) (comparable saved)
                    (format "~a: the dict reads back as loaded" label))
      (steps! opt net 2)
      (steps! twin-opt twin 2)
      (check-equal? (layer-bytes twin) (layer-bytes net) (format "~a: parameters" label))
      (check-equal? (comparable (optimizer-state-dict twin-opt twin))
                    (comparable (optimizer-state-dict opt net))
                    (format "~a: dicts after the steps" label))))

  (test-case "T3-L10: a state moves between modes and reads back exactly"
    (for* ([c (in-list cases)]
           [from (in-list (list-ref c 3))]
           [to-mode (in-list (list-ref c 3))])
      (match-define (list name make kws _modes) c)
      (define net (fresh-net))
      (define opt (build make kws from net))
      (steps! opt net 3)
      (define saved (optimizer-state-dict opt net))
      (define other (build make kws to-mode net))
      (load-optimizer-state! other net saved)
      (check-equal? (comparable (optimizer-state-dict other net)) (comparable saved)
                    (format "~a: ~a -> ~a" name from to-mode))))

  (define momentum-changes
    `(("sgd 0.9 -> 0" ,sgd ((#:lr . 0.1) (#:momentum . 0.9)) 0.0 0.9 (loop foreach fused))
      ("sgd 0 -> 0.9" ,sgd ((#:lr . 0.1)) 0.9 0.0 (loop foreach fused))
      ("rmsprop 0.9 -> 0" ,rmsprop ((#:lr . 0.01) (#:momentum . 0.9)) 0.0 0.9 (loop foreach))
      ("rmsprop 0 -> 0.9" ,rmsprop ((#:lr . 0.01)) 0.9 0.0 (loop foreach))))

  (test-case "T3-L10: a momentum changed before the save, in either direction"
    (for* ([c (in-list momentum-changes)] [mode (in-list (list-ref c 5))] [after (in-range 2)])
      (match-define (list name make kws changed back _modes) c)
      (define label (format "~a, ~a, ~a step(s) after the change" name mode after))
      (define net (fresh-net))
      (define opt (build make kws mode net))
      (steps! opt net 2)
      (param-group-set! opt 0 'momentum changed)
      (steps! opt net after)
      (define saved (optimizer-state-dict opt net))
      (define twin (layer-copy net))
      (define twin-opt (build make kws mode twin))
      (load-optimizer-state! twin-opt twin saved)
      (check-equal? (comparable (optimizer-state-dict twin-opt twin)) (comparable saved) label)
      (for ([o (list opt twin-opt)] [n (list net twin)])
        (steps! o n 2)
        (param-group-set! o 0 'momentum back)
        (steps! o n 2))
      (check-equal? (layer-bytes twin) (layer-bytes net) (format "~a: parameters" label))
      (check-equal? (comparable (optimizer-state-dict twin-opt twin))
                    (comparable (optimizer-state-dict opt net))
                    (format "~a: dicts after the steps" label))))

  (test-case "T3-L10: a dict held across a step is the state when it was taken"
    (for* ([c (in-list cases)] [mode (in-list (list-ref c 3))])
      (match-define (list name make kws _modes) c)
      (define label (format "~a, ~a" name mode))
      (define net (fresh-net))
      (define opt (build make kws mode net))
      (steps! opt net 1)
      (define held (optimizer-state-dict opt net))
      (define taken (comparable held))
      (define twin (layer-copy net))
      (steps! opt net 1)
      (check-equal? (comparable held) taken (format "~a: the held dict" label))
      (define twin-opt (build make kws mode twin))
      (load-optimizer-state! twin-opt twin held)
      (steps! twin-opt twin 1)
      (check-equal? (layer-bytes twin) (layer-bytes net) (format "~a: parameters" label))))

  (test-case "T3-L10: the state of an optimizer that never stepped"
    (for ([c (in-list cases)])
      (match-define (list name make kws _modes) c)
      (define net (fresh-net))
      (define opt (build make kws 'foreach net))
      (define saved (optimizer-state-dict opt net))
      (check-equal? (hash-ref saved "state") (hash) name)
      (define twin-opt (build make kws 'foreach net))
      (load-optimizer-state! twin-opt net saved)
      (check-equal? (comparable (optimizer-state-dict twin-opt net)) (comparable saved) name)))

  (test-case "T3-L10: two groups keep their own hyperparameters, initial_lr included"
    (define net (fresh-net))
    (define opt (adamw (weight-decay-param-groups net #:weight-decay 0.1) #:lr 0.01))
    (void (cosine-annealing-lr opt #:t-max 6))
    (steps! opt net 3)
    (define saved (optimizer-state-dict opt net))
    (define (group-values key state)
      (map (lambda (grp) (hash-ref grp key #f)) (hash-ref state "param_groups")))
    (check-equal? (length (hash-ref saved "param_groups")) 2)
    (check-equal? (group-values "weight_decay" saved) '(0.1 0.0))
    (check-equal? (group-values "initial_lr" saved) '(0.01 0.01) "the scheduler's initial_lr")
    (define twin-opt (adamw (weight-decay-param-groups net #:weight-decay 0.1) #:lr 0.5))
    (load-optimizer-state! twin-opt net saved)
    (check-equal? (group-values "initial_lr" (optimizer-state-dict twin-opt net)) '(0.01 0.01)
                  "initial_lr loads into an optimizer that had none"))

  (test-case "T3-L10: a group hyperparameter given exact saves as a flonum"
    (define net (fresh-net))
    (define saved (optimizer-state-dict (sgd (parameters net) #:lr 1/10) net))
    (check-eqv? (hash-ref (car (hash-ref saved "param_groups")) "lr") 0.1))

  (test-case "T3-B16: a model moved after its optimizer was built keeps its names"
    (define net (fresh-net))
    (define opt (adam (parameters net) #:lr 0.01))
    (steps! opt net 1)
    (to net 'float64)
    (zero-grads! opt)
    (backward! (mse-loss (net (to x 'float64)) (to y 'float64)))
    (step! opt)
    (define saved (optimizer-state-dict opt net))
    (check-equal? (sort (hash-keys (hash-ref saved "state")) string<?) '("bias" "weight"))
    (check-equal? (dtype (hash-ref (hash-ref (hash-ref saved "state") "weight") "exp_avg"))
                  'float64))

  (define (adam-after-steps net)
    (define opt (adam (parameters net) #:lr 0.01))
    (steps! opt net 3)
    (values opt (optimizer-state-dict opt net)))

  (define (check-refused label opt net state rx)
    (define before (comparable (optimizer-state-dict opt net)))
    (check-exn rx
               (lambda () (load-optimizer-state! opt net state))
               label)
    (check-equal? (comparable (optimizer-state-dict opt net)) before
                  (format "~a: the optimizer is unchanged" label)))

  (define (update-group state f)
    (hash-update state "param_groups" (lambda (groups) (cons (f (car groups)) (cdr groups)))))

  (define (update-slot state name slot f)
    (hash-update state "state"
                 (lambda (st) (hash-update st name (lambda (slots) (hash-update slots slot f))))))

  (test-case "T3-L11: a state that does not fit the optimizer raises, naming the entry"
    (define net (fresh-net))
    (define-values (opt saved) (adam-after-steps net))
    (define two-groups
      (optimizer-state-dict (adam (weight-decay-param-groups net #:weight-decay 0.1)) net))
    (check-refused "group count" opt net two-groups #rx"param_groups")
    (check-refused "a missing name" opt net
                   (update-group saved (lambda (grp) (hash-set grp "params" '("weight"))))
                   #rx"bias")
    (check-refused "an extra name" opt net
                   (update-group saved (lambda (grp) (hash-update grp "params"
                                                                  (lambda (ps) (cons "ghost" ps)))))
                   #rx"ghost")
    (check-refused "a slot's shape" opt net
                   (update-slot saved "weight" "exp_avg" (lambda (_) (zeros 3)))
                   #rx"weight.*exp_avg|exp_avg.*weight")
    (check-refused "a slot's dtype" opt net
                   (update-slot saved "weight" "exp_avg" (lambda (t) (to t 'int64)))
                   #rx"weight.*exp_avg|exp_avg.*weight"))

  (test-case "T3-L11: a load that fails while placing the slots leaves the optimizer unchanged"
    (define net (fresh-net))
    (define-values (opt saved) (adam-after-steps net))
    (define (fails-without-grad slots)
      (chaperone-hash slots
                      (lambda (h k)
                        (values k (lambda (h k v)
                                    (if (grad-enabled?) v (error 'placing "injected failure")))))
                      (lambda (h k v) (values k v))
                      (lambda (h k) k)
                      (lambda (h k) k)))
    (check-refused "an injected failure" opt net
                   (~> saved
                       (update-group (lambda (grp) (hash-set grp "lr" 0.5)))
                       (hash-update "state" (lambda (st) (hash-update st "weight" fails-without-grad))))
                   #rx"injected failure"))

  (test-case "T3-L10: a floating slot takes its parameter's dtype, as torch's load does"
    (define net (fresh-net))
    (define-values (opt saved) (adam-after-steps net))
    (load-optimizer-state! opt net
                           (update-slot saved "weight" "exp_avg" (lambda (t) (to t 'float64))))
    (check-equal? (comparable (optimizer-state-dict opt net)) (comparable saved)))

  (test-case "T3-L11: hyperparameters and slots that do not fit the optimizer raise"
    (define net (fresh-net))
    (define-values (opt saved) (adam-after-steps net))
    (check-refused "a key another optimizer has" opt net
                   (update-group saved (lambda (grp) (hash-set grp "alpha" 0.9)))
                   #rx"alpha")
    (check-refused "a negative rate" opt net
                   (update-group saved (lambda (grp) (hash-set grp "lr" -1.0)))
                   #rx"lr.*negative")
    (check-refused "a duplicated name" opt net
                   (update-group saved
                                 (lambda (grp) (hash-set grp "params" '("weight" "bias" "bias"))))
                   #rx"duplicated")
    (check-refused "a slot the optimizer does not keep" opt net
                   (hash-update saved "state"
                                (lambda (st)
                                  (hash-update st "weight" (lambda (sl) (hash-remove sl "exp_avg")))))
                   #rx"weight")
    (check-refused "a state for a parameter outside the optimizer" opt net
                   (hash-update saved "state" (lambda (st) (hash-set st "ghost" (hash))))
                   #rx"ghost"))

  (test-case "T3-L11: an sgd or rmsprop group's own constraints"
    (define net (fresh-net))
    (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
    (steps! opt net 1)
    (define saved (optimizer-state-dict opt net))
    (check-refused "nesterov without momentum" opt net
                   (update-group saved (lambda (grp) (hash-set* grp "nesterov" #t "momentum" 0.0)))
                   #rx"nesterov")
    (define r (rmsprop (parameters net) #:lr 0.1))
    (steps! r net 1)
    (check-refused "a negative eps" r net
                   (update-group (optimizer-state-dict r net)
                                 (lambda (grp) (hash-set grp "eps" -1.0)))
                   #rx"eps"))

  (test-case "T3-L11: a malformed state, or a parameter outside the model, blames the caller"
    (define net (fresh-net))
    (define-values (opt saved) (adam-after-steps net))
    (check-exn blames-this-test? (lambda () (load-optimizer-state! opt net (hash "state" (hash)))))
    (check-exn blames-this-test?
               (lambda () (load-optimizer-state! opt net (hash-set saved "state" 3))))
    (check-exn blames-this-test? (lambda () (optimizer-state-dict opt (fresh-net))))
    (check-exn blames-this-test? (lambda () (load-optimizer-state! opt (fresh-net) saved)))))
