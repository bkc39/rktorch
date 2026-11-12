#lang racket/base

(module+ test
  (require (only-in racket/list filter-not last remove-duplicates split-at)
           (only-in racket/match match match-define)
           (only-in racket/string string-split)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    backward! manual-seed! randn requires-grad! reshape shape sum tensor->list)
           (only-in "../nn.rkt"
                    Linear Parameter Sequential adam adamw cosine-annealing-lr exponential-lr
                    lambda-lr learning-rate linear-lr multi-step-lr named-parameters
                    one-cycle-lr optimizer-param-groups param-group param-group-count
                    param-group-hyperparameters param-group-params param-group-ref
                    param-group-set! parameters rmsprop scheduler-rate set-learning-rate! sgd
                    step! step-lr weight-decay-param-groups zero-grads!)
           (only-in (submod "../nn/optim.rkt" private) optimizer-state)
           (only-in "../vision/vit.rkt" VisionTransformer)
           (only-in "private/blame.rkt" blamed contract-violation?)
           (only-in "private/op-calls.rkt" crossings-of)
           (only-in "private/within.rkt" bounded-test))

  (define this-test (blamed "optim-param-groups-test"))

  (struct config (name optimizer options modes))

  (define configs
    (list (config "sgd" sgd '((#:lr . 0.1) (#:momentum . 0.9)) '(loop foreach fused))
          (config "adam" adam '((#:lr . 0.01)) '(loop foreach fused))
          (config "adamw" adamw '((#:lr . 0.01) (#:weight-decay . 0.05)) '(loop foreach fused))
          (config "rmsprop" rmsprop '((#:lr . 0.01) (#:momentum . 0.5)) '(loop foreach))))

  (define (mode-option mode)
    (match mode
      ['loop '((#:foreach? . #f))]
      ['foreach '((#:foreach? . #t))]
      ['fused '((#:fused? . #t))]))

  (define (build optimizer options params)
    (match-define (list (cons kws vals) ...)
      (sort (remove-duplicates options eq? #:key car) keyword<? #:key car))
    (keyword-apply optimizer kws vals (list params)))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (mlp)
    (manual-seed! 0)
    (Sequential (Linear 4 8) (Linear 8 2)))

  (define (train! net opts [steps 5] #:after [after void])
    (manual-seed! 1)
    (define x (randn 3 4))
    (for ([n (in-range 1 (add1 steps))])
      (for-each zero-grads! opts)
      (~> x net sum backward!)
      (for-each step! opts)
      (after n))
    (map flat (parameters net)))

  (define (halves net)
    (split-at (parameters net) 2))

  (for* ([c (in-list configs)] [mode (in-list (config-modes c))])
    (match-define (config name optimizer options _) c)
    (define opts (append options (mode-option mode)))
    (bounded-test (format "G1-L1: ~a, ~a: two like groups step as one list" name mode)
      (define one (mlp))
      (define two (mlp))
      (define-values (front back) (halves two))
      (check-equal? (train! two (list (build optimizer opts
                                             (list (param-group front) (param-group back)))))
                    (train! one (list (build optimizer opts (parameters one))))))
    (bounded-test (format "G1-L2: ~a, ~a: groups with their own rates step as two optimizers"
                          name mode)
      (define grouped (mlp))
      (define split (mlp))
      (define-values (front back) (halves grouped))
      (define-values (front* back*) (halves split))
      (check-equal?
       (train! grouped (list (build optimizer opts
                                    (list (param-group front #:lr 0.03)
                                          (param-group back #:lr 0.002)))))
       (train! split (list (build optimizer (cons '(#:lr . 0.03) opts) front*)
                           (build optimizer (cons '(#:lr . 0.002) opts) back*))))))

  (bounded-test "G1-L3: a weight decay written after step 2 is the decay from step 3 on"
    (define written (mlp))
    (define rebuilt (mlp))
    (define opt (sgd (list (param-group (parameters written))) #:lr 0.1))
    (define after-two
      (train! written (list opt) #:after (lambda (n)
                                           (when (= n 2)
                                             (param-group-set! opt 0 'weight-decay 0.01)))))
    (check-equal? (param-group-ref opt 0 'weight-decay) 0.01)
    (train! rebuilt (list (sgd (parameters rebuilt) #:lr 0.1)) 2)
    (check-equal? after-two
                  (train! rebuilt (list (sgd (parameters rebuilt) #:lr 0.1 #:weight-decay 0.01))
                          3)))

  (bounded-test "G1-L3: a written hyperparameter keeps the momentum buffers"
    (define net (mlp))
    (define opt (sgd (parameters net) #:lr 0.1 #:momentum 0.9))
    (train! net (list opt) 2)
    (define buffers (optimizer-state opt))
    (param-group-set! opt 0 'weight-decay 0.01)
    (check-equal? (length buffers) (length (parameters net)))
    (check-true (andmap eq? (optimizer-state opt) buffers)))

  (define (steady-crossings groups depth #:optimizer [optimizer adam] #:options [options '()])
    (manual-seed! 0)
    (define nets
      (for/list ([_ (in-range groups)])
        (apply Sequential (for/list ([_ (in-range depth)]) (Linear 3 3)))))
    (define opt
      (build optimizer options (for/list ([net (in-list nets)]) (param-group (parameters net)))))
    (define x (randn 2 3))
    (for/last ([_ (in-range 4)])
      (zero-grads! opt)
      (for ([net (in-list nets)]) (~> x net sum backward!))
      (crossings-of (lambda () (step! opt)))))

  (bounded-test "G1-L5: steady crossings grow by a constant per group, at any depth"
    (define cs (for/list ([g (in-range 1 4)]) (steady-crossings g 1)))
    (match-define (list c1 c2 c3) cs)
    (check-equal? (- c3 c2) (- c2 c1))
    (check-equal? (for/list ([g (in-range 1 4)]) (steady-crossings g 2)) cs))

  (bounded-test "G1-L6: an adamw group without decay saves its decay crossings, same bits"
    (define (adamw-crossings mode wd)
      (steady-crossings 1 1 #:optimizer adamw
                        #:options (cons `(#:weight-decay . ,wd) (mode-option mode))))
    (check-equal? (for/list ([mode (in-list '(loop foreach fused))])
                    (- (adamw-crossings mode 0.01) (adamw-crossings mode 0.0)))
                  (list (length (parameters (Linear 3 3))) 1 0))
    (define decoupled (mlp))
    (define plain (mlp))
    (check-equal? (train! decoupled (list (adamw (parameters decoupled) #:weight-decay 0.0)))
                  (train! plain (list (adam (parameters plain))))))

  (bounded-test "G1-L7: a single group steps with a flat list's crossings"
    (manual-seed! 0)
    (define (crossings params)
      (define net (Linear 3 3))
      (define opt (~> (parameters net) params (sgd #:lr 0.1 #:momentum 0.9)))
      (for/last ([_ (in-range 4)])
        (zero-grads! opt)
        (~> (randn 2 3) net sum backward!)
        (crossings-of (lambda () (step! opt)))))
    (check-equal? (crossings (lambda (ps) (list (param-group ps)))) (crossings values)))

  (bounded-test "G1-L8: weight-decay-param-groups splits a ViT by rank and name"
    (define vit (VisionTransformer #:image-size 8 #:patch 4 #:width 8 #:depth 1 #:heads 2
                                   #:classes 3))
    (define named (named-parameters vit))
    (define frozen (map cdr (filter (lambda (n) (member (car n) '("head.weight" "head.bias")))
                                    named)))
    (check-equal? (map (lambda (t) (length (shape t))) frozen) '(2 1))
    (for ([t (in-list frozen)]) (requires-grad! t #f))
    (define trainable (filter-not (lambda (n) (memq (cdr n) frozen)) named))
    (define (exempt? n)
      (or (~> (cdr n) shape length (<= 1))
          (~> (car n) (string-split ".") last (member '("cls-token" "pos-embed")))))
    (match-define (list decay no-decay) (weight-decay-param-groups vit #:weight-decay 0.05))
    (check-equal? (param-group-params decay) (map cdr (filter (lambda (n) (not (exempt? n)))
                                                              trainable)))
    (check-equal? (param-group-params no-decay) (map cdr (filter exempt? trainable)))
    (check-equal? (param-group-hyperparameters decay) (hasheq 'weight-decay 0.05))
    (check-equal? (param-group-hyperparameters no-decay) (hasheq 'weight-decay 0.0))
    (match-define (list inherit _) (weight-decay-param-groups vit))
    (check-equal? (param-group-hyperparameters inherit) (hasheq)))

  (define (two-group-sgd)
    (sgd (list (~> (Linear 2 2) parameters (param-group #:lr 0.1))
               (~> (Linear 2 2) parameters (param-group #:lr 0.02)))
         #:lr 0.1))

  (define (rmsprop-groups)
    (rmsprop (for/list ([_ (in-range 2)]) (~> (Linear 2 2) parameters param-group))))

  (define bad-calls
    `(("a parameter in two groups" #rx"a parameter appears in more than one parameter group"
       ,(lambda ()
          (define ps (parameters (Linear 2 2)))
          (sgd (list (param-group ps) (param-group (list (car ps)))) #:lr 0.1)))
      ("momentum on an adam group" #rx"sets a hyperparameter this optimizer does not have"
       ,(lambda () (adam (list (~> (Linear 2 2) parameters (param-group #:momentum 0.9))))))
      ("a group index past the last in param-group-ref" #rx"no parameter group with this index"
       ,(lambda () (param-group-ref (two-group-sgd) 2 'lr)))
      ("a group index past the last in param-group-set!" #rx"no parameter group with this index"
       ,(lambda () (param-group-set! (two-group-sgd) 2 'lr 0.1)))
      ("a group index past the last in learning-rate" #rx"no parameter group with this index"
       ,(lambda () (learning-rate (two-group-sgd) #:group 2)))
      ("a negative rate written" #rx"expected: [(]>=/c 0[)]"
       ,(lambda () (param-group-set! (two-group-sgd) 0 'lr -1.0)))
      ("'beta1 on sgd" #rx"this optimizer has no such hyperparameter"
       ,(lambda () (param-group-ref (two-group-sgd) 0 'beta1)))
      ("'beta1 written to sgd" #rx"this optimizer has no such hyperparameter"
       ,(lambda () (param-group-set! (two-group-sgd) 0 'beta1 0.9)))
      ("a negative eps written to rmsprop" #rx"expected: [(]>=/c 0[)]"
       ,(lambda () (param-group-set! (rmsprop-groups) 1 'eps -1e-8)))))

  (for ([case (in-list bad-calls)])
    (match-define (list label rx call) case)
    (bounded-test (format "G1-L9: ~a blames the caller" label)
      (check-exn (contract-violation? this-test rx) call)))

  (bounded-test "a snapshot has every key filled and ignores later writes"
    (define opt (adam (list (~> (Linear 2 2) parameters (param-group #:lr 1/100)))))
    (match-define (list snapshot) (optimizer-param-groups opt))
    (define filled (hasheq 'lr 0.01 'beta1 0.9 'beta2 0.999 'eps 1e-8 'weight-decay 0.0))
    (check-equal? (param-group-hyperparameters snapshot) filled)
    (param-group-set! opt 0 'lr 0.5)
    (check-equal? (param-group-hyperparameters snapshot) filled)
    (check-equal? (param-group-ref opt 0 'lr) 0.5))

  (for ([mode (in-list '(loop foreach fused))])
    (bounded-test (format "a group without gradients is left alone, ~a" mode)
      (define used (mlp))
      (define idle (Linear 2 2))
      (define before (map flat (parameters idle)))
      (define moved
        (train! used (list (build adam (cons '(#:lr . 0.01) (mode-option mode))
                                  (list (param-group (parameters used))
                                        (param-group (parameters idle) #:lr 0.1))))))
      (check-equal? (map flat (parameters idle)) before)
      (check-not-equal? moved (map flat (parameters (mlp))))))

  (define race-writes 50000)

  (define (lost-writes opt i key)
    (for/sum ([k (in-range 1 (add1 race-writes))])
      (define v (* 1e-6 k))
      (param-group-set! opt i key v)
      (if (eqv? (param-group-ref opt i key) v) 0 1)))

  (bounded-test "a write from another thread is never lost"
    (define opt (two-group-sgd))
    (define other (box #f))
    (define t (thread (lambda () (set-box! other (lost-writes opt 1 'weight-decay)))))
    (define here (lost-writes opt 0 'lr))
    (thread-wait t)
    (check-equal? (list here (unbox other)) '(0 0))
    (check-equal? (list (param-group-ref opt 0 'lr) (param-group-ref opt 1 'weight-decay))
                  (list (* 1e-6 race-writes) (* 1e-6 race-writes))))

  (bounded-test "a write while a schedule steps on another thread is never lost"
    (define opt (two-group-sgd))
    (define s (step-lr opt #:step-size 3 #:gamma 0.5))
    (define t (thread (lambda () (for ([_ (in-range race-writes)]) (step! s)))))
    (define lost (lost-writes opt 1 'weight-decay))
    (thread-wait t)
    (check-equal? lost 0)
    (check-equal? (param-group-ref opt 1 'weight-decay) (* 1e-6 race-writes)))

  (define shapes
    `(("step" ,(lambda (o) (step-lr o #:step-size 3 #:gamma 0.5)))
      ("multi_step" ,(lambda (o) (multi-step-lr o #:milestones '(2 5 9))))
      ("exponential" ,(lambda (o) (exponential-lr o #:gamma 0.9)))
      ("cosine" ,(lambda (o) (cosine-annealing-lr o #:t-max 10 #:eta-min 0.01)))
      ("linear" ,(lambda (o) (linear-lr o #:start-factor 0.25 #:total-iters 4)))
      ("one_cycle" ,(lambda (o) (one-cycle-lr o #:max-lr 0.1 #:total-steps 12)))
      ("lambda" ,(lambda (o) (lambda-lr o (lambda (t) (/ 1.0 (add1 t))))))))

  (define (single-rates make base)
    (define s (make (sgd (list (Parameter (randn 1))) #:lr base)))
    (for/list ([_ (in-range 12)]) (step! s) (scheduler-rate s)))

  (for ([shape (in-list shapes)])
    (match-define (list name make) shape)
    (bounded-test (format "G1-L10: ~a writes each group's closed form from its own base" name)
      (define opt (two-group-sgd))
      (define s (make opt))
      (define rates
        (for/list ([_ (in-range 12)])
          (step! s)
          (list (scheduler-rate s #:group 0) (scheduler-rate s #:group 1)
                (learning-rate opt #:group 0) (learning-rate opt #:group 1))))
      (match-define (list first second first-lr second-lr) (apply map list rates))
      (check-equal? first (single-rates make 0.1))
      (check-equal? second (single-rates make 0.02))
      (check-equal? first-lr first)
      (check-equal? second-lr second)))

  (bounded-test "G1-L11: without #:group the rate reads group 0 and writes every group"
    (define opt (two-group-sgd))
    (check-equal? (param-group-count opt) 2)
    (check-equal? (learning-rate opt) 0.1)
    (check-equal? (learning-rate opt #:group 1) 0.02)
    (set-learning-rate! opt 0.5)
    (check-equal? (list (learning-rate opt #:group 0) (learning-rate opt #:group 1)) '(0.5 0.5))
    (set-learning-rate! opt 0.25 #:group 1)
    (check-equal? (list (learning-rate opt) (learning-rate opt #:group 1)) '(0.5 0.25))))
