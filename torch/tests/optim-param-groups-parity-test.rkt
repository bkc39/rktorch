#lang racket/base

(module+ test
  (require (only-in racket/match match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    backward! cat manual-seed! randn relu reshape tensor->list)
           (only-in "../nn.rkt"
                    Linear adamw define-layer mse-loss param-group param-group-set! parameters
                    rmsprop sgd step! zero-grads!)
           (only-in "private/python-env.rkt" python-check python-torch-available?))

  (define-layer mlp-twin (fc1 fc2)
    #:init ()
    (set! fc1 (Linear 4 8))
    (set! fc2 (Linear 8 2))
    #:forward (x)
    (~> x fc1 relu fc2))

  (define (run make-opt #:after [after void])
    (manual-seed! 0)
    (define net (mlp-twin))
    (define xs (randn 16 4))
    (define ys (randn 16 2))
    (match-define (list w1 b1 w2 b2) (parameters net))
    (define opt (make-opt (list w1 w2) (list b1 b2)))
    (for ([n (in-range 1 6)])
      (zero-grads! opt)
      (~> (net xs) (mse-loss ys) backward!)
      (step! opt)
      (after opt n))
    (tensor->list (cat (for/list ([p (in-list (parameters net))]) (reshape p -1)))))

  (define (mode-keywords mode)
    (match mode
      ['loop '((#:foreach? . #f))]
      ['foreach '((#:foreach? . #t))]
      ['fused '((#:fused? . #t))]))

  (define (((grouped optimizer options group-options) mode) decay no-decay)
    (define groups
      (for/list ([ps (in-list (list decay no-decay))]
                 [kvs (in-list group-options)])
        (apply-keywords param-group kvs ps)))
    (apply-keywords optimizer (append options (mode-keywords mode)) groups))

  (define (apply-keywords proc kvs . positional)
    (match-define (list (cons kws vals) ...) (sort kvs keyword<? #:key car))
    (keyword-apply proc kws vals positional))

  (define configs
    (list (list 'adamw
                (grouped adamw '((#:lr . 1e-3))
                         '(((#:weight-decay . 0.05)) ((#:weight-decay . 0.0) (#:lr . 2e-3))))
                '(loop foreach fused))
          (list 'sgd_momentum
                (grouped sgd '((#:lr . 0.1)) '(((#:momentum . 0.9)) ((#:momentum . 0.5))))
                '(loop foreach fused))
          (list 'rmsprop_alpha
                (grouped rmsprop '((#:lr . 0.01)) '(((#:alpha . 0.9)) ((#:alpha . 0.99))))
                '(loop foreach))))

  (cond
    [(not (python-torch-available?))
     (displayln "[optim-param-groups-parity-test] skipped: python3 `torch` not available")]
    [else
     (define j (python-check "optim_param_groups.py"))
     (for ([c (in-list configs)])
       (match-define (list name make modes) c)
       (for ([mode (in-list modes)])
         (check-equal? (run (make mode))
                       (~> j (hash-ref 'groups) (hash-ref name) (hash-ref mode))
                       (format "G1-L4: ~a, ~a" name mode))))
     (check-equal? (run (lambda (decay no-decay)
                          (sgd (list (param-group (append decay no-decay))) #:lr 0.1
                               #:momentum 0.9))
                        #:after (lambda (opt n)
                                  (when (= n 2) (param-group-set! opt 0 'weight-decay 0.01))))
                   (hash-ref j 'set_weight_decay)
                   "G1-L3: weight_decay written to param_groups[0] after step 2")]))
