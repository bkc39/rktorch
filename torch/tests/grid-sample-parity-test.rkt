#lang racket/base

;; Every draw stays in the order of the twin, torch/tests/python/grid_sample.py.

(module+ test
  (require rackunit
           (only-in racket/match match-define)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * - affine-grid backward! cuda-available? grad grid-sample
                    in-flattened-tensor manual-seed! rand randn requires-grad!
                    shape sum to-device upsample-bicubic2d with-default-device)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available?))

  (define (leaf t device)
    (requires-grad! (to-device t device)))

  (define (weighed out device)
    (define w (~> out shape randn (to-device device)))
    (~> (* out w) sum backward!))

  (define (affine device align?)
    (define theta (leaf (randn 2 2 3) device))
    (define out (affine-grid theta '(2 3 4 5) #:align-corners? align?))
    (weighed out device)
    (values out (list theta)))

  (define (sampled device mode padding align?)
    (define x (leaf (randn 2 3 4 5) device))
    (define grid (leaf (~> (rand 2 6 7 2) (* 2.4) (- 1.2)) device))
    (define out (grid-sample x grid #:mode mode #:padding-mode padding
                             #:align-corners? align?))
    (weighed out device)
    (values out (list x grid)))

  (define (resized device align?)
    (define x (leaf (randn 1 2 5 6) device))
    (define out (upsample-bicubic2d x '(7 9) #:align-corners? align?))
    (weighed out device)
    (values out (list x)))

  (define (python-bool b)
    (if b "True" "False"))

  (define (cases device)
    (append
     (for/list ([align? (in-list '(#f #t))])
       (cons (format "affine-~a" (python-bool align?))
             (lambda () (affine device align?))))
     (for*/list ([mode (in-list '(bilinear nearest bicubic))]
                 [padding (in-list '(zeros border reflection))]
                 [align? (in-list '(#f #t))])
       (cons (format "sample-~a-~a-~a" mode padding (python-bool align?))
             (lambda () (sampled device mode padding align?))))
     (for/list ([align? (in-list '(#f #t))])
       (cons (format "bicubic-~a" (python-bool align?))
             (lambda () (resized device align?))))))

  (define (check-values label t expected tolerance)
    (define got (for/list ([v (in-flattened-tensor (to-device t 'cpu))]) v))
    (check-equal? (length got) (length expected) label)
    (for ([a (in-list got)] [b (in-list expected)] [i (in-naturals)])
      (check-= a b tolerance (format "~a ~a" label i))))

  (define (check-twin device tolerance)
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
       (lambda () (python-check "grid_sample.py"))))
    (with-default-device 'cpu
      (manual-seed! 0)
      (for ([named (in-list (cases device))])
        (match-define (cons key run) named)
        (define expected (hash-ref j (string->symbol key)))
        (define label (format "~a [~a]" key device))
        (define-values (out leaves) (run))
        (check-values (format "~a: output" label) out (hash-ref expected 'out)
                      tolerance)
        (for ([l (in-list leaves)]
              [g (in-list (hash-ref expected 'grads))]
              [i (in-naturals)])
          (check-values (format "~a: gradient ~a" label i) (grad l) g
                        tolerance)))))

  (cond
    [(not (python-torch-available?))
     (printf "[grid-sample-parity-test] skipped: python3 `torch` ~a\n"
             "not available (run inside `nix develop`)")]
    [else
     (check-twin 'cpu 1e-5)
     (when (and (cuda-available?) (python-cuda-available?))
       (check-twin 'cuda 1e-4))]))
