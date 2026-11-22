#lang racket/base

(module+ test
  (require (only-in racket/file file->bytes)
           rackunit
           (only-in "../nn.rkt" Linear mse-loss parameters save-state! sgd step-lr)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" call-with-live-run saving-callback Small)
           (only-in "private/comparable.rkt" layer-bytes)
           (only-in "private/trainer-kit.rkt" toy-loader))

  (define (saved-with dir callbacks #:name [name "e.ckpt"] #:weights-only? [weights-only? #f])
    (define path (build-path dir name))
    (void (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f
               #:callbacks (append callbacks
                                   (list (saving-callback (lambda (_r) path)
                                                          #:weights-only? weights-only?)))))
    path)

  (define (keeper key state seen)
    (lambda-callback #:state-key key
                     #:state-dict (lambda (_r) state)
                     #:load-state! (lambda (_r st) (set-box! seen st))))

  (define (resume-capturing trainer path callbacks)
    (define receiver (make-log-receiver (current-logger) 'warning 'rktorch-train))
    (define err (open-output-string))
    (parameterize ([current-error-port err])
      (fit trainer (toy-loader 4) #:max-epochs 2 #:logger #f #:ckpt-path path #:callbacks callbacks))
    (values (let loop ()
              (define v (sync/timeout 0 receiver))
              (if v (cons (vector-ref v 1) (loop)) '()))
            (get-output-string err)))

  (test-case "T3-L17: a callback's state arrives equal? and immutable"
    (with-temporary-directory (dir #:template "protocol-~a")
      (define state (make-hash (list (cons "a" (list 1 2.5 'x)) (cons "b" (make-hasheq '((k . "v")))))))
      (define path (saved-with dir (list (keeper "keep" state (box #f)))))
      (define seen (box #f))
      (void (resume-capturing (Small) path (list (keeper "keep" state seen))))
      (check-equal? (unbox seen) (hash "a" (list 1 2.5 'x) "b" (hasheq 'k "v")))
      (check-true (immutable? (unbox seen)))
      (check-true (immutable? (hash-ref (unbox seen) "b")))))

  (test-case "T3-L17: an orphan entry warns, a callback without an entry loads nothing"
    (with-temporary-directory (dir #:template "orphan-~a")
      (define path (saved-with dir (list (keeper "keep" (hash) (box #f))
                                         (keeper "gone" (hash "n" 1) (box #f)))))
      (define fresh (box 'untouched))
      (define-values (warnings err)
        (resume-capturing (Small) path (list (keeper "keep" (hash) (box #f))
                                             (keeper "fresh" (hash) fresh))))
      (check-equal? (unbox fresh) 'untouched)
      (check-true (for/or ([w (in-list warnings)]) (regexp-match? #rx"\"gone\"|gone" w)) "the log")
      (check-regexp-match #rx"gone" err)
      (check-false (for/or ([w (in-list warnings)]) (regexp-match? #rx"fresh" w)))))

  (test-case "T3-L17: no state-dict hook, no entry; a state-dict that is no value raises"
    (with-temporary-directory (dir #:template "entries-~a")
      (define path (saved-with dir (list (lambda-callback #:state-key "silent"))))
      (check-false (hash-has-key? (hash-ref (read-checkpoint path) "callbacks") "silent"))
      (check-exn (lambda (e) (and (exn:fail:contract? e) (regexp-match? #rx"bad" (exn-message e))))
                 (lambda ()
                   (call-with-live-run
                    #:callbacks (list (lambda-callback #:state-key "bad"
                                                       #:state-dict (lambda (_r) (lambda () 1))))
                    (lambda (r) (save-checkpoint! r (build-path dir "bad.ckpt"))))))
      (check-false (file-exists? (build-path dir "bad.ckpt")))))

  (define-trainer Extra (net head)
    (define-init () (set! net (Linear 2 1)) (set! head (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Missing (other)
    (define-init () (set! other (Linear 2 1)))
    (define-optimizers (sgd (parameters other) #:lr 0.1))
    (define-training-step (x y) (mse-loss (other x) y)))

  (define-trainer Wide (net)
    (define-init () (set! net (Linear 2 3)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "T3-L18: a trainer of another shape raises after on-exception and teardown"
    (with-temporary-directory (dir #:template "shape-~a")
      (define path (saved-with dir '()))
      (for ([make (list Extra Missing Wide)] [rx (list #rx"head" #rx"net|other" #rx"shape|size")])
        (define tr (make))
        (define before (layer-bytes tr))
        (define seen (box '()))
        (define (note! name) (lambda (_r . _) (set-box! seen (cons name (unbox seen)))))
        (check-exn rx (lambda ()
                        (fit tr (toy-loader 4) #:max-epochs 2 #:logger #f #:ckpt-path path
                             #:callbacks (list (lambda-callback #:on-exception (note! 'on-exception)
                                                                #:teardown (note! 'teardown))))))
        (check-equal? (reverse (unbox seen)) '(on-exception teardown) (format "~a" rx))
        (check-equal? (layer-bytes tr) before (format "~a: parameters" rx)))))

  (define-trainer Pair (net)
    #:automatic-optimization? #f
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (values (sgd (parameters net) #:lr 0.1) (sgd (parameters net) #:lr 0.1)))
    (define-training-step (x y)
      (for ([opt (in-list (run-optimizers run))])
        (manual-zero-grads! run opt)
        (manual-backward! run (mse-loss (net x) y))
        (manual-step! run opt))
      #f))

  (define-trainer Scheduled (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers
      (define opt (sgd (parameters net) #:lr 0.1))
      (values opt (step-lr opt #:step-size 1)))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "T3-L19: optimizer and scheduler counts, and a weights-only file"
    (with-temporary-directory (dir #:template "counts-~a")
      (define pair-path (build-path dir "pair.ckpt"))
      (void (fit (Pair) (toy-loader 4) #:max-epochs 1 #:logger #f
                 #:callbacks (list (saving-callback (lambda (_r) pair-path)))))
      (check-exn #rx"optimizer.*2.*1|optimizer.*1.*2"
                 (lambda () (fit (Small) (toy-loader 4) #:max-epochs 2 #:logger #f
                                 #:ckpt-path pair-path)))
      (define path (saved-with dir '()))
      (check-exn #rx"scheduler.*0.*1|scheduler.*1.*0"
                 (lambda () (fit (Scheduled) (toy-loader 4) #:max-epochs 2 #:logger #f
                                 #:ckpt-path path)))
      (define weights (saved-with dir '() #:name "weights.ckpt" #:weights-only? #t))
      (check-exn #rx"contains only the model"
                 (lambda () (fit (Small) (toy-loader 4) #:max-epochs 2 #:logger #f
                                 #:ckpt-path weights)))))

  (test-case "T3-L20: fit with a file that is not a version-1 checkpoint runs no hook"
    (with-temporary-directory (dir #:template "refused-fit-~a")
      (define state-file (build-path dir "state.safetensors"))
      (save-state! (Linear 2 1) state-file)
      (define good (saved-with dir '()))
      (define v2 (build-path dir "v2.ckpt"))
      (call-with-output-file v2
        (lambda (o) (write-bytes (regexp-replace #rx#"(\"version\" *: *)\"1\"" (file->bytes good)
                                                 #"\\1\"2\"")
                                 o)))
      (for ([path (list state-file v2 (build-path dir "absent.ckpt"))])
        (define seen (box '()))
        (check-exn exn:fail?
                   (lambda ()
                     (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f #:ckpt-path path
                          #:callbacks (list (lambda-callback
                                             #:setup (lambda (_r _stage)
                                                       (set-box! seen '(setup))))))))
        (check-equal? (unbox seen) '() (format "~a" path))))))
