#lang racket/base

(module+ test
  (require (only-in racket/file file->bytes)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "../foreign/raw/fault.rkt"
                    native-fault-policy note-native-fault! reset-native-fault-latch!)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd state-dict)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/comparable.rkt" comparable host-bytes)
           (only-in "private/trainer-kit.rkt" toy-loader))

  (define-trainer Failing (net at-raise how)
    (define-init (h [b (box #f)]) (set! net (Linear 2 1)) (set! at-raise b) (set! how h))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y)
      (when (and how (= (run-epoch run) 1) (= (run-batch-index run) 1))
        (set-box! at-raise (for/hash ([e (in-list (state-dict net))])
                             (values (string-append "net." (car e)) (host-bytes (cdr e)))))
        (how))
      (mse-loss (net x) y)))

  (define (fit-failing tr dir)
    (fit tr (toy-loader 6) #:max-epochs 3 #:logger #f
         #:callbacks (list (on-exception-checkpoint #:dirpath dir))))

  (define (file-in dir) (build-path dir "on_exception.ckpt"))

  (test-case "T3-L28 (a): a failure leaves the parameters at the raise, and re-raises"
    (with-temporary-directory (dir #:template "on-exception-~a")
      (define at-raise (box #f))
      (define tr (Failing (lambda () (error 'training-step "planned failure")) at-raise))
      (check-exn #rx"planned failure" (lambda () (fit-failing tr dir)))
      (check-equal? (comparable (hash-ref (read-checkpoint (file-in dir)) "state_dict"))
                    (unbox at-raise))))

  (test-case "T3-L28 (b): a fit that finishes removes a stale file"
    (with-temporary-directory (dir #:template "stale-~a")
      (call-with-output-file (file-in dir) (lambda (o) (write-bytes #"stale" o)))
      (void (fit-failing (Failing #f) dir))
      (check-false (file-exists? (file-in dir)))))

  (test-case "T3-L28 (c): after a native fault nothing is written and the original raise survives"
    (unless (eq? native-fault-policy 'raise)
      (displayln "skipped: RKTORCH_ON_NATIVE_FAULT=exit"))
    (when (eq? native-fault-policy 'raise)
      (with-temporary-directory (dir #:template "latched-~a")
        (call-with-output-file (file-in dir) (lambda (o) (write-bytes #"earlier" o)))
        (define receiver (make-log-receiver (current-logger) 'warning 'rktorch-train))
        (define tr (Failing (lambda ()
                              (note-native-fault! "a planned fault")
                              (error 'training-step "original failure"))))
        (dynamic-wind
         void
         (lambda ()
           (parameterize ([current-error-port (open-output-nowhere)])
             (check-exn #rx"original failure" (lambda () (fit-failing tr dir)))))
         reset-native-fault-latch!)
        (check-equal? (file->bytes (file-in dir)) #"earlier")
        (check-true (let loop ()
                      (define v (sync/timeout 0 receiver))
                      (and v (or (regexp-match? #rx"disabled|native" (vector-ref v 1)) (loop))))))))

  (test-case "T3-L28 (d): a break gives a file"
    (with-temporary-directory (dir #:template "break-~a")
      (define tr (Failing (lambda () (break-thread (current-thread)) (sleep 0))))
      (check-true (with-handlers ([exn:break? (lambda (_) #t)]) (fit-failing tr dir) #f))
      (check-true (file-exists? (file-in dir))))))
