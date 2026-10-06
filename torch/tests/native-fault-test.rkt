#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in ffi/unsafe _int _intptr _pointer cast ptr-ref)
           (only-in racket/port port->string)
           (only-in racket/sandbox
                    get-output kill-evaluator sandbox-error-output
                    sandbox-eval-limits sandbox-memory-limit sandbox-output
                    sandbox-path-permissions sandbox-security-guard)
           rackunit
           (only-in scribble/example make-base-eval)
           syntax/parse/define
           (only-in "../foreign/raw/fault.rkt" native-fault? native-fault-policy))

  ;; A handle at address 8 faults on first use, the same way every time.
  (define prelude
    '((require torch
               (only-in ffi/unsafe cast _intptr)
               (only-in ffi/unsafe/alloc allocator)
               (only-in torch/foreign/raw/fault native-faulted)
               (only-in torch/foreign/raw/syntax _Tensor)
               (only-in torch/foreign/structs tensor-impl)
               (only-in torch/foreign/raw/tensor tr-tensor-dtype/raw)
               (only-in torch/foreign/raw/memory
                        tr-tensor-free/finalizer finalizer-failures))
      (define (bad) (cast 8 _intptr _Tensor))
      (define (then-probe)
        (with-handlers ([exn:fail? (lambda (e)
                                     (printf "then: ~a\n" (exn-message e)))])
          (void (ones 2 2))
          (printf "then: usable\n")))))

  (define finalizer-fault
    '((define adopt ((allocator tr-tensor-free/finalizer) values))
      (for ([_ (in-range 5)]) (void (adopt (bad))))
      (for ([_ (in-range 3)]) (collect-garbage))
      (sleep 1)
      (printf "survived failures=~a\n" (finalizer-failures))
      (then-probe)))

  (define printer-fault
    '((printf "~a\n" (tensor-impl (bad) '(2 2)))
      (printf "survived\n")
      (then-probe)))

  (define call-fault
    '((with-handlers ([exn:fail? (lambda (_) (printf "caught\n"))])
        (tr-tensor-dtype/raw (bad)))
      (printf "survived\n")
      (then-probe)))

  (define (environment-with policy)
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"RKTORCH_ON_NATIVE_FAULT"
                                (and policy (string->bytes/utf-8 policy)))
    env)

  ;; Under the raise policy a fresh evaluator has its own instance of the
  ;; latch, so each case starts with the library enabled.
  (define (make-fault-eval)
    (parameterize ([current-environment-variables (environment-with #f)]
                   [sandbox-output 'string]
                   [sandbox-error-output 'string]
                   [sandbox-memory-limit #f]
                   [sandbox-eval-limits #f]
                   [sandbox-security-guard current-security-guard]
                   [sandbox-path-permissions '((exists "/"))])
      (apply make-base-eval prelude)))

  (define-syntax-parse-rule (with-fault-eval (ev:id) body:expr ...+)
    (let ([ev (make-fault-eval)])
      (dynamic-wind void (lambda () body ...) (lambda () (kill-evaluator ev)))))

  ;; What the program printed, and what disabled the library, if anything.
  (define (run-in-sandbox program)
    (with-fault-eval (ev)
      (for ([form (in-list program)]) (ev form))
      (values (get-output ev) (ev '(native-faulted)))))

  ;; The exit policy ends the process, and its notice goes to file
  ;; descriptor 2 past every port, so those cases run in a child racket.
  (define (run-racket program #:policy policy)
    (parameterize ([current-environment-variables (environment-with policy)])
      (define-values (sp out in err)
        (subprocess #f #f #f (find-system-path 'exec-file)
                    "-e" (format "~s" `(begin ,@prelude ,@program))))
      (close-output-port in)
      (define stdout (box ""))
      (define stderr (box ""))
      (define out-t (thread (lambda () (set-box! stdout (port->string out)))))
      (define err-t (thread (lambda () (set-box! stderr (port->string err)))))
      (define exited?
        (and (sync/timeout 120 (thread (lambda () (subprocess-wait sp)))) #t))
      (unless exited? (subprocess-kill sp #t))
      (sync/timeout 10 out-t)
      (sync/timeout 10 err-t)
      (close-input-port out)
      (close-input-port err)
      (unless exited?
        (error 'run-racket "child did not exit within 120s"))
      (values (subprocess-status sp) (unbox stdout) (unbox stderr))))

  (define disabled #rx"then: rktorch: native code faulted earlier, while ")

  (test-case "a segfault in native code is recognised as a native fault"
    (define e
      (with-handlers ([(lambda (_) #t) values])
        (ptr-ref (cast 8 _intptr _pointer) _int)))
    (check-true (native-fault? e))
    (check-false (native-fault? (exn:fail "boom" (current-continuation-marks))))
    (check-false (native-fault? 'invalid-memory-reference)))

  (test-case "the default policy raises"
    (check-equal? native-fault-policy 'raise))

  (test-case "a finalizer fault disables the library and is survived"
    (define-values (stdout faulted) (run-in-sandbox finalizer-fault))
    (check-true (regexp-match? #rx"survived failures=5" stdout) stdout)
    (check-true (regexp-match? disabled stdout) stdout)
    (check-equal? faulted "running a finalizer"))

  (test-case "under the exit policy a finalizer fault stops the process"
    (define-values (code stdout stderr) (run-racket finalizer-fault #:policy "exit"))
    (check-equal? code 70)
    (check-false (regexp-match? #rx"survived" stdout))
    (check-equal? (length (regexp-match* #rx"faulted .* while running a finalizer"
                                         stderr))
                  1
                  stderr)
    (check-true (< (string-length stderr) 4096) "one message, not a cascade"))

  (test-case "a fault while printing falls back to the shape and disables"
    (define-values (stdout faulted) (run-in-sandbox printer-fault))
    (check-true (regexp-match? #rx"#<tensor:2x2>\nsurvived" stdout) stdout)
    (check-true (regexp-match? disabled stdout) stdout)
    (check-equal? faulted "printing a tensor"))

  (test-case "under the exit policy a fault while printing stops the process"
    (define-values (code stdout stderr) (run-racket printer-fault #:policy "exit"))
    (check-equal? code 70)
    (check-false (regexp-match? #rx"survived" stdout))
    (check-true (regexp-match? #rx"while printing a tensor" stderr) stderr))

  (test-case "a fault in a caller's own call raises to it and disables nothing"
    (define-values (stdout faulted) (run-in-sandbox call-fault))
    (check-true (regexp-match? #rx"caught\nsurvived\nthen: usable" stdout) stdout)
    (check-false faulted)))
