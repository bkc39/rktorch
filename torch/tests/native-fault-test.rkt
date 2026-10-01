#lang racket/base

(module+ test
  (require (only-in ffi/unsafe _int _intptr _pointer cast ptr-ref)
           (only-in racket/port port->string)
           (only-in racket/string string-join)
           rackunit
           (only-in "../foreign/raw/fault.rkt" native-fault? native-fault-policy))

  ;; A handle at address 8 faults on first use, the same way every time.
  (define prelude
    (string-join
     '("(require torch"
       "  (only-in ffi/unsafe cast _intptr)"
       "  (only-in ffi/unsafe/alloc allocator)"
       "  (only-in torch/foreign/raw/syntax _Tensor)"
       "  (only-in torch/foreign/structs tensor-impl)"
       "  (only-in torch/foreign/raw/tensor tr-tensor-dtype/raw)"
       "  (only-in torch/foreign/raw/memory"
       "           tr-tensor-free/finalizer finalizer-failures))"
       "(define (bad) (cast 8 _intptr _Tensor))"
       "(define (then-probe)"
       "  (with-handlers ([exn:fail? (lambda (e)"
       "                               (printf \"then: ~a\\n\" (exn-message e)))])"
       "    (void (ones 2 2))"
       "    (printf \"then: usable\\n\")))")
     "\n"))

  (define finalizer-fault
    (string-append
     prelude
     "(define adopt ((allocator tr-tensor-free/finalizer) values))"
     "(for ([_ (in-range 5)]) (void (adopt (bad))))"
     "(for ([_ (in-range 3)]) (collect-garbage))"
     "(sleep 1)"
     "(printf \"survived failures=~a\\n\" (finalizer-failures))"
     "(then-probe)"))

  (define printer-fault
    (string-append
     prelude
     "(printf \"~a\\n\" (tensor-impl (bad) '(2 2)))"
     "(printf \"survived\\n\")"
     "(then-probe)"))

  (define call-fault
    (string-append
     prelude
     "(with-handlers ([exn:fail? (lambda (_) (printf \"caught\\n\"))])"
     "  (tr-tensor-dtype/raw (bad)))"
     "(printf \"survived\\n\")"
     "(then-probe)"))

  (define (run-racket program #:policy policy)
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (environment-variables-set! (current-environment-variables)
                                  #"RKTORCH_ON_NATIVE_FAULT"
                                  (and policy (string->bytes/utf-8 policy)))
      (define-values (sp out in err)
        (subprocess #f #f #f (find-system-path 'exec-file) "-e" program))
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

  (test-case "a finalizer fault disables the library, once, and is survived"
    (define-values (code stdout stderr) (run-racket finalizer-fault #:policy #f))
    (check-equal? code 0 stderr)
    (check-true (regexp-match? #rx"survived failures=5" stdout) stdout)
    (check-true (regexp-match? disabled stdout) stdout)
    (check-equal? (length (regexp-match* #rx"while running a finalizer" stderr))
                  1
                  stderr))

  (test-case "under the exit policy a finalizer fault stops the process"
    (define-values (code stdout stderr) (run-racket finalizer-fault #:policy "exit"))
    (check-equal? code 70)
    (check-false (regexp-match? #rx"survived" stdout))
    (check-true (regexp-match? #rx"faulted .* while running a finalizer" stderr)
                stderr)
    (check-true (< (string-length stderr) 4096) "one message, not a cascade"))

  (test-case "a fault while printing falls back to the shape and disables"
    (define-values (code stdout stderr) (run-racket printer-fault #:policy #f))
    (check-equal? code 0 stderr)
    (check-true (regexp-match? #rx"#<tensor:2x2>\nsurvived" stdout) stdout)
    (check-true (regexp-match? disabled stdout) stdout)
    (check-true (regexp-match? #rx"while printing a tensor" stderr) stderr))

  (test-case "under the exit policy a fault while printing stops the process"
    (define-values (code stdout stderr) (run-racket printer-fault #:policy "exit"))
    (check-equal? code 70)
    (check-false (regexp-match? #rx"survived" stdout))
    (check-true (regexp-match? #rx"while printing a tensor" stderr) stderr))

  (test-case "a fault in a caller's own call raises to it and disables nothing"
    (define-values (code stdout stderr) (run-racket call-fault #:policy #f))
    (check-equal? code 0 stderr)
    (check-true (regexp-match? #rx"caught\nsurvived\nthen: usable" stdout) stdout)))
