#lang racket/base

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
         (only-in ffi/unsafe -> _fun _int get-ffi-obj)
         (only-in racket/format ~a)
         (only-in racket/port copy-port)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in threading ~>)
         (only-in "pytorch.rkt" python-child-environment))

(provide python-runner?
         child-command
         run-child)

(define-runtime-path recorder "training.rkt")

(define (python-runner? runner)
  (regexp-match? #rx"[.]py$" (path->string runner)))

(define (child-command runner out)
  (cond
    [(python-runner? runner)
     (list (find-executable-path "python3") (path->string runner) (path->string out))]
    [else
     (define form
       `(begin
          (require (only-in (file ,(path->string recorder)) record-steps!))
          (record-steps! ,(path->string runner) ,(path->string out))))
     (list (find-exe) "-l" "racket/base" "-e" (format "~s" form))]))

(define (child-environment runner cleared settings)
  (define env (if (python-runner? runner)
                  (python-child-environment)
                  (environment-variables-copy (current-environment-variables))))
  (for ([name (in-list cleared)])
    (environment-variables-set! env (string->bytes/utf-8 name) #f))
  (for ([s (in-list settings)])
    (environment-variables-set! env (string->bytes/utf-8 (car s))
                                (~> s cdr ~a string->bytes/utf-8)))
  env)

(define (call-with-child-log dir log proc)
  (cond
    [(file-stream-port? log) (proc log)]
    [else
     (define path (build-path dir "child.log"))
     (define child-log (open-output-file path))
     (dynamic-wind
      void
      (lambda () (proc child-log))
      (lambda ()
        (close-output-port child-log)
        (call-with-input-file path (lambda (in) (copy-port in log)))))]))

(define kill (get-ffi-obj "kill" #f (_fun _int _int -> _int)))

(define sigkill 9)

(define (kill-process-group! child)
  (~> (subprocess-pid child) - (kill sigkill) void))

(define (run-child runner dir out cleared settings log time-limit)
  (call-with-child-log
   dir log
   (lambda (child-log)
     (parameterize-break #f
       (define-values (child _stdout stdin _stderr)
         (parameterize ([current-environment-variables (child-environment runner cleared settings)]
                        [current-directory dir]
                        [subprocess-group-enabled #t])
           (apply subprocess child-log #f 'stdout (child-command runner out))))
       (close-output-port stdin)
       (define exited?
         (with-handlers ([exn:break? (lambda (e) (kill-process-group! child) (raise e))])
           (sync/timeout/enable-break time-limit child)))
       (kill-process-group! child)
       (subprocess-wait child)
       (list (subprocess-status child) (not exited?))))))
