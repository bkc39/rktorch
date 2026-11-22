#lang racket/base

(require (only-in json read-json)
         (only-in racket/file file->string)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/system system*)
         (only-in "../../private/util.rkt" with-temporary-directory)
         (only-in "python-env.rkt" python python-torch-available? with-python-env)
         (only-in "safetensors-reader.rkt" read-safetensors))

(provide call-with-optim-oracle oracle-json oracle-tensors)

(define-runtime-path script "../python/optim_state.py")

(define (call-with-optim-oracle who proc)
  (cond
    [(not (file-exists? script))
     (printf "[~a] skipped: no optim_state.py (installed package)\n" who)]
    [(not (python-torch-available?))
     (printf "[~a] skipped: python3 `torch` not available\n" who)]
    [else
     (with-temporary-directory (dir #:template "optim-state-~a")
       (define log (open-output-string))
       (define ok?
         (parameterize ([current-output-port log] [current-error-port log])
           (with-python-env (system* python "-I" script (path->string dir)))))
       (unless ok? (error who "optim_state.py failed:\n~a" (get-output-string log)))
       (proc dir))]))

(define (oracle-json dir case)
  (call-with-input-file (build-path dir (string-append case ".json")) read-json))

(define (oracle-tensors dir case)
  (read-safetensors (build-path dir (string-append case ".safetensors"))))
