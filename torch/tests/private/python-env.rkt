#lang racket/base

;; Shared python-subprocess infra for the PyTorch parity suites.

;; whole-module require: define-runtime-path needs phase-1 bindings only-in
;; would strip
(require racket/runtime-path
         (only-in json read-json)
         (only-in racket/port open-output-nowhere)
         (only-in racket/system system*))

(provide python
         call-with-python-env
         with-python-env
         python-module-available?
         python-torch-available?
         python-cuda-available?
         python-result
         python-check
         tol)

(define-runtime-path examples-dir "../../../examples")
(define-runtime-path python-checks-dir "../python")

(define python (find-executable-path "python3"))

;; Under the .#cuda shell the process LD_LIBRARY_PATH (Racket's libtorch 2.9)
;; would shadow the Python wheel's own libtorch and break `import torch`;
;; pin each python child to the host-driver farm the cudaHook exports.
(define cuda-driver-path (getenv "RKTORCH_CUDA_DRIVER_PATH"))

;; Adjusts a *copy* of the environment, never the process-wide one.
(define (call-with-python-env thunk #:env [extra '()])
  (define env (environment-variables-copy (current-environment-variables)))
  (when cuda-driver-path
    (environment-variables-set! env #"LD_LIBRARY_PATH"
                                (string->bytes/utf-8 cuda-driver-path)))
  (for ([nv (in-list extra)])
    (environment-variables-set! env (string->bytes/utf-8 (car nv))
                                (string->bytes/utf-8 (cdr nv))))
  (parameterize ([current-environment-variables env]) (thunk)))

(define-syntax-rule (with-python-env body ...)
  (call-with-python-env (lambda () body ...)))

(define (python-module-available? mod)
  (and python
       (with-python-env
        (parameterize ([current-output-port (open-output-nowhere)]
                       [current-error-port (open-output-nowhere)])
          (system* python "-c" (format "import ~a" mod))))))

(define (python-torch-available?)
  (and python
       (with-python-env
        (parameterize ([current-output-port (open-output-nowhere)]
                       [current-error-port (open-output-nowhere)])
          (system* python "-c" "import torch")))))

(define (python-cuda-available?)
  (and python
       (with-python-env
        (parameterize ([current-output-port (open-output-nowhere)]
                       [current-error-port (open-output-nowhere)])
          (system* python "-c"
                   "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)")))))

;; Runs a Python reference file relative to examples/.
(define (python-result rel-path)
  (define py (build-path examples-dir rel-path))
  (define out (open-output-string))
  (define ok?
    (with-python-env
     (parameterize ([current-output-port out]
                    [current-error-port (open-output-nowhere)])
       (system* python (path->string py)))))
  (unless ok?
    (error 'python-cross-test "python failed for ~a" rel-path))
  (read-json (open-input-string (get-output-string out))))

;; Runs a standalone reference program in torch/tests/python/.
(define (python-check name)
  (define py (build-path python-checks-dir name))
  (define out (open-output-string))
  (define err (open-output-string))
  (define ok?
    (with-python-env
     (parameterize ([current-output-port out]
                    [current-error-port err])
       (system* python (path->string py)))))
  (unless ok?
    (error 'python-cross-test "python check ~a failed:\n~a"
           name (get-output-string err)))
  (read-json (open-input-string (get-output-string out))))

;; Absorbs libtorch-bin vs Python-torch patch skew; flip the flake's
;; torchSource to "python" for bit-exact parity.
(define tol 1e-4)
