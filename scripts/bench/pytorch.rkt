#lang racket/base

(require (for-syntax racket/base)
         (only-in json read-json)
         (only-in racket/format ~a)
         (only-in racket/hash hash-union)
         (only-in racket/match match)
         (only-in racket/port copy-port)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in torch/tests/private/python-env
                  call-with-python-env python python-torch-available?)
         (only-in "harness.rkt" load-average summarize))

(provide python-dir
         twin-script
         pytorch-available?
         require-pytorch
         python-child-environment
         twin-records
         twin-record
         merged-meta)

(define-runtime-path python-dir "python")

(define (twin-script name)
  (build-path python-dir name))

(define (pytorch-available?)
  (and python (python-torch-available?) #t))

(define (require-pytorch who)
  (unless (pytorch-available?)
    (raise-user-error who "needs python3 with torch (run inside `nix develop`)")))

(define (python-child-environment)
  (call-with-python-env
   (lambda () (environment-variables-copy (current-environment-variables)))))

(define (twin-records script args #:time-limit [time-limit 3600] #:log [log (current-error-port)])
  (define-values (child out in err)
    (parameterize ([current-environment-variables (python-child-environment)])
      (apply subprocess #f #f #f python (path->string (twin-script script)) (map ~a args))))
  (close-output-port in)
  (define relay (thread (lambda () (copy-port err log))))
  (define records (box '()))
  (define reader
    (thread (lambda ()
              (for ([r (in-port read-json out)])
                (set-box! records (cons r (unbox records)))))))
  (define status
    (cond
      [(sync/timeout time-limit child) (subprocess-status child)]
      [else (subprocess-kill child #t) (subprocess-wait child) 'no-exit]))
  (for ([t (in-list (list reader relay))])
    (sync/timeout 10 t))
  (close-input-port out)
  (close-input-port err)
  (match status
    ['no-exit (raise-user-error 'pytorch "~a: no exit within ~a s" script time-limit)]
    [0 (reverse (unbox records))]
    [code (raise-user-error 'pytorch "~a exited with ~a" script code)]))

(define (merged-meta parent child)
  (define twin (or child (hasheq)))
  (hash-union parent
              (if (hash-has-key? twin 'gpu)
                  (hash-set (hash-remove twin 'gpu) 'torch_gpu (hash-ref twin 'gpu))
                  twin)
              #:combine/key (lambda (_k _parent from-twin) from-twin)))

(define (twin-record child #:suite suite #:meta meta #:load [load (load-average)])
  (hash-union (hasheq 'schema 1
                      'suite suite
                      'impl "pytorch"
                      'stats (summarize (hash-ref child 'samples))
                      'load load
                      'load_end (load-average)
                      'meta (merged-meta meta (hash-ref child 'meta #f)))
              (hash-remove child 'meta)
              #:combine/key (lambda (_k _record from-child) from-child)))

(module+ test
  (require rackunit)

  (test-case "a twin's meta keeps the parent's GPU, and names torch's beside it"
    (define parent (hasheq 'host "lab" 'device "cuda" 'gpu #f 'git_sha "abc"))
    (define child (hasheq 'impl "pytorch" 'device "cuda" 'gpu "NVIDIA GeForce RTX 3090 Ti"))
    (check-equal? (merged-meta parent child)
                  (hasheq 'host "lab" 'device "cuda" 'gpu #f 'git_sha "abc" 'impl "pytorch"
                          'torch_gpu "NVIDIA GeForce RTX 3090 Ti"))
    (check-equal? (merged-meta parent #f) parent)
    (define r (twin-record (hasheq 'case "add" 'samples '(1.0 3.0) 'meta child)
                           #:suite "micro" #:meta parent #:load '(0.5)))
    (check-equal? (hash-ref r 'case) "add")
    (check-equal? (hash-ref (hash-ref r 'stats) 'median) 2.0)
    (check-false (hash-ref (hash-ref r 'meta) 'gpu))
    (check-equal? (hash-ref r 'load) '(0.5))))
