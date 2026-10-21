#lang racket/base

(require (for-syntax racket/base)
         (only-in json read-json)
         (only-in racket/format ~a)
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
  (define exited? (sync/timeout time-limit child))
  (unless exited?
    (subprocess-kill child #t))
  (subprocess-wait child)
  (for ([t (in-list (list reader relay))])
    (sync/timeout 10 t))
  (close-input-port out)
  (close-input-port err)
  (cond
    [(not exited?) (raise-user-error 'pytorch "~a: no exit within ~a s" script time-limit)]
    [(zero? (subprocess-status child)) (reverse (unbox records))]
    [else (raise-user-error 'pytorch "~a exited with ~a" script (subprocess-status child))]))

(define (merged-meta parent child)
  (for/fold ([m parent]) ([(k v) (in-hash (or child (hasheq)))])
    (hash-set m k v)))

(define (twin-record child #:suite suite #:meta meta #:load [load (load-average)])
  (define samples (hash-ref child 'samples))
  (for/fold ([h (hasheq 'schema 1
                        'suite suite
                        'impl "pytorch"
                        'stats (summarize samples)
                        'load load
                        'load_end (load-average)
                        'meta (merged-meta meta (hash-ref child 'meta #f)))])
            ([(k v) (in-hash child)] #:unless (eq? k 'meta))
    (hash-set h k v)))
