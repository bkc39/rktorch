#lang racket/base

(require (only-in racket/contract/base
                  -> ->* and/c any any/c flat-named-contract not/c)
         (only-in "../private/contract.rkt" define/contract-out))

(provide drain-deadline
         reap-when-dead!
         reaper-count)

(module+ private
  (provide heap-object/c
           reaper-statistics
           reaper-threads
           stop-reapers!))

(define heap-object/c
  (flat-named-contract 'heap-object/c
    (lambda (v) (not (or (fixnum? v) (char? v) (boolean? v) (null? v)
                         (void? v) (eof-object? v))))))

(struct reaped (name release guardian)
  #:constructor-name reaped/guardian
  #:omit-define-syntaxes)

(define drain-deadline (make-parameter 2000))

(define reaper-count 2)

(define/contract-out (reaped name release)
  (-> symbol? (-> any/c any) reaped?)
  (error 'unimplemented))

(define (reap-when-dead! kind v)
  (error 'unimplemented))

(define/contract-out (drain-reapers! #:deadline-ms [ms (drain-deadline)])
  (->* () (#:deadline-ms (and/c real? (not/c negative?))) boolean?)
  (error 'unimplemented))

(define/contract-out (wake-reapers!) (-> void?)
  (error 'unimplemented))

(define (reaper-threads)
  (error 'unimplemented))

(define (reaper-statistics)
  (error 'unimplemented))

(define (stop-reapers!)
  (error 'unimplemented))
