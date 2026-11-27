#lang racket/base

(require (only-in ffi/unsafe _fun _int _long _pointer _uint64 _ptr define-cstruct get-ffi-obj)
         (only-in racket/match match match*)
         (only-in "platform.rkt" host-platform))

(provide clock-gettime
         clock-id
         make-timespec
         os-thread-id
         set-timespec-nsec!
         set-timespec-sec!
         timespec->milliseconds
         timespec-nsec
         timespec-sec)

(define-cstruct _timespec ([sec _long] [nsec _long]) #:malloc-mode 'atomic-interior)

(define clock-gettime (get-ffi-obj "clock_gettime" #f (_fun _int _pointer -> _int)))

(define (clock-id name [platform host-platform])
  (match* (platform name)
    [('linux 'monotonic) 1]
    [('linux 'thread-cputime) 3]
    [('macosx 'monotonic) 6]
    [('macosx 'thread-cputime) 16]
    [(_ _) #f]))

(define (timespec->milliseconds ts)
  (+ (* 1000.0 (timespec-sec ts)) (/ (timespec-nsec ts) 1e6)))

(define os-thread-id
  (match host-platform
    ['linux (get-ffi-obj "gettid" #f (_fun -> _int))]
    ['macosx
     (define threadid
       (get-ffi-obj "pthread_threadid_np" #f
                    (_fun _pointer (id : (_ptr o _uint64)) -> (rc : _int) -> (and (zero? rc) id))))
     (lambda () (threadid #f))]
    [_ (lambda () #f)]))

(module+ test
  (require rackunit)

  (test-case "each OS reads its own clock ids, and an unknown OS has none"
    (check-equal? (map (lambda (os) (clock-id 'thread-cputime os)) '(linux macosx freebsd))
                  '(3 16 #f))
    (check-equal? (map (lambda (os) (clock-id 'monotonic os)) '(linux macosx)) '(1 6)))

  (test-case "two live threads on their own OS threads read different ids, each the same twice"
    (define ready (make-semaphore 0))
    (define gate (make-semaphore 0))
    (define results (list (box #f) (box #f)))
    (define threads
      (for/list ([result (in-list results)])
        (thread (lambda ()
                  (define here (os-thread-id))
                  (semaphore-post ready)
                  (semaphore-wait gate)
                  (set-box! result (list here (os-thread-id))))
                #:pool 'own)))
    (for ([_ (in-list threads)]) (semaphore-wait ready))
    (for ([_ (in-list threads)]) (semaphore-post gate))
    (for-each thread-wait threads)
    (match (map unbox results)
      [(list (list a a*) (list b b*))
       (check-equal? a a*)
       (check-equal? b b*)
       (when (os-thread-id)
         (check-not-equal? a b)
         (check-not-equal? a (os-thread-id)))]))

  (test-case "the monotonic clock reads where the OS has one"
    (define ts (make-timespec 0 0))
    (when (clock-id 'monotonic)
      (check-equal? (clock-gettime (clock-id 'monotonic) ts) 0)
      (check-true (positive? (timespec->milliseconds ts))))))
