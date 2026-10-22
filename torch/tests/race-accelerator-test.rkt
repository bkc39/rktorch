#lang racket/base

(module+ test
  (require rackunit
           (only-in "../foreign.rkt" mps-device)
           (only-in "../foreign/accelerator.rkt" make-mps-probe pick-accelerator)
           (only-in "../foreign/raw/fault.rkt" native-fault?)
           (only-in "private/race-harness.rkt"
                    gate-hook hooks join join-all make-gate release-gate! spawn
                    stop! wait-for-arrival with-race-hook worker-thread))

  (define pools '(#f own))

  (define (counted outcome)
    (define calls 0)
    (values (λ () (set! calls (add1 calls)) (outcome))
            (λ () calls)))

  (define ((asking probe))
    (pick-accelerator #:cuda? #f #:mps? #t #:probe probe))

  (define (with-probe-gates probe pool points body)
    (define gates
      (for/list ([point (in-list points)])
        (make-gate point #:when (λ (subject) (eq? subject probe)))))
    (define started '())
    (define (start thunk)
      (define w (spawn thunk #:pool pool))
      (set! started (cons w started))
      w)
    (with-race-hook (apply hooks (map gate-hook gates))
      (dynamic-wind void
                    (λ () (apply body start gates))
                    (λ () (for-each stop! started)))))

  (define (parked? g)
    (eq? (wait-for-arrival g #:seconds 5) 'parked))

  (define (outcome-within seconds w)
    (with-handlers ([(λ (_) #t) values])
      (join w #:seconds seconds)))

  (test-case "a waiter arriving between the claim and the probe run shares that probe"
    (for ([pool (in-list pools)])
      (define-values (run calls) (counted void))
      (define probe (make-mps-probe run))
      (with-probe-gates probe pool '(mps-probe-claimed mps-probe-waiting)
        (λ (start claimed waiting)
          (define claimant (start (asking probe)))
          (check-true (parked? claimed))
          (define waiter (start (asking probe)))
          (check-true (parked? waiting) "the waiter did not wait on the published claim")
          (check-equal? (calls) 0)
          (release-gate! waiting)
          (release-gate! claimed)
          (check-equal? (join-all (list claimant waiter) #:seconds 5)
                        (list (mps-device) (mps-device)))
          (check-equal? (calls) 1 (format "~a" pool))))))

  (test-case "a claimant killed after storing the answer leaves its waiter that answer"
    (for ([pool (in-list pools)])
      (define-values (run calls) (counted void))
      (define probe (make-mps-probe run))
      (with-probe-gates probe pool '(mps-probe-claimed mps-probe-waiting mps-probe-settled)
        (λ (start claimed waiting settled)
          (define claimant (start (asking probe)))
          (check-true (parked? claimed))
          (define waiter (start (asking probe)))
          (check-true (parked? waiting))
          (release-gate! waiting)
          (release-gate! claimed)
          (check-true (parked? settled))
          (stop! claimant)
          (check-equal? (outcome-within 5 waiter) (mps-device) (format "~a" pool))
          (check-equal? (calls) 1 (format "~a" pool))))))

  (test-case "a break delivered at the claim boundary is deferred past the stored answer"
    (for ([pool (in-list pools)])
      (define-values (run calls) (counted void))
      (define probe (make-mps-probe run))
      (define broke (make-semaphore 0))
      (define (broken-then-lingering)
        (with-handlers ([exn:break? (λ (_) (semaphore-post broke) (sync never-evt))])
          (parameterize-break #t
            ((asking probe))
            (sync never-evt))))
      (with-probe-gates probe pool '(mps-probe-claimed)
        (λ (start claimed)
          (define claimant (start broken-then-lingering))
          (check-true (parked? claimed))
          (break-thread (worker-thread claimant))
          (release-gate! claimed)
          (check-not-false (sync/timeout 5 broke) "the break was lost")
          (check-equal? (outcome-within 5 (start (asking probe))) (mps-device)
                        (format "another thread waited on a claim its living holder abandoned (~a)"
                                pool))
          (check-equal? (calls) 1 (format "~a" pool))))))

  (define (fault)
    (raise (exn:fail "invalid memory reference.  Some debugging context lost"
                     (current-continuation-marks))))

  (test-case "a native fault in the probe reaches a parked waiter and later callers"
    (for ([pool (in-list pools)])
      (define-values (run calls) (counted fault))
      (define probe (make-mps-probe run))
      (parameterize ([current-logger (make-logger)])
        (with-probe-gates probe pool '(mps-probe-claimed mps-probe-waiting)
          (λ (start claimed waiting)
            (define claimant (start (asking probe)))
            (check-true (parked? claimed))
            (define waiter (start (asking probe)))
            (check-true (parked? waiting))
            (release-gate! waiting)
            (release-gate! claimed)
            (for ([w (list claimant waiter (start (asking probe)))])
              (check-pred native-fault? (outcome-within 5 w) (format "~a" pool)))
            (check-equal? (calls) 1 (format "~a" pool))))))))
