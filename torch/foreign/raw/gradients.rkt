#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in racket/match match)
         (only-in "../device-type.rkt" device)
         (only-in "memory.rkt"
                  adopt-gradient!
                  free-handles!
                  record-allocation!
                  register-tensors!
                  registered)
         (only-in "pressure.rkt" collect-under-pressure!)
         (only-in "race-points.rkt" race-point))

(provide (struct-out fetched-gradient)
         gradients-allocator)

(struct fetched-gradient (handle nbytes type index))

(define registering/fetched
  (registered (lambda (rows) (register-tensors! (map fetched-gradient-handle rows)))))

;; Nothing returns the handles until every one is accounted, so a raise here
;; frees them all before it propagates.
(define (account-gradients! params rows)
  (define handles (map fetched-gradient-handle rows))
  (with-handlers ([(lambda (_) #t)
                   (lambda (e)
                     (free-handles! (filter values handles))
                     (raise e))])
    (define devs
      (for/list ([p (in-list params)] [row (in-list rows)])
        (match row
          [(fetched-gradient #f _ _ _) #f]
          [(fetched-gradient h nbytes type index)
           (define dev (device type (if (eq? type 'cpu) 0 index)))
           (record-allocation! h nbytes dev
                               #:adopt (lambda (entry) (adopt-gradient! (list p) h entry)))
           (race-point ledger-entry-added/batched h)
           dev])))
    (for ([dev (in-list (remove-duplicates (filter values devs)))])
      (collect-under-pressure! dev)))
  handles)

(define (gradients-allocator raw-fn)
  (define fetch (registering/fetched raw-fn))
  (lambda (params)
    (define rows (fetch params))
    (and rows (account-gradients! params rows))))
