#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in racket/contract/base -> any/c)
         (only-in racket/match match match-define)
         (only-in "../foreign.rkt"
                  current-generator device device-type make-generator tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "dataset.rkt" collate/c dataset-batch dataset? indices/c))

;; Workers hold the queue side only: their mailbox, the results channel and
;; the stop semaphore. The pool, its reorder buffer and the traversal that
;; holds it stay unreachable from them, so the finalizer can stop them.
(struct worker-pool (threads results stop buffer))

(define (on-host? v)
  (or (not (tensor? v)) (eq? (device-type (device v)) 'cpu)))

(define (build ds collate r indices)
  (with-handlers ([(lambda (v) (not (exn:break? v)))
                   (lambda (v) (vector r #f v))])
    (define values-list
      (call-with-values (lambda () (dataset-batch ds indices collate)) list))
    (cond
      [(andmap on-host? values-list) (vector r #t values-list)]
      [else
       (raise (exn:fail:contract
               (format "dataloader: workers build CPU tensors, but batch ~a holds one on an accelerator"
                       r)
               (current-continuation-marks)))])))

(define (worker-prng seed)
  (define prng (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator prng])
    (random-seed (bitwise-and seed #x7FFFFFFF)))
  prng)

(define (serve ds collate results stop)
  (define stopped (semaphore-peek-evt stop))
  (let loop ([fifo '()])
    (cond
      [(sync/timeout 0 stopped) (void)]
      [(and (pair? fifo) (sync/timeout 0 (channel-put-evt results (car fifo))))
       (loop (cdr fifo))]
      [(thread-try-receive)
       => (lambda (request)
            (match-define (cons r indices) request)
            (loop (append fifo (list (build ds collate r indices)))))]
      [(null? fifo)
       (sync (handle-evt (thread-receive-evt) (lambda (_) (loop fifo)))
             stopped)]
      [else
       (sync (handle-evt (channel-put-evt results (car fifo))
                         (lambda (_) (loop (cdr fifo))))
             (handle-evt (thread-receive-evt) (lambda (_) (loop fifo)))
             stopped)])))

(define/contract-out (start-workers ds collate n base-seed) ;; noqa
  (-> dataset? collate/c exact-positive-integer? exact-nonnegative-integer?
      worker-pool?)
  (define results (make-channel))
  (define stop (make-semaphore 0))
  (define threads
    (for/vector #:length n ([k (in-range n)])
      (define seed (+ base-seed k))
      (thread (lambda ()
                (parameterize ([current-generator (make-generator seed)]
                               [current-pseudo-random-generator (worker-prng seed)])
                  (serve ds collate results stop)))
              #:pool 'own)))
  (define pool (worker-pool threads results stop (make-hasheqv)))
  (register-finalizer pool (lambda (_pool) (semaphore-post stop)))
  pool)

(define (worker-of pool r)
  (define threads (worker-pool-threads pool))
  (modulo r (vector-length threads)))

(define/contract-out (hand-off! pool r indices) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? indices/c void?)
  (void (thread-send (vector-ref (worker-pool-threads pool) (worker-of pool r))
                     (cons r indices)
                     #f)))

;; ok? #t: the payload is the batch's values as a list; #f: the raised value
(define/contract-out (await-batch pool r) ;; noqa
  (-> worker-pool? exact-nonnegative-integer? (values boolean? any/c))
  (define buffer (worker-pool-buffer pool))
  (define k (worker-of pool r))
  (define dead (thread-dead-evt (vector-ref (worker-pool-threads pool) k)))
  (let loop ()
    (match (hash-ref buffer r #f)
      [(vector _ ok? payload)
       (hash-remove! buffer r)
       (values ok? payload)]
      [#f
       (sync (handle-evt (worker-pool-results pool)
                         (lambda (entry)
                           (hash-set! buffer (vector-ref entry 0) entry)
                           (loop)))
             (handle-evt dead
                         (lambda (_)
                           (error 'dataloader
                                  "worker ~a exited before delivering batch ~a"
                                  k r))))])))

(define/contract-out (stop-workers! pool) (-> worker-pool? void?) ;; noqa
  (semaphore-post (worker-pool-stop pool)))
