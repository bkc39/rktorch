#lang racket/base

;; The CUDA cases are where flattening happens; on the CPU the same loops run
;; the bookkeeping with nothing to pack.

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in (submod "../nn/recurrent.rkt" private)
                    after-packing flatten-count flattened-signature
                    storage-signature))

  (define device (if (cuda-available?) 'cuda 'cpu))

  (define (packed? signature)
    (apply = (map cdr signature)))

  (define (forward m x)
    (call-with-values (lambda () (m x)) list))

  (define (spawn pool thunk)
    (thread thunk #:pool pool))

  (define (join-all threads)
    (for-each thread-wait threads))

  (define (racing-layer)
    (to (LSTM 8 16 #:num-layers 3 #:bidirectional? #t) device))

  (define (forwards-pack-once pool)
    (define lstm (racing-layer))
    (define weights (parameters lstm))
    (define x (randn 4 2 8 #:device device))
    (define failures (box '()))
    (define extra
      (for/sum ([_ (in-range 40)])
        (to lstm 'float64)
        (to lstm 'float32)
        (define before (flatten-count))
        (define gate (make-semaphore 0))
        (define threads
          (for/list ([_ (in-range 4)])
            (spawn pool
                   (lambda ()
                     (semaphore-wait gate)
                     (with-handlers ([exn:fail?
                                      (lambda (e)
                                        (set-box! failures
                                                  (cons (exn-message e)
                                                        (unbox failures))))])
                       (forward lstm x))))))
        (for ([_ (in-list threads)]) (semaphore-post gate))
        (join-all threads)
        (- (flatten-count) before (if (eq? device 'cuda) 1 0))))
    (check-equal? (unbox failures) '())
    (check-equal? extra 0 "a forward packed weights another had just packed")
    (check-equal? (flattened-signature (car weights))
                  (storage-signature weights)))

  (test-case "forwards racing on moved weights pack them once"
    (forwards-pack-once #f))

  (test-case "parallel forwards racing on moved weights pack them once"
    (forwards-pack-once 'own))

  (test-case "a move racing forwards never leaves a stale packing recorded"
    (define lstm (racing-layer))
    (define weights (parameters lstm))
    (define x (randn 4 2 8 #:device device))
    (define placements
      (if (eq? device 'cuda)
          '((cuda float32) (cuda float64) (cpu float32))
          '((cpu float32) (cpu float64))))
    (define turn (make-semaphore 1))
    (define stale-claims (box 0))
    (define forwards (box 0))
    (define done? (box #f))
    (define (check-claim!)
      (define now (storage-signature weights))
      (define claimed? (equal? now (flattened-signature (car weights))))
      (define on-cuda? (eq? (device-type (tensor-device (car weights))) 'cuda))
      (when (and claimed? on-cuda? (not (packed? now)))
        (set-box! stale-claims (add1 (unbox stale-claims)))))
    (define mover
      (thread
       (lambda ()
         (for ([_ (in-range 300)]
               [placement (in-cycle (in-list placements))])
           (call-with-semaphore turn
                                (lambda () (apply to lstm placement))))
         (set-box! done? #t))))
    (define forwarder
      (thread
       (lambda ()
         (let loop ()
           (with-handlers ([exn:fail? void])
             (forward lstm x)
             (set-box! forwards (add1 (unbox forwards))))
           (call-with-semaphore turn check-claim!)
           (unless (unbox done?) (loop))))))
    (join-all (list mover forwarder))
    (check-equal? (unbox stale-claims) 0
                  "a packing was recorded for weights a move had unpacked")
    (check-true (positive? (unbox forwards)) "no forward ran to completion")
    (apply to lstm (car placements))
    (forward lstm x)
    (check-equal? (flattened-signature (car weights))
                  (storage-signature weights))
    (when (eq? device 'cuda)
      (check-true (packed? (storage-signature weights)))))

  (when (cuda-available?)
    (test-case "a move between packing and recording leaves nothing recorded"
      (define lstm (racing-layer))
      (define weights (parameters lstm))
      (define x (randn 4 2 8 #:device 'cuda))
      (parameterize ([after-packing (lambda () (to lstm 'float64))])
        (check-exn exn:fail? (lambda () (forward lstm x))))
      (check-false (equal? (flattened-signature (car weights))
                           (storage-signature weights))
                   "unpacked weights were recorded as packed")
      (to lstm 'float32)
      (forward lstm x)
      (check-true (packed? (storage-signature weights))))))
