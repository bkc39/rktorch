#lang racket/base

(module+ test
  (require (only-in racket/list make-list)
           (only-in racket/match == match-lambda)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in (submod "../nn/recurrent.rkt" private)
                    flatten-refused? flattened-signature packing-hook
                    packing-record storage-signature))

  (define device (if (cuda-available?) 'cuda 'cpu))

  (define (packed? signature)
    (apply = (map cdr signature)))

  (define (forward m x)
    (define-values (out _h _c) (m x))
    out)

  (define placement-race-failure
    (regexp (string-append "get_mkldnn_dtype: unsupported data type"
                           "|Input and parameter tensors are not at the same device"
                           "|Input and parameter tensors are not the same dtype"
                           "|(mat1|self) and mat2 must have the same dtype")))

  (define (expected-race-failure? e)
    (and (exn:fail? e)
         (regexp-match? placement-race-failure (exn-message e))))

  (define (racing-layer)
    (to (LSTM 8 16 #:num-layers 2 #:bidirectional? #t) device))

  (define (update! b f)
    (let retry ()
      (define seen (unbox b))
      (unless (box-cas! b seen (f seen))
        (retry))))

  (define (bump! counter)
    (update! counter add1))

  (define (count-decisions thunk)
    (define stale (box 0))
    (define packings (box 0))
    (parameterize ([packing-hook
                    (match-lambda
                      ['stale (bump! stale) (sleep 0.01)]
                      ['before (bump! packings)]
                      [_ (void)])])
      (thunk))
    (list (unbox stale) (unbox packings)))

  (define found '((100 . 100) (200 . 200)))
  (define buffer-start 1000)
  (define packed '((1000 . 1000) (1064 . 1000)))

  (test-case "weights that lie in the buffer record the packing"
    (check-equal? (packing-record found buffer-start packed #f) packed))

  (test-case "a weight moved off the buffer before recording forgets the packing"
    (check-false
     (packing-record found buffer-start '((1000 . 1000) (300 . 300)) #f)))

  (test-case "a packing during which the layer moved forgets it"
    (check-false (packing-record found buffer-start packed #t)))

  (test-case "a refusal on weights that did not move records them as they are"
    (check-equal? (packing-record found #f found #f) found))

  (test-case "a refusal on weights a move rebound forgets them"
    (check-false (packing-record found #f '((300 . 300) (400 . 400)) #f)))

  (test-case "a refusal on weights a round trip gave back their addresses forgets them"
    (check-false (packing-record found #f found #t)))

  (define (forward-together lstm x pool)
    (define failures (box '()))
    (define gate (make-semaphore 0))
    (define threads
      (for/list ([_ (in-range 4)])
        (thread (lambda ()
                  (semaphore-wait gate)
                  (with-handlers ([exn:fail?
                                   (lambda (e)
                                     (update! failures
                                              (lambda (seen)
                                                (cons (exn-message e) seen))))])
                    (forward lstm x)))
                #:pool pool)))
    (for ([_ (in-list threads)]) (semaphore-post gate))
    (for-each thread-wait threads)
    (unbox failures))

  (define (forwards-pack-once pool)
    (define lstm (racing-layer))
    (define weights (parameters lstm))
    (define x (randn 4 2 8 #:device device))
    (define failures '())
    (define counts
      (for/list ([_ (in-range 16)])
        (to lstm 'float64)
        (to lstm 'float32)
        (count-decisions
         (lambda ()
           (set! failures (append (forward-together lstm x pool) failures))))))
    (check-equal? failures '())
    (check-equal? counts (make-list 16 (list 1 (if (eq? device 'cuda) 1 0)))
                  "a forward found stale weights another had just checked")
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
    (define unexpected (box '()))
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
         (for ([_ (in-range 120)]
               [placement (in-cycle (in-list placements))])
           (call-with-semaphore turn
                                (lambda () (apply to lstm placement))))
         (set-box! done? #t))))
    (define forwarder
      (thread
       (lambda ()
         (let loop ()
           (with-handlers ([expected-race-failure? void]
                           [exn:fail?
                            (lambda (e)
                              (set-box! unexpected
                                        (cons (exn-message e)
                                              (unbox unexpected))))])
             (forward lstm x)
             (set-box! forwards (add1 (unbox forwards))))
           (call-with-semaphore turn check-claim!)
           (unless (unbox done?) (loop))))))
    (for-each thread-wait (list mover forwarder))
    (check-equal? (unbox unexpected) '())
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
    (define (move-at phase lstm placement)
      (match-lambda
        [(== phase) (apply to lstm placement)]
        [_ (void)]))

    (test-case "a move between packing and recording leaves nothing recorded"
      (define lstm (racing-layer))
      (define weights (parameters lstm))
      (define x (randn 4 2 8 #:device 'cuda))
      (parameterize ([packing-hook (move-at 'after lstm '(cuda float64))])
        (check-exn #rx"not the same dtype" (lambda () (forward lstm x))))
      (check-false (flattened-signature (car weights))
                   "unpacked weights were recorded as packed")
      (to lstm 'float32)
      (forward lstm x)
      (check-true (packed? (storage-signature weights))))

    (test-case "a refusal caused by a racing move is neither recorded nor latched"
      (define lstm (racing-layer))
      (define weights (parameters lstm))
      (define x (randn 4 2 8 #:device 'cuda))
      (parameterize ([packing-hook (move-at 'before lstm '(cpu float32))])
        (check-exn #rx"not at the same device" (lambda () (forward lstm x))))
      (check-false (flattened-signature (car weights))
                   "the moved weights' addresses were recorded as refused")
      (check-false (flatten-refused? (car weights))
                   "a refusal caused by the move was latched")
      (to lstm 'cuda)
      (forward lstm x)
      (check-true (packed? (storage-signature weights))))))
