#lang racket/base

(module+ cases
  (require (for-syntax racket/base)
           (only-in racket/list drop-right flatten last)
           rackunit
           syntax/parse/define
           (only-in "../audio/data.rkt" load-audio save-audio)
           (only-in "../audio/functional.rkt" log-mel-spectrogram)
           (only-in "../audio/librispeech.rkt"
                    librispeech librispeech-fixture-utterance librispeech?
                    utterance utterance-path utterance-transcript)
           (only-in "../data/loader.rkt"
                    dataloader dataset-batch dataset-length dataset-ref
                    default-collate)
           (only-in "../main.rkt"
                    device-type dtype make-generator narrow native-memory-use
                    ref shape sum tensor tensor->list tensor? zeros)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "private/race-harness.rkt" settle!))

  (define blamed #rx"blaming: [(][^)]*librispeech-dataset-test[.]rkt")

  (define ((contract-violation? rx) e)
    (and (exn:fail:contract? e)
         (regexp-match? blamed (exn-message e))
         (regexp-match? rx (exn-message e))))

  (define frame-counts '(8000 21000 12000 16000 30000))

  (define (call-with-utterances proc)
    (define-values (samples rate)
      (load-audio (utterance-path (librispeech-fixture-utterance))))
    (with-temporary-directory (dir)
      (proc (for/list ([frames (in-list frame-counts)] [k (in-naturals)])
              (define path (build-path dir (format "9-9-~a.flac" k)))
              (save-audio path (narrow samples 1 0 frames) rate)
              (utterance (format "9-9-~a" k) path (format "UTTERANCE ~a" k))))))

  (define-syntax-parse-rule (with-utterances (us:id) body:expr ...+)
    (call-with-utterances (lambda (us) body ...)))

  (define (record v)
    (if (tensor? v) (list (shape v) (dtype v) (tensor->list v)) v))

  (define (records . vs)
    (map record vs))

  (define (item-of ds i)
    (call-with-values (lambda () (dataset-ref ds i)) list))

  (define (batch-of ds indices [collate default-collate])
    (call-with-values (lambda () (dataset-batch ds indices collate)) list))

  (define (median-allocation thunk)
    (call-with-values thunk void)
    (define runs
      (sort (for/list ([_ (in-range 15)])
              (define before (current-memory-use 'cumulative))
              (call-with-values thunk void)
              (- (current-memory-use 'cumulative) before))
            <))
    (list-ref runs 7))

  (define (cpu-bytes)
    (for/sum ([entry (in-list (native-memory-use))]
              #:when (eq? (device-type (car entry)) 'cpu))
      (cdr entry)))

  (define (log-mel samples rate)
    (log-mel-spectrogram (ref samples 0) #:sample-rate rate))

  (test-case "L2: an item is the decoded samples and the transcript"
    (with-utterances (us)
      (define ds (librispeech us))
      (check-true (librispeech? ds))
      (for ([u (in-list us)] [frames (in-list frame-counts)] [i (in-naturals)])
        (define-values (x transcript) (dataset-ref ds i))
        (define-values (samples _rate) (load-audio (utterance-path u)))
        (check-equal? (shape x) (list 1 frames))
        (check-eq? (dtype x) 'float32)
        (check-equal? (record x) (record samples))
        (check-equal? transcript (utterance-transcript u)))))

  (test-case "L3: the transform receives the samples and the rate"
    (with-utterances (us)
      (define ds
        (librispeech us #:transform (lambda (s r) (narrow s 1 0 (quotient r 1000)))))
      (define-values (x _transcript) (dataset-ref ds 2))
      (check-equal? (shape x) '(1 16))))

  (test-case "L1, L4, L5: a batch pads the last dimension, in index order"
    (with-utterances (us)
      (for ([transform (in-list (list (lambda (s _r) s) log-mel))])
        (define ds (librispeech us #:transform transform))
        (check-equal? (dataset-length ds) (length frame-counts))
        (define indices '(3 0 4 1))
        (define items (for/list ([i (in-list indices)]) (item-of ds i)))
        (define-values (x lengths transcripts)
          (dataset-batch ds indices default-collate))
        (check-equal? lengths (for/list ([it (in-list items)]) (last (shape (car it)))))
        (check-equal? transcripts (map cadr items))
        (check-equal? transcripts
                      '("UTTERANCE 3" "UTTERANCE 0" "UTTERANCE 4" "UTTERANCE 1"))
        (define t-max (apply max lengths))
        (define lead (drop-right (shape (car (car items))) 1))
        (check-equal? (shape x) `(4 ,@lead ,t-max))
        (for ([it (in-list items)] [t (in-list lengths)] [j (in-naturals)])
          (define row (ref x j))
          (define time (sub1 (length (shape row))))
          (check-equal? (record (narrow row time 0 t)) (record (car it)))
          (define tail
            (if (< t t-max) (flatten (tensor->list (narrow row time t (- t-max t)))) '()))
          (check-true (andmap zero? tail) (format "row ~a has a non-zero tail" j))))))

  (test-case "L6: an int64 index tensor batches like its list"
    (with-utterances (us)
      (define ds (librispeech us))
      (check-equal? (apply records (batch-of ds (tensor '(3 0 4) #:dtype 'int64)))
                    (apply records (batch-of ds '(3 0 4))))))

  (test-case "L7: another collate gets (list features transcript) per index"
    (with-utterances (us)
      (define ds (librispeech us))
      (define collated (dataset-batch ds '(3 0) (lambda (items) items)))
      (check-equal? (for/list ([it (in-list collated)]) (apply records it))
                    (for/list ([i (in-list '(3 0))]) (apply records (item-of ds i))))))

  (test-case "L13: items differing outside the last dimension are refused"
    (with-utterances (us)
      (define ds
        (librispeech us
                     #:transform (lambda (s _r)
                                   (zeros (if (> (last (shape s)) 10000) 2 1) 5))))
      (check-exn (lambda (e)
                   (and (exn:fail:contract? e)
                        (regexp-match? #rx"librispeech" (exn-message e))))
                 (lambda () (dataset-batch ds '(0 1) default-collate)))))

  (test-case "L12: contract violations blame the caller"
    (check-exn (contract-violation? #rx"utterance[?]")
               (lambda () (librispeech '("x"))))
    (with-utterances (us)
      (check-exn (contract-violation? #rx"librispeech")
                 (lambda () (librispeech us #:transform (lambda (s) s))))
      (define ds (librispeech us #:transform (lambda (s _r) (sum s))))
      (check-exn (contract-violation? #rx"tensor-with-a-time-dimension")
                 (lambda () (dataset-ref ds 0)))))

  (test-case "L8: concurrent refs and batches equal the sequential ones"
    (with-utterances (us)
      (define ds (librispeech us))
      (define indices (for/list ([i (in-range (length us))]) i))
      (define (work)
        (list (for/list ([i (in-list indices)]) (item-of ds i))
              (batch-of ds '(4 0 2 1 3))))
      (define (recorded result)
        (for/list ([part (in-list result)])
          (for/list ([v (in-list part)])
            (if (list? v) (map record v) (record v)))))
      (define expected (recorded (work)))
      (define results (make-vector 4 #f))
      (define threads
        (for/list ([k (in-range 4)])
          (thread (lambda ()
                    (vector-set! results k
                                 (with-handlers ([(lambda (_) #t) values])
                                   (for/list ([_ (in-range 3)]) (work)))))
                  #:pool 'own)))
      (for-each thread-wait threads)
      (for ([rounds (in-vector results)] [k (in-naturals)])
        (check-true (list? rounds) (format "thread ~a raised: ~a" k rounds))
        (for ([round (in-list rounds)])
          (check-equal? (recorded round) expected)))))

  (test-case "L9: loader batches at any worker count equal #:workers 0"
    (with-utterances (us)
      (define ds (librispeech us))
      (define (epoch workers shuffle?)
        (for/list ([vs (in-values-sequence
                        (dataloader ds #:batch-size 2 #:shuffle? shuffle?
                                    #:generator (make-generator 7)
                                    #:workers workers))])
          (apply records vs)))
      (for ([shuffle? (in-list '(#f #t))])
        (define expected (epoch 0 shuffle?))
        (for ([workers (in-list '(1 2 4))])
          (check-equal? (epoch workers shuffle?) expected
                        (format "workers ~a, shuffle? ~a" workers shuffle?))))))

  (test-case "L10: the Racket heap per item does not grow with its length"
    (with-utterances (us)
      (define ds (librispeech (list (librispeech-fixture-utterance) (car us))))
      (define long-ref (median-allocation (lambda () (dataset-ref ds 0))))
      (define short-ref (median-allocation (lambda () (dataset-ref ds 1))))
      (check-true (< long-ref 32768) (format "~a bytes per long ref" long-ref))
      (check-true (< short-ref 32768) (format "~a bytes per short ref" short-ref))
      (check-true (< (abs (- long-ref short-ref)) 4096)
                  (format "long ~a, short ~a" long-ref short-ref))
      (define (batch indices)
        (median-allocation (lambda () (dataset-batch ds indices default-collate))))
      (define long-batch (batch '(0 0 0 0)))
      (define short-batch (batch '(1 1 1 1)))
      (check-true (< long-batch 65536) (format "~a bytes per long batch" long-batch))
      (check-true (< short-batch 65536) (format "~a bytes per short batch" short-batch))
      (check-true (< (abs (- long-batch short-batch)) 8192)
                  (format "long ~a, short ~a" long-batch short-batch))))

  (test-case "L11: a worker epoch leaves no CPU native memory behind"
    (with-utterances (us)
      (define ds (librispeech us))
      (settle!)
      (define before (cpu-bytes))
      (for ([(x lengths transcripts) (dataloader ds #:batch-size 2 #:workers 4)])
        (void))
      (settle!)
      (define after (cpu-bytes))
      (check-true (<= after before) (format "~a bytes before, ~a after" before after)))))

(module+ test
  (require (only-in "private/wedge.rkt" call-with-process-deadline))
  (define cases
    (module-path-index-join '(submod ".." cases)
                            (variable-reference->module-path-index (#%variable-reference))))
  (call-with-process-deadline 300 (lambda () (dynamic-require cases #f))))
