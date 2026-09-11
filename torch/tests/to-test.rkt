#lang racket/base

;; CPU cases run everywhere; the CUDA/MPS cases are `when`-guarded, so the
;; suite verifies device moves for real on a CUDA host or Apple Silicon.

(module+ test
  (require (only-in racket/file make-temporary-file)
           (only-in racket/string string-split)
           rackunit
           (only-in (submod "../foreign.rkt" unsafe) tensor-free! to!)
           "../main.rkt"
           "../nn.rkt")

  (define (current-rss-bytes)
    (and (file-exists? "/proc/self/status")
         (call-with-input-file "/proc/self/status"
           (lambda (in)
             (for/first ([l (in-lines in)]
                         #:when (regexp-match? #rx"^VmRSS" l))
               (* 1024 (string->number (cadr (string-split l)))))))))

  (define (bytes-on dev)
    (cond [(assoc dev (native-memory-use)) => cdr]
          [else 0]))

  (define (ledger-entries)
    (cdr (assq 'ledger-entries (finalizer-diagnostics))))

  ;; unconditional rounds: earlier tests' finalizers may still be in flight
  (define (settle!)
    (for ([_ (in-range 3)])
      (collect-garbage)
      (sleep 0.01)))

  (define (param-values m)
    (map tensor->list (parameters m)))

  (define-layer Mixed (w b mask)
    #:init ([mask (tensor '(1.0 0.0 1.0))])
    (set! w (Parameter (tensor '(1.0 2.0 3.0))))
    (set! b (Buffer (tensor '(0.5 0.5 0.5))))
    #:forward (x)
    (sum (mul (mul (mul w w) x) mask)))

  (test-case "to on a tensor: dtype, device forms, both axes, argument order"
    (define t (tensor '(1 2 3)))
    (define wide (to t 'float64))
    (check-equal? (tensor-dtype wide) 'float64)
    (check-equal? (tensor->list wide) '(1.0 2.0 3.0))
    (check-equal? (tensor-dtype t) 'int64 "the source is untouched")
    (check-equal? (tensor-device (to t 'cpu)) (cpu-device))
    (check-equal? (tensor-device (to t (cpu-device))) (cpu-device))
    (check-equal? (tensor-device (to t (device 'cpu))) (cpu-device))
    (define both (to t (cpu-device) 'float64))
    (check-equal? (tensor-dtype both) 'float64)
    (check-equal? (tensor-device both) (cpu-device))
    (check-exn #rx"a dtype target takes no second argument"
               (lambda () (to t 'float64 'float32)))
    (check-exn exn:fail:contract? (lambda () (to t 'float16)))
    (check-exn exn:fail:contract? (lambda () (to 5 'cpu)))
    (check-exn exn:fail:contract? (lambda () (to t 'cpu 'cpu))))

  (test-case "to is the identity when nothing changes, with no ledger entry"
    (define t (tensor '(1.0 2.0)))
    (define entries (ledger-entries))
    (check-eq? (to t 'cpu) t)
    (check-eq? (to t 'float32) t)
    (check-eq? (to t (cpu-device) 'float32) t)
    (check-eq? (to-device t 'cpu) t "to-device is a primitive alias")
    (check-eq? (to-dtype t 'float32) t "to-dtype is a primitive alias")
    (check-equal? (ledger-entries) entries)
    (check-false (eq? (to-dtype t 'int64) t))
    (check-equal? (tensor-dtype (to-dtype t 'int64)) 'int64))

  (test-case "to on a layer moves parameters and buffers in place"
    (define mask (tensor '(1.0 0.0 1.0)))
    (define m (Mixed mask))
    (define ps (parameters m))
    (define bs (buffers m))
    (check-true (to-able? m))
    (check-false (to-able? (ones 1)))
    (check-eq? (to m 'float64) m "returns the layer")
    (check-equal? (map eq? (parameters m) ps) '(#t)
                  "parameter objects keep their identity")
    (check-equal? (map eq? (buffers m) bs) '(#t))
    (check-equal? (map tensor-dtype (parameters m)) '(float64))
    (check-equal? (map tensor-dtype (buffers m)) '(float64))
    (check-equal? (tensor-dtype mask) 'float32
                  "a plain tensor field is not a parameter and stays")
    (check-equal? (tensor->list (car (parameters m))) '(1.0 2.0 3.0))
    (check-eq? (~> m (to 'float32)) m "threads")
    (check-equal? (map tensor-dtype (parameters m)) '(float32)))

  (test-case "every layer kind answers to: containers, procedures, hand-written"
    (manual-seed! 0)
    (define seq (Sequential (Linear 2 3) relu (Linear 3 1)))
    (define ll (LayerList (list (Linear 2 2) (Linear 2 2))))
    (define w (Parameter (ones 2)))
    (define proc
      (procedure->Layer (lambda (x) (mul x w))
                        #:parameters (list (cons "w" w))))
    (struct Hand (p)
      #:methods gen:layer
      [(define (layer-forward self . inputs) (mul (car inputs) (Hand-p self)))
       (define (layer-parameters self) (list (Hand-p self)))])
    (define hand (Hand (Parameter (ones 2))))
    (for ([m (in-list (list seq ll proc hand))])
      (define before (param-values m))
      (check-eq? (to m 'float64) m)
      (check-true (andmap (lambda (p) (eq? (tensor-dtype p) 'float64))
                          (parameters m)))
      (check-equal? (param-values m) before)
      (check-eq? (to m 'float32) m)
      (check-equal? (param-values m) before))
    (check-equal? (tensor->list (seq (ones 1 2)))
                  (tensor->list (seq (ones 1 2))) "still callable"))

  (test-case "autograd survives an in-place move: leaf, grad, optimizers"
    (define m (Mixed))
    (define w (car (parameters m)))
    (backward! (m (ones 3)))
    (check-equal? (tensor->list (grad w)) '(2.0 0.0 6.0))
    (to m 'float64)
    (check-true (requires-grad? w))
    (check-true (has-grad? w))
    (check-equal? (tensor-dtype (grad w)) 'float64 "the grad moved along")
    (check-equal? (tensor->list (grad w)) '(2.0 0.0 6.0))
    (backward! (m (to (ones 3) 'float64)))
    (check-equal? (tensor->list (grad w)) '(4.0 0.0 12.0)
                  "still a leaf: a fresh backward accumulates")
    (step! (sgd (parameters m) #:lr 0.5))
    (check-equal? (tensor->list w) '(-1.0 2.0 -3.0))
    (check-equal? (tensor-dtype w) 'float64)
    ;; Adam built before the move: its moments are created lazily at the
    ;; first step, on the parameter's device and dtype
    (define m2 (Mixed))
    (define opt (adam (parameters m2) #:lr 0.1))
    (to m2 'float64)
    (backward! (m2 (to (ones 3) 'float64)))
    (step! opt)
    (define w2 (car (parameters m2)))
    (check-equal? (tensor-dtype w2) 'float64)
    (check-true (< (car (tensor->list w2)) 1.0) "the moved parameter stepped")
    (check-true (= (cadr (tensor->list w2)) 2.0) "a zero-grad entry is held"))

  (test-case "checkpoints compose with moves in either order"
    (manual-seed! 0)
    (define net (Linear 3 2))
    (define path (make-temporary-file "rkt-to-~a.safetensors"))
    (save-state! net path)
    (define expected (param-values net))
    (define moved-first (to (Linear 3 2) 'float64))
    (load-state! moved-first path)
    (check-equal? (map tensor-dtype (parameters moved-first))
                  '(float64 float64) "load-state! keeps the moved dtype")
    (check-equal? (param-values moved-first) expected)
    (define loaded-first (Linear 3 2))
    (load-state! loaded-first path)
    (to loaded-first 'float64)
    (check-equal? (param-values loaded-first) expected)
    (to loaded-first 'float32)
    (save-state! loaded-first path)
    (define again (Linear 3 2))
    (load-state! again path)
    (check-equal? (param-values again) expected))

  (test-case "an in-place move re-accounts the same ledger entry"
    (settle!)
    (define base (bytes-on (cpu-device)))
    (define m (Linear 2048 2048)) ;; 16 MiB weight + 8 KiB bias, float32
    (settle!)
    (define entries (ledger-entries))
    (define before (- (bytes-on (cpu-device)) base))
    (check-true (>= before (* 16 1024 1024)))
    (define cmu-before (current-memory-use))
    (to m 'float64)
    (check-true (> (- (current-memory-use) cmu-before) (* 8 1024 1024))
                "the phantom delta is visible to the GC's own accounting")
    (settle!)
    (check-equal? (ledger-entries) entries "replaced, not added")
    (check-equal? (- (bytes-on (cpu-device)) base) (* 2 before)
                  "the CPU bucket grew by exactly the dtype delta")
    (to m 'float32)
    (settle!)
    (check-equal? (- (bytes-on (cpu-device)) base) before
                  "moving back returns the bucket to where it was")
    (check-equal? (ledger-entries) entries)
    ;; the layer must outlive the checks, or the GC reclaims it first
    (check-true (layer? m)))

  (test-case "tensor-free! after an in-place move unaccounts synchronously"
    (settle!)
    (define base (bytes-on (cpu-device)))
    (define t (zeros 1024 1024)) ;; 4 MiB
    (to! t 'float64)
    (check-equal? (tensor-dtype t) 'float64)
    (check-true (>= (- (bytes-on (cpu-device)) base) (* 8 1024 1024)))
    (tensor-free! t)
    ;; no collect-garbage on purpose — the free itself must unaccount
    (check-true (< (- (bytes-on (cpu-device)) base) (* 512 1024))))

  (test-case "to results are collected under pressure without manual collects"
    (settle!)
    (define base (bytes-on (cpu-device)))
    (define rss-before (current-rss-bytes))
    (define high-water
      (for/fold ([hw 0]) ([_ (in-range 100)])
        (void (to (zeros 1024 1024) 'float64)) ;; 4 MiB + 8 MiB, dropped
        (max hw (- (bytes-on (cpu-device)) base))))
    (check-true (< high-water (* 600 1024 1024))
                (format "high-water ~a of ~a churned — pressure never fired"
                        high-water (* 1200 1024 1024)))
    (when rss-before
      (check-true (< (- (current-rss-bytes) rss-before) (* 700 1024 1024))
                  "RSS grew by ~the whole churn — native buffers not freed")))

  (test-case "in-place flips release the old storage"
    (settle!)
    (define base (bytes-on (cpu-device)))
    (define m (Linear 2048 2048)) ;; 16 MiB float32
    (settle!)
    (define entries (ledger-entries))
    (define before (- (bytes-on (cpu-device)) base))
    (define rss-before (current-rss-bytes))
    (for ([_ (in-range 50)]) ;; 50 x (32 + 16) MiB of storage retired
      (to m 'float64)
      (to m 'float32))
    (check-equal? (ledger-entries) entries)
    (check-equal? (- (bytes-on (cpu-device)) base) before)
    (when rss-before
      (check-true (< (- (current-rss-bytes) rss-before) (* 256 1024 1024))
                  "RSS grew across the flips — set_data kept old storage"))
    (check-true (layer? m)))

  (define (check-device-moves dev dev-name)
    (set-default-device! 'cpu)
    (settle!)
    (define base-cpu (bytes-on (cpu-device)))
    (define base-dev (bytes-on dev))

    (define t (zeros 1024 1024)) ;; 4 MiB
    (define g (to t dev))
    (check-equal? (tensor-device g) dev)
    (check-eq? (to g dev) g (format "~a: identity on the device" dev-name))
    (check-equal? (tensor-device (to t dev-name)) dev "the symbol form")
    (check-equal? (tensor->list (to g 'cpu)) (tensor->list t))
    (settle!)
    (check-true (>= (- (bytes-on dev) base-dev) (* 4 1024 1024))
                (format "~a bucket charged" dev-name))
    (check-true (< (- (bytes-on (cpu-device)) base-cpu) (* 5 1024 1024))
                "the CPU bucket holds only the source")
    (set! g #f)
    (reclaim-native-memory!)
    (check-true (< (- (bytes-on dev) base-dev) (* 1024 1024))
                (format "~a bucket released" dev-name))

    (manual-seed! 0)
    (define m (Linear 1024 1024))
    (define before (param-values m))
    (define opt (adam (parameters m) #:lr 0.01))
    (settle!)
    (define entries (ledger-entries))
    (define cpu-with-model (- (bytes-on (cpu-device)) base-cpu))
    (check-eq? (to m dev) m)
    (check-equal? (map tensor-device (parameters m)) (list dev dev))
    (settle!)
    (check-equal? (ledger-entries) entries "moved, not duplicated")
    (check-true (< (- (bytes-on (cpu-device)) base-cpu)
                   (- cpu-with-model (* 4 1024 1024)))
                "the CPU bucket dropped by the model's bytes")
    (check-true (>= (- (bytes-on dev) base-dev) (* 4 1024 1024))
                "the device bucket gained them")
    (define x (to (ones 1 1024) dev))
    (define y (m x))
    (check-equal? (tensor-device y) dev)
    (backward! (sum y))
    (check-equal? (tensor-device (grad (car (parameters m)))) dev)
    (step! opt)
    (check-eq? (to m 'cpu) m)
    (check-equal? (map tensor-device (parameters m))
                  (list (cpu-device) (cpu-device)))
    (check-equal? (tensor-device (grad (car (parameters m)))) (cpu-device)
                  "the grad came back too")
    (check-false (equal? (param-values m) before) "the step took effect")
    (settle!)
    (check-true (< (- (bytes-on dev) base-dev) (* 1024 1024))
                "nothing of the model is left on the device")

    (define path (make-temporary-file "rkt-to-dev-~a.safetensors"))
    (to m dev)
    (save-state! m path)
    (define reloaded (Linear 1024 1024))
    (load-state! reloaded path)
    (check-equal? (param-values reloaded) (param-values (to m 'cpu)))

    (define high-water
      (for/fold ([hw 0]) ([_ (in-range 100)])
        (void (to (zeros 1024 1024) dev)) ;; 4 MiB there, dropped
        (max hw (- (bytes-on dev) base-dev))))
    (check-true (< high-water (* 600 1024 1024))
                (format "~a high-water ~a — pressure never fired"
                        dev-name high-water))
    (reclaim-native-memory!)
    (check-true (< (- (bytes-on dev) base-dev) (* 1024 1024))))

  (test-case "cuda: tensor and layer moves, ledger buckets, churn"
    (when (cuda-available?)
      ;; warm-up with the same shapes: libtorch keeps its cuBLAS workspace
      ;; allocated after the first matmul, which the gauge would otherwise
      ;; report as a leak
      (let ([warm (to (Linear 1024 1024) 'cuda)])
        (backward! (sum (warm (to (ones 1 1024) 'cuda))))
        (step! (adam (parameters warm) #:lr 0.01)))
      (reclaim-native-memory!)
      (define allocated-before
        (cdr (assq 'allocated (cuda-memory-stats))))
      (check-device-moves (cuda-device) 'cuda)
      (define both (to (tensor '(1 2 3)) 'cuda 'float64))
      (check-equal? (tensor-device both) (cuda-device))
      (check-equal? (tensor-dtype both) 'float64)
      (set! both #f)
      (reclaim-native-memory!)
      (check-true (<= (- (cdr (assq 'allocated (cuda-memory-stats)))
                         allocated-before)
                      (* 1024 1024))
                  "the caching allocator's own gauge returned to baseline")))

  (test-case "mps: tensor and layer moves, ledger buckets, churn"
    (when (mps-available?)
      (check-device-moves (mps-device) 'mps)
      (mps-empty-cache!))))
