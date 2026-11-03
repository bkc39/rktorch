#lang racket/base

;; Single-threaded cost of each op of the scope-pipelines augmentations,
;; rktorch against PyTorch on the same shapes:
;;   racket pipeline-ops.rkt [--pipelines jpeg,cifar,text,overhead] [--out f.jsonl]
;; Each step runs for at least a second; rktorch runs it bare (off: its
;; tensors go to the finalizers) and inside a with-tensor-scope per iteration
;; (scope: freed at the scope's exit). "tensors" is how many tensors one
;; iteration registers, read from the scope's exit-frees.

(require (for-syntax racket/base)
         (only-in json read-json write-json)
         (only-in racket/cmdline command-line)
         (only-in racket/file file->bytes)
         (only-in racket/list make-list take)
         (only-in racket/math exact-round)
         (only-in racket/port port->lines)
         (only-in racket/runtime-path define-runtime-path)
         (only-in racket/string string-split)
         (only-in torch
                  + / < narrow rand select shape stack to-dtype unsqueeze where
                  zeros ~>)
         (only-in torch/foreign/raw/scope scope-diagnostics)
         (only-in torch/foreign/scope with-tensor-scope)
         (only-in torch/vision/image decode-image read-image)
         (only-in torch/vision/transforms
                  convert-image-dtype imagenet-mean imagenet-std normalize
                  random-crop random-resized-crop)
         (only-in "diag-counters.rkt" clock-ms)
         (only-in "workloads/pipelines.rkt"
                  cifar-augment cifar-mean cifar-sources cifar-std flip-one
                  jpeg-augment jpeg-files text-sources text-window))

(define-runtime-path twin "python/pipelines.py")

(define (cycling xs f)
  (define v (list->vector xs))
  (define k 0)
  (lambda ()
    (set! k (modulo (add1 k) (vector-length v)))
    (f (vector-ref v k))))

(define (jpeg-steps)
  (define files (take (map car (vector->list (jpeg-files))) 32))
  (define blobs (map file->bytes files))
  (define images (map (lambda (b) (decode-image b #:mode 'rgb)) blobs))
  (define floats (map convert-image-dtype images))
  (define out
    (~> (car floats) (random-resized-crop 224) flip-one
        (normalize imagenet-mean imagenet-std)))
  (define batch (make-list 32 out))
  (list (cons "read" (cycling files file->bytes))
        (cons "decode" (cycling blobs (lambda (b) (decode-image b #:mode 'rgb))))
        (cons "to-float (full image)" (cycling images convert-image-dtype))
        (cons "rrc (float input)" (cycling floats (lambda (f) (random-resized-crop f 224))))
        (cons "flip" (lambda () (flip-one out)))
        (cons "normalize" (lambda () (normalize out imagenet-mean imagenet-std)))
        (cons "sample (read..normalize)"
              (cycling files (lambda (p) (jpeg-augment (read-image p #:mode 'rgb)))))
        (cons "collate (stack 32)" (lambda () (stack batch)))))

(define (cifar-steps)
  (define-values (images _labels) (cifar-sources))
  (define pixels (~> (select images 0 0) (to-dtype 'float32) (/ 255.0)))
  (define ks (for/list ([k (in-range (car (shape images)))]) k))
  (define batch (make-list 128 pixels))
  (list (cons "select" (cycling ks (lambda (k) (select images 0 k))))
        (cons "to-float /255" (lambda () (~> (select images 0 0) (to-dtype 'float32) (/ 255.0))))
        (cons "random crop pad 4"
              (lambda () (~> pixels (unsqueeze 0) (random-crop #:padding 4) (select 0 0))))
        (cons "flip" (lambda () (flip-one pixels)))
        (cons "normalize" (lambda () (normalize pixels cifar-mean cifar-std)))
        (cons "sample (select..normalize)"
              (cycling ks (lambda (k) (cifar-augment (select images 0 k)))))
        (cons "collate (stack 128)" (lambda () (stack batch)))))

(define (text-steps)
  (define-values (ids mask-id) (text-sources))
  (define m (~> (shape ids) car (- 128)))
  (define span (narrow ids 0 0 129))
  (define x (narrow span 0 0 128))
  (define r (rand 128))
  (define mask (< r 0.15))
  (define batch (make-list 64 (+ x 0)))
  (list (cons "narrow 129" (lambda () (narrow ids 0 (random m) 129)))
        (cons "rand 128" (lambda () (rand 128)))
        (cons "< 0.15" (lambda () (< r 0.15)))
        (cons "where" (lambda () (where mask mask-id x)))
        (cons "narrow x, y" (lambda () (values (narrow span 0 0 128) (narrow span 0 1 128))))
        (cons "sample (window)" (lambda () (text-window ids (random m) mask-id)))
        (cons "collate (stack 64, x and y)" (lambda () (values (stack batch) (stack batch))))))

(define (overhead-steps)
  (define small (zeros 1))
  (list (cons "add scalar to [1]" (lambda () (+ small 1.0)))))

(define (timed step [min-ms 1000.0] [min-iters 20])
  (step)
  (define gc0 (current-gc-milliseconds))
  (define t0 (clock-ms))
  (let loop ([iters 1])
    (step)
    (define elapsed (- (clock-ms) t0))
    (if (and (>= elapsed min-ms) (>= iters min-iters))
        (values (/ (* 1000.0 elapsed) iters) iters
                (/ (- (current-gc-milliseconds) gc0) (exact->inexact elapsed)))
        (loop (add1 iters)))))

(define (tensors-per-call step)
  (define before (hash-ref (scope-diagnostics) 'exit-frees))
  (with-tensor-scope (step) (void))
  (- (hash-ref (scope-diagnostics) 'exit-frees) before))

(define (rktorch-ops name)
  (define steps
    (case name
      [(jpeg) (jpeg-steps)]
      [(cifar) (cifar-steps)]
      [(text) (text-steps)]
      [(overhead) (overhead-steps)]))
  (for/list ([s (in-list steps)])
    (define step (cdr s))
    (define-values (off-us off-iters off-gc) (timed (lambda () (call-with-values step void))))
    (define-values (scope-us scope-iters scope-gc)
      (timed (lambda () (with-tensor-scope (call-with-values step void)))))
    (hasheq 'pipeline (symbol->string name) 'op (car s)
            'off-us off-us 'scope-us scope-us 'tensors (tensors-per-call step)
            'off-gc-share off-gc 'scope-gc-share scope-gc)))

(define (pytorch-ops name)
  (define-values (sub out in err)
    (subprocess #f #f (current-error-port)
                (find-executable-path "python3") "-P" (path->string twin)
                "--ops" (symbol->string name)))
  (close-output-port in)
  (define lines (port->lines out))
  (subprocess-wait sub)
  (close-input-port out)
  (for/list ([line (in-list lines)] #:unless (string=? line ""))
    (read-json (open-input-string line))))

(define (show us)
  (cond
    [(not us) "-"]
    [(>= us 100) (number->string (exact-round us))]
    [else (real->decimal-string us 1)]))

(define (print-table name rk py)
  (printf "\n## ~a (us per call, 1 ATen thread)\n\n" name)
  (displayln "| op | PyTorch | rktorch off | rktorch scope | off/PyTorch | tensors | off GC share |")
  (displayln "|---|---|---|---|---|---|---|")
  (define (py-us op)
    (for/first ([p (in-list py)] #:when (equal? (hash-ref p 'op) op)) (hash-ref p 'us)))
  (for ([r (in-list rk)])
    (define p (py-us (hash-ref r 'op)))
    (printf "| ~a | ~a | ~a | ~a | ~a | ~a | ~a% |\n" (hash-ref r 'op) (show p)
            (show (hash-ref r 'off-us)) (show (hash-ref r 'scope-us))
            (if p (~> (hash-ref r 'off-us) (/ p) (real->decimal-string 2)) "-")
            (hash-ref r 'tensors)
            (~> (hash-ref r 'off-gc-share) (* 100) exact-round)))
  (for ([p (in-list py)]
        #:unless (for/or ([r (in-list rk)]) (equal? (hash-ref r 'op) (hash-ref p 'op))))
    (printf "| ~a | ~a | - | - | - | - | - |\n" (hash-ref p 'op) (show (hash-ref p 'us)))))

(module+ main
  (define names '(overhead text cifar jpeg))
  (define out-path #f)
  (command-line
   #:once-each
   [("--pipelines") p "Comma-separated pipelines"
                    (set! names (map string->symbol (string-split p ",")))]
   [("--out") p "JSON-lines file the timings are appended to" (set! out-path p)])
  (for ([name (in-list names)])
    (define py (pytorch-ops name))
    (define rk (rktorch-ops name))
    (when out-path
      (call-with-output-file out-path #:exists 'append
        (lambda (o)
          (for ([r (in-list (append (map (lambda (h) (hash-set h 'side "pytorch")) py)
                                    (map (lambda (h) (hash-set h 'side "rktorch")) rk)))])
            (write-json r o)
            (newline o)))))
    (print-table name rk py)
    (flush-output)))
