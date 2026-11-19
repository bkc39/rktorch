#lang racket/base

(require (only-in json write-json)
         (only-in racket/cmdline command-line)
         (only-in torch/tests/private/op-calls op-calls-of)
         (only-in torch/vision/hymenoptera hymenoptera-root)
         (only-in torch/vision/image read-image)
         (only-in torch/vision/image-folder image-folder-samples)
         (only-in torch/vision/transforms
                  convert-image-dtype imagenet-mean imagenet-std normalize
                  random-horizontal-flip random-resized-crop resize))

(define chain
  (list (cons 'random-resized-crop (lambda (x) (random-resized-crop x 224)))
        (cons 'flip random-horizontal-flip)
        (cons 'convert convert-image-dtype)
        (cons 'normalize (lambda (x) (normalize x imagenet-mean imagenet-std)))))

(define asides
  (list (cons 'resize-256-uint8 (lambda (decoded _float) (resize decoded 256)))
        (cons 'resize-256-float32 (lambda (_decoded float) (resize float 256)))))

(define (decode path)
  (read-image path #:mode 'rgb))

(define step-names
  (append '(decode) (map car chain) (map car asides)))

(define (timed totals name thunk)
  (define t0 (current-inexact-monotonic-milliseconds))
  (define y (thunk))
  (when totals
    (hash-update! totals name
                  (lambda (v) (+ v (- (current-inexact-monotonic-milliseconds) t0)))
                  0.0))
  y)

(define (run-pass! paths totals)
  (for ([path (in-vector paths)])
    (define decoded (timed totals 'decode (lambda () (decode path))))
    (for/fold ([x decoded]) ([step (in-list chain)])
      (timed totals (car step) (lambda () ((cdr step) x))))
    (define float (convert-image-dtype decoded))
    (for ([aside (in-list asides)])
      (timed totals (car aside) (lambda () ((cdr aside) decoded float))))))

(define (calls-of thunk)
  (thunk)
  (op-calls-of thunk))

(define (calls-per-step path)
  (define decoded (decode path))
  (define float (convert-image-dtype decoded))
  (define-values (counts _)
    (for/fold ([counts (hasheq 'decode (calls-of (lambda () (decode path))))]
               [x decoded])
              ([step (in-list chain)])
      (define f (cdr step))
      (values (hash-set counts (car step) (calls-of (lambda () (f x))))
              (f x))))
  (for/fold ([counts counts]) ([aside (in-list asides)])
    (hash-set counts (car aside)
              (calls-of (lambda () ((cdr aside) decoded float))))))

(define (measure passes)
  (define paths
    (for/vector ([sample (in-list (image-folder-samples
                                   (build-path (hymenoptera-root) "train")))])
      (car sample)))
  (run-pass! paths #f)
  (define totals (make-hasheq))
  (collect-garbage)
  (define gc0 (current-gc-milliseconds))
  (define t0 (current-inexact-monotonic-milliseconds))
  (for ([_ (in-range passes)])
    (run-pass! paths totals))
  (define wall (- (current-inexact-monotonic-milliseconds) t0))
  (define n (* passes (vector-length paths)))
  (define (per-sample ms) (/ (* 1000.0 ms) n))
  (hasheq 'impl "rktorch"
          'samples n
          'wall_us_per_sample (per-sample wall)
          'gc_us_per_sample (per-sample (- (current-gc-milliseconds) gc0))
          'steps (for/hasheq ([name (in-list step-names)])
                   (values name (per-sample (hash-ref totals name))))
          'calls (calls-per-step (vector-ref paths 0))))

(module+ main
  (define passes
    (command-line
     #:program "jpeg-steps"
     #:args ([passes "3"])
     (string->number passes)))
  (write-json (measure passes))
  (newline))
