#lang racket/base

(module+ test
  ;; whole-module: define-runtime-path needs phase-1 bindings only-in strips
  (require racket/runtime-path
           (only-in rackunit check-equal? check-true)
           (only-in "../main.rkt"
                    abs bytes->tensor item max mean mul sub tensor->bytes
                    tensor-shape to-dtype)
           (only-in "../vision/image.rkt" read-image)
           (only-in "../vision/transforms.rkt"
                    center-crop convert-image-dtype imagenet-normalize
                    normalize resize imagenet-mean imagenet-std)
           "private/python-env.rkt")

  (define-runtime-path fixtures "../vision/fixtures/images")

  (define (hex->bytes s)
    (apply bytes
           (for/list ([i (in-range 0 (string-length s) 2)])
             (string->number (substring s i (+ i 2)) 16))))

  (define (unpack j dtype)
    (bytes->tensor (hex->bytes (hash-ref j 'hex)) dtype (hash-ref j 'shape)))

  (define (gap a b)
    (define d (abs (sub (to-dtype a 'float64) (to-dtype b 'float64))))
    (values (item (max d)) (item (mean d))))

  (define (fixture name) (build-path fixtures name))

  (define (jpeg? name) (regexp-match? #rx"[.]jpg$" name))

  (define (check-transform label ours theirs tolerance)
    (check-equal? (tensor-shape ours) (hash-ref theirs 'shape)
                  (format "~a: shape" label))
    (define-values (worst _) (gap ours (unpack theirs 'float32)))
    (check-true (<= worst tolerance)
                (format "~a: max |difference| ~a over ~a" label worst
                        tolerance)))

  (cond
    [(not (python-module-available? "torchvision"))
     (printf "[image-parity-test] skipped: python3 `torchvision` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (define j (python-check "image_parity.py"))

     (for* ([(key modes) (in-hash (hash-ref j 'decoded))]
            [mode (in-list '(unchanged rgb))]
            ;; torchvision 0.27 answers a palette PNG's unchanged mode with
            ;; one channel of uninitialized memory, different on every
            ;; decode; stb expands the palette, as the rgb mode does
            #:unless (and (eq? key 'palette.png) (eq? mode 'unchanged)))
       (define name (symbol->string key))
       (define theirs (hash-ref modes mode))
       (define ours (read-image (fixture name) #:mode mode))
       (define label (format "read-image ~a #:mode '~a" name mode))
       (check-equal? (tensor-shape ours) (hash-ref theirs 'shape)
                     (format "~a: shape" label))
       (cond
         [(jpeg? name)
          ;; stb and libjpeg-turbo differ in the IDCT and in chroma
          ;; upsampling, by a count or two
          (define-values (worst typical) (gap ours (unpack theirs 'uint8)))
          (check-true (and (<= worst 3) (<= typical 0.25))
                      (format "~a: max ~a mean ~a" label worst typical))]
         [else
          (check-equal? (tensor->bytes ours) (hex->bytes (hash-ref theirs 'hex))
                        (format "~a: every byte" label))]))
     (check-equal? (tensor->bytes (read-image (fixture "palette.png")))
                   (tensor->bytes (read-image (fixture "palette.png")
                                              #:mode 'rgb))
                   "read-image: a palette PNG comes back as RGB")

     (define gradient (convert-image-dtype (read-image (fixture "gradient.png"))))
     (for ([case (in-list (hash-ref j 'resize))])
       (define size (hash-ref case 'size))
       (define antialias? (hash-ref case 'antialias))
       (check-transform (format "resize ~a #:antialias? ~a" size antialias?)
                        (resize gradient size #:antialias? antialias?)
                        case 1e-5))
     (for ([case (in-list (hash-ref j 'crop))])
       (check-transform (format "center-crop ~a" (hash-ref case 'size))
                        (center-crop gradient (hash-ref case 'size))
                        case 0.0))
     (check-transform "normalize"
                      (normalize gradient imagenet-mean imagenet-std)
                      (hash-ref j 'normalize) 1e-6)
     (check-equal? (tensor->bytes (convert-image-dtype (mul gradient 0.75)
                                                       'uint8))
                   (hex->bytes (hash-ref (hash-ref j 'to_uint8) 'hex))
                   "convert-image-dtype: float32 -> uint8")

     (define (chain pixels)
       (imagenet-normalize
        (center-crop (resize (convert-image-dtype pixels) 256) 224)))
     (define their-pixels
       (unpack (hash-ref (hash-ref (hash-ref j 'decoded) 'smooth-401x299.jpg)
                         'unchanged)
               'uint8))
     (check-transform "resize 256, center-crop 224, imagenet-normalize"
                      (chain their-pixels) (hash-ref j 'chain) 1e-5)
     (define-values (worst typical)
       (gap (chain (read-image (fixture "smooth-401x299.jpg")))
            (unpack (hash-ref j 'chain) 'float32)))
     (check-true (and (<= worst 0.05) (<= typical 0.005))
                 (format "the chain from our own decode: max ~a mean ~a"
                         worst typical))]))
