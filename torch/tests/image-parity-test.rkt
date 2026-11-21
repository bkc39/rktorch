#lang racket/base

(module+ test
  (require (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in rackunit check-equal? check-true)
           (only-in "../main.rkt"
                    abs bytes->tensor cpu-capability item max mean mul permute
                    sub tensor->bytes tensor-shape to-dtype)
           (only-in "../vision/image.rkt" read-image)
           (only-in "../vision/transforms.rkt"
                    center-crop convert-image-dtype imagenet-normalize
                    normalize resize imagenet-mean imagenet-std)
           (only-in (submod "../vision/transforms.rkt" private)
                    native-uint8-resize? resize/path resized-crop)
           (only-in "private/within.rkt" bounded-test)
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

  (define (check-transform label ours theirs tolerance [dtype 'float32])
    (check-equal? (tensor-shape ours) (hash-ref theirs 'shape)
                  (format "~a: shape" label))
    (define-values (worst _) (gap ours (unpack theirs dtype)))
    (check-true (<= worst tolerance)
                (format "~a: max |difference| ~a over ~a" label worst
                        tolerance)))

  (define (check-bytes label ours theirs)
    (check-equal? (tensor-shape ours) (hash-ref theirs 'shape)
                  (format "~a: shape" label))
    (check-equal? (tensor->bytes ours) (hex->bytes (hash-ref theirs 'hex))
                  (format "~a: every byte" label)))

  (define (case-label name case)
    (format "~a ~a #:antialias? ~a" name (hash-ref case 'size)
            (hash-ref case 'antialias)))

  (cond
    [(not (python-module-available? "torchvision"))
     (printf "[image-parity-test] skipped: python3 `torchvision` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (define j (python-check "image_parity.py"))

     (check-equal? (sort (map symbol->string (hash-keys (hash-ref j 'decoded)))
                         string<?)
                   (sort (remove "NOTICE" (map path->string (directory-list fixtures)))
                         string<?)
                   "image_parity.py decodes every image fixture")

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
                        case 1e-6))
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
                         worst typical))

     (define (channels-last image)
       (define hwc (permute image 1 2 0))
       (permute (bytes->tensor (tensor->bytes hwc) 'uint8 (tensor-shape hwc)) 2 0 1))
     (define (decoded-pixels name mode)
       (channels-last
        (unpack (hash-ref (hash-ref (hash-ref j 'decoded) name) mode) 'uint8)))
     (define sources
       (hash 'gradient (decoded-pixels 'gradient.png 'unchanged)
             'photo (decoded-pixels 'smooth-401x299.jpg 'rgb)))

     (bounded-test "P0: the capability and the uint8 path are torchvision's"
       (check-equal? (symbol->string (cpu-capability)) (hash-ref j 'capability))
       (check-equal? native-uint8-resize? (hash-ref j 'native_uint8)
                     (format "on ~a" (hash-ref j 'machine))))

     (bounded-test "P1: uint8 resize equals torchvision's byte for byte"
       (for ([case (in-list (hash-ref j 'resize_u8))])
         (define src (string->symbol (hash-ref case 'src)))
         (check-bytes (case-label (format "uint8 ~a resize" src) case)
                      (resize (hash-ref sources src) (hash-ref case 'size)
                              #:antialias? (hash-ref case 'antialias))
                      case)))

     (define (window-label name case)
       (format "~a ~a" (case-label name case) (hash-ref case 'window)))

     (bounded-test "P2: uint8 resized-crop equals torchvision's byte for byte"
       (for ([case (in-list (hash-ref j 'resized_crop_u8))])
         (define size (hash-ref case 'size))
         (check-bytes (window-label "uint8 resized-crop" case)
                      (resized-crop (hash-ref sources 'photo) (hash-ref case 'window)
                                    (list size size)
                                    #:antialias? (hash-ref case 'antialias))
                      case)))

     (bounded-test "P3: the uint8 float path equals v1 resize byte for byte"
       (for ([case (in-list (hash-ref j 'resize_u8_float_path))])
         (check-bytes (case-label "uint8 float path" case)
                      (resize/path (hash-ref sources 'gradient) (hash-ref case 'size)
                                   (hash-ref case 'antialias) 'float)
                      case)))

     (bounded-test "P4: float64 resize and float32 resized-crop within float rounding"
       (for ([case (in-list (hash-ref j 'resize_f64))])
         (check-transform (case-label "float64 resize" case)
                          (resize (to-dtype gradient 'float64) (hash-ref case 'size)
                                  #:antialias? (hash-ref case 'antialias))
                          case 1e-12 'float64))
       (define photo-f32 (convert-image-dtype (hash-ref sources 'photo)))
       (for ([case (in-list (hash-ref j 'resized_crop_f32))])
         (define size (hash-ref case 'size))
         (check-transform (window-label "float32 resized-crop" case)
                          (resized-crop photo-f32 (hash-ref case 'window) (list size size)
                                        #:antialias? (hash-ref case 'antialias))
                          case 1e-6)))

     (bounded-test "P5: the half dtypes equal torchvision's through float32"
       (for ([case (in-list (hash-ref j 'resize_half))])
         (define half (string->symbol (hash-ref case 'dtype)))
         (check-bytes (case-label (format "~a resize" half) case)
                      (resize (to-dtype gradient half) (hash-ref case 'size)
                              #:antialias? (hash-ref case 'antialias))
                      case)))]))
