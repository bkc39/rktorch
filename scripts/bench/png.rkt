#lang racket/base

(require (only-in file/gzip deflate)
         (only-in racket/contract/base -> integer-in)
         (only-in "../../torch/private/contract.rkt" define/contract-out))

(define crc-table
  (for/vector #:length 256 ([n (in-range 256)])
    (for/fold ([c n]) ([_ (in-range 8)])
      (if (odd? c)
          (bitwise-xor #xEDB88320 (arithmetic-shift c -1))
          (arithmetic-shift c -1)))))

(define/contract-out (crc32 bs)
  (-> bytes? exact-nonnegative-integer?)
  (bitwise-xor #xFFFFFFFF
               (for/fold ([c #xFFFFFFFF]) ([b (in-bytes bs)])
                 (bitwise-xor (vector-ref crc-table (bitwise-and (bitwise-xor c b) #xFF))
                              (arithmetic-shift c -8)))))

(define/contract-out (adler32 bs)
  (-> bytes? exact-nonnegative-integer?)
  (define-values (a b)
    (for/fold ([a 1] [b 0]) ([x (in-bytes bs)])
      (define a* (modulo (+ a x) 65521))
      (values a* (modulo (+ b a*) 65521))))
  (bitwise-ior (arithmetic-shift b 16) a))

(define (u32 n) (integer->integer-bytes n 4 #f #t))

(define (chunk type data)
  (define typed (bytes-append type data))
  (bytes-append (u32 (bytes-length data)) typed (u32 (crc32 typed))))

(define (zlib bs)
  (define out (open-output-bytes))
  (deflate (open-input-bytes bs) out)
  (bytes-append #"\x78\x9c" (get-output-bytes out) (u32 (adler32 bs))))

(define (scanlines width height pixel)
  (define stride (add1 (* 3 width)))
  (define out (make-bytes (* height stride) 0))
  (for* ([y (in-range height)] [x (in-range width)] [c (in-range 3)])
    (bytes-set! out (+ (* y stride) 1 (* 3 x) c) (pixel x y c)))
  out)

(define/contract-out (rgb8-png width height pixel)
  (-> exact-positive-integer? exact-positive-integer?
      (-> exact-nonnegative-integer? exact-nonnegative-integer? (integer-in 0 2) byte?)
      bytes?)
  (bytes-append #"\x89PNG\r\n\x1a\n"
                (chunk #"IHDR" (bytes-append (u32 width) (u32 height) (bytes 8 2 0 0 0)))
                (chunk #"IDAT" (zlib (scanlines width height pixel)))
                (chunk #"IEND" #"")))

(module+ test
  (require rackunit
           (only-in torch dtype reshape shape tensor->list to-dtype ~>)
           (only-in torch/vision/image decode-image))

  (test-case "the checksums are PNG's CRC-32 and zlib's Adler-32"
    (check-equal? (crc32 #"IEND") #xAE426082)
    (check-equal? (adler32 #"Wikipedia") #x11E60398))

  (test-case "an image decodes back to its pixels, channel by channel"
    (define (pixel x y c) (modulo (+ (* 7 x) (* 13 y) (* 101 c)) 256))
    (define image (decode-image (rgb8-png 5 3 pixel) #:mode 'rgb))
    (check-equal? (shape image) '(3 3 5))
    (check-equal? (dtype image) 'uint8)
    (check-equal? (~> image (to-dtype 'int64) (reshape -1) tensor->list)
                  (for*/list ([c (in-range 3)] [y (in-range 3)] [x (in-range 5)])
                    (pixel x y c)))))
