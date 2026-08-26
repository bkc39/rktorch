#lang racket/base

;; whole-module on purpose: the expansion needs bindings only-in would strip
(require racket/runtime-path
         (only-in racket/file make-directory*)
         (only-in racket/math exact-round)
         (only-in racket/port copy-port)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in "../private/util.rkt" with-temporary-file)
         (only-in "../main.rkt" tensor tensor->list tensor-shape tensor?))

(provide load-wav
         write-wav
         load-audio-fixture
         download-audio-cached)

(define (read-exactly in n who)
  (define bs (read-bytes n in))
  (unless (and (bytes? bs) (= (bytes-length bs) n))
    (error who "truncated file: wanted ~a bytes, got ~a"
           n (if (bytes? bs) (bytes-length bs) 0)))
  bs)

(define (u16 bs offset) (integer-bytes->integer bs #f #f offset (+ offset 2)))
(define (u32 bs offset) (integer-bytes->integer bs #f #f offset (+ offset 4)))

;; Returns (values samples sample-rate): samples is a float32 tensor of
;; shape (channels n) in [-1, 1), torchaudio.load's convention.
(define (load-wav path)
  (call-with-input-file path
    (lambda (in)
      (define preamble (read-exactly in 12 'load-wav))
      (unless (and (equal? (subbytes preamble 0 4) #"RIFF")
                   (equal? (subbytes preamble 8 12) #"WAVE"))
        (error 'load-wav "~a is not a RIFF/WAVE file" path))
      (let loop ([fmt #f])
        (define header (read-bytes 8 in))
        (unless (and (bytes? header) (= (bytes-length header) 8))
          (error 'load-wav "~a has no data chunk" path))
        (define chunk-id (subbytes header 0 4))
        (define chunk-size (u32 header 4))
        (cond
          [(equal? chunk-id #"fmt ")
           (define payload (read-exactly in chunk-size 'load-wav))
           (unless (even? chunk-size)
             (read-exactly in 1 'load-wav))
           (loop payload)]
          [(equal? chunk-id #"data")
           (unless fmt
             (error 'load-wav "~a has a data chunk before fmt" path))
           (define audio-format (u16 fmt 0))
           (unless (= audio-format 1)
             (error 'load-wav
                    "~a: only PCM (format 1) is supported, got format ~a"
                    path audio-format))
           (define channels (u16 fmt 2))
           (define sample-rate (u32 fmt 4))
           (define bits (u16 fmt 14))
           (unless (= bits 16)
             (error 'load-wav "~a: only 16-bit PCM is supported, got ~a-bit"
                    path bits))
           (define data (read-exactly in chunk-size 'load-wav))
           (define n (quotient chunk-size (* 2 channels)))
           (values
            (tensor
             (for/list ([c (in-range channels)])
               (for/list ([i (in-range n)])
                 (/ (integer-bytes->integer
                     data #t #f
                     (* 2 (+ (* i channels) c))
                     (+ 2 (* 2 (+ (* i channels) c))))
                    32768.0))))
            sample-rate)]
          [else
           ;; chunk payloads are padded to even length
           (read-exactly in (+ chunk-size (modulo chunk-size 2)) 'load-wav)
           (loop fmt)])))))

(define (write-wav path samples sample-rate)
  (define shape (tensor-shape samples))
  (define-values (channels n)
    (case (length shape)
      [(1) (values 1 (car shape))]
      [(2) (values (car shape) (cadr shape))]
      [else (error 'write-wav
                   "samples must be rank 1 or (channels n), got shape ~a"
                   shape)]))
  (define flat (list->vector (tensor->list samples)))
  (define data (make-bytes (* 2 channels n)))
  (for* ([c (in-range channels)] [i (in-range n)])
    (integer->integer-bytes
     (max -32768 (min 32767 (exact-round (* (vector-ref flat (+ (* c n) i))
                                            32768))))
     2 #t #f data (* 2 (+ (* i channels) c))))
  (define (put-u32 out v) (write-bytes (integer->integer-bytes v 4 #f #f) out))
  (define (put-u16 out v) (write-bytes (integer->integer-bytes v 2 #f #f) out))
  (call-with-output-file path #:exists 'truncate
    (lambda (out)
      (write-bytes #"RIFF" out)
      (put-u32 out (+ 36 (bytes-length data)))
      (write-bytes #"WAVEfmt " out)
      (put-u32 out 16)
      (put-u16 out 1)
      (put-u16 out channels)
      (put-u32 out sample-rate)
      (put-u32 out (* sample-rate channels 2))
      (put-u16 out (* channels 2))
      (put-u16 out 16)
      (write-bytes #"data" out)
      (put-u32 out (bytes-length data))
      (write-bytes data out)))
  (void))

(define-runtime-path audio-fixture "fixtures/sine-440-16k.wav")

(define (load-audio-fixture)
  (load-wav audio-fixture))

(define (audio-cache-dir)
  (define override (getenv "RKTORCH_AUDIO_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "audio")))

;; Returns the cached path (audio archives are large; no in-memory read).
;; valid? gates the cache write so a rate-limit page or truncated body
;; cannot poison the cache.
(define (download-audio-cached name url #:valid? [valid? (lambda (path) #t)])
  (define dest (build-path (audio-cache-dir) name))
  (unless (file-exists? dest)
    (make-directory* (audio-cache-dir))
    (with-temporary-file (tmp #:template "audio-~a.part"
                              #:directory (audio-cache-dir))
      (call/input-url (string->url url)
                      (lambda (u) (get-pure-port u '() #:redirections 3))
                      (lambda (in)
                        (call-with-output-file tmp #:exists 'truncate
                          (lambda (out) (copy-port in out)))
                        (unless (valid? tmp)
                          (raise (exn:fail:network
                                  (format
                                   "download-audio-cached: fetched ~a failed validation; not caching (bad response from ~a?)"
                                   name url)
                                  (current-continuation-marks))))
                        (rename-file-or-directory tmp dest #t)))))
  dest)
