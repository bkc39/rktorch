#lang racket/base

;; whole-module on purpose: the expansion needs bindings only-in would strip
(require racket/runtime-path
         (only-in racket/file make-directory*)
         (only-in racket/math exact-round)
         (only-in racket/path normalize-path)
         (only-in racket/port copy-port)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in "../foreign/error.rkt" check-handle check-ok)
         (only-in "../foreign/raw/audio.rkt"
                  tr-audio-info/raw tr-audio-load/raw tr-audio-save/raw)
         (only-in "../foreign/structs.rkt" wrap-tensor)
         (only-in "../private/util.rkt" with-temporary-file)
         (only-in "../main.rkt" tensor tensor-shape tensor->list tensor?))

(provide audio-info
         load-audio
         load-wav
         save-audio
         write-wav
         load-audio-fixture
         download-audio-cached)

(define (audio-info path)
  (define-values (rc frames rate channels) (tr-audio-info/raw path))
  (check-ok rc 'audio-info)
  (values frames rate channels))

(define (load-audio path #:frame-offset [frame-offset 0]
                    #:num-frames [num-frames #f])
  (define samples
    (wrap-tensor
     (check-handle 'load-audio
                   (tr-audio-load/raw path frame-offset
                                      (or num-frames -1)))))
  (define-values (_frames rate _channels) (audio-info path))
  (values samples rate))

(define (save-audio path samples rate)
  (unless (tensor? samples)
    (error 'save-audio "samples must be a tensor: ~e" samples))
  (check-ok (tr-audio-save/raw path samples rate) 'save-audio)
  (void))

(define (read-exactly in n who)
  (define bs (read-bytes n in))
  (unless (and (bytes? bs) (= (bytes-length bs) n))
    (error who "truncated file: wanted ~a bytes, got ~a"
           n (if (bytes? bs) (bytes-length bs) 0)))
  bs)

(define (u16 bs offset) (integer-bytes->integer bs #f #f offset (+ offset 2)))
(define (u32 bs offset) (integer-bytes->integer bs #f #f offset (+ offset 4)))

(define (load-wav path)
  (call-with-input-file path
    (lambda (in)
      (define preamble (read-exactly in 12 'load-wav))
      (unless (and (equal? (subbytes preamble 0 4) #"RIFF")
                   (equal? (subbytes preamble 8 12) #"WAVE"))
        (error 'load-wav "~a is not a RIFF/WAVE file" path))
      (define riff-limit (+ 8 (u32 preamble 4)))
      ;; declared sizes bound every later allocation — cap them by the
      ;; physical file before believing them
      (when (> riff-limit (file-size path))
        (error 'load-wav
               "~a: declared RIFF size ~a exceeds the file's ~a bytes"
               path riff-limit (file-size path)))
      (let loop ([fmt #f])
        (when (> (+ (file-position in) 8) riff-limit)
          (error 'load-wav "~a has no data chunk" path))
        (define header (read-bytes 8 in))
        (unless (and (bytes? header) (= (bytes-length header) 8))
          (error 'load-wav "~a has no data chunk" path))
        (define chunk-id (subbytes header 0 4))
        (define chunk-size (u32 header 4))
        ;; the pad byte counts toward the boundary for every chunk we
        ;; consume past; a final unpadded data chunk stays tolerated
        (define padded
          (+ chunk-size
             (if (equal? chunk-id #"data") 0 (modulo chunk-size 2))))
        (when (> (+ (file-position in) padded) riff-limit)
          (error 'load-wav
                 "~a: ~a chunk extends past the declared RIFF size"
                 path chunk-id))
        (cond
          [(equal? chunk-id #"fmt ")
           (when (> chunk-size 1024)
             (error 'load-wav
                    "~a: fmt chunk is implausibly large (~a bytes)"
                    path chunk-size))
           (define payload (read-exactly in chunk-size 'load-wav))
           (unless (even? chunk-size)
             (read-exactly in 1 'load-wav))
           (loop payload)]
          [(equal? chunk-id #"data")
           (unless fmt
             (error 'load-wav "~a has a data chunk before fmt" path))
           (unless (>= (bytes-length fmt) 16)
             (error 'load-wav "~a: fmt chunk is too short (~a bytes)"
                    path (bytes-length fmt)))
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
           (when (zero? channels)
             (error 'load-wav "~a: fmt chunk declares 0 channels" path))
           (when (zero? sample-rate)
             (error 'load-wav "~a: fmt chunk declares a 0 sample rate"
                    path))
           (unless (= (u16 fmt 12) (* 2 channels))
             (error 'load-wav
                    "~a: block alignment ~a does not match ~a-channel PCM16"
                    path (u16 fmt 12) channels))
           (unless (= (u32 fmt 8) (* sample-rate channels 2))
             (error 'load-wav
                    "~a: byte rate ~a does not match rate ~a x ~a channels"
                    path (u32 fmt 8) sample-rate channels))
           (unless (zero? (remainder chunk-size (* 2 channels)))
             (error 'load-wav
                    "~a: ~a-byte data chunk is not whole ~a-channel frames"
                    path chunk-size channels))
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
           ;; chunk payloads are padded to even length; seek, don't read —
           ;; a forged size must not allocate its declared bytes
           (file-position in (+ (file-position in) chunk-size
                                (modulo chunk-size 2)))
           (loop fmt)])))))

(define (write-wav path samples sample-rate)
  (unless (exact-positive-integer? sample-rate)
    (error 'write-wav "sample rate must be a positive exact integer: ~e"
           sample-rate))
  (define shape (tensor-shape samples))
  (define-values (channels n)
    (case (length shape)
      [(1) (values 1 (car shape))]
      [(2) (values (car shape) (cadr shape))]
      [else (error 'write-wav
                   "samples must be rank 1 or (channels n), got shape ~a"
                   shape)]))
  (when (zero? channels)
    (error 'write-wav "zero channels: shape ~a" shape))
  (when (> (* channels 2) 65535)
    (error 'write-wav "~a channels do not fit the WAV header" channels))
  (when (> (+ 36 (* 2 channels n)) 4294967295)
    (error 'write-wav
           "~a frames of ~a channels overflow the RIFF size field"
           n channels))
  (when (>= (* sample-rate channels 2) (expt 2 32))
    (error 'write-wav
           "byte rate ~a does not fit the WAV header (rate ~a, ~a channels)"
           (* sample-rate channels 2) sample-rate channels))
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
      ;; complete, or the containment walk-up never reaches a root
      (path->complete-path (string->path override))
      (build-path (find-system-path 'cache-dir) "rktorch" "audio")))

(define (download-audio-cached name url #:valid? [valid? (lambda (path) #t)])
  (let ([p (string->path name)])
    (unless (and (relative-path? p)
                 (for/and ([elem (in-list (explode-path p))]) (path? elem)))
      (error 'download-audio-cached
             "cache name must stay inside the cache directory: ~e" name)))
  (define dest (build-path (audio-cache-dir) name))
  (make-directory* (audio-cache-dir))
  (define-values (parent _name _dir?) (split-path dest))
  ;; lexical checks miss symlinks: the deepest existing ancestor must
  ;; resolve inside the cache root before anything is created or served
  (define existing-ancestor
    (let loop ([p parent])
      (cond
        [(directory-exists? p) p]
        [else
         (define-values (up _ __) (split-path p))
         (loop up)])))
  (let loop ([c (explode-path (normalize-path existing-ancestor))]
             [r (explode-path (normalize-path (audio-cache-dir)))])
    (cond
      [(null? r) (void)]
      [(and (pair? c) (equal? (car c) (car r))) (loop (cdr c) (cdr r))]
      [else (error 'download-audio-cached
                   "cache name must stay inside the cache directory: ~e"
                   name)]))
  (when (link-exists? dest)
    (error 'download-audio-cached
           "cache name must stay inside the cache directory: ~e" name))
  (unless (file-exists? dest)
    (make-directory* parent)
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
