#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/file
                    delete-directory/files file->bytes
                    make-temporary-directory)
           (only-in "../audio/data.rkt"
                    download-audio-cached load-audio-fixture load-wav
                    write-wav)
           (only-in "../main.rkt" tensor tensor-shape tensor->list zeros))

  (define (riff-chunk id payload)
    (bytes-append id
                  (integer->integer-bytes (bytes-length payload) 4 #f #f)
                  payload
                  (if (odd? (bytes-length payload)) (bytes 0) (bytes))))

  (define (riff-file . chunks)
    (define body (apply bytes-append #"WAVE" chunks))
    (bytes-append #"RIFF"
                  (integer->integer-bytes (bytes-length body) 4 #f #f)
                  body))

  (define (fmt-payload channels rate)
    (bytes-append (integer->integer-bytes 1 2 #f #f)
                  (integer->integer-bytes channels 2 #f #f)
                  (integer->integer-bytes rate 4 #f #f)
                  (integer->integer-bytes (* rate channels 2) 4 #f #f)
                  (integer->integer-bytes (* channels 2) 2 #f #f)
                  (integer->integer-bytes 16 2 #f #f)))

  (define (call-with-wav-bytes bs proc)
    (define dir (make-temporary-directory))
    (define p (build-path dir "case.wav"))
    (call-with-output-file p (lambda (out) (write-bytes bs out)))
    (dynamic-wind void
                  (lambda () (proc p))
                  (lambda () (delete-directory/files dir))))

  (test-case "wav fixture loads with pinned samples (#83)"
    (define-values (samples rate) (load-audio-fixture))
    (check-equal? rate 16000)
    (check-equal? (tensor-shape samples) '(1 1600))
    (check-equal? (map (lambda (s) (* s 32768.0))
                       (for/list ([v (in-list (tensor->list samples))]
                                  [_ (in-range 4)])
                         v))
                  '(0.0 2817.0 5550.0 8117.0)))

  (test-case "write-wav round-trips bit-exactly (#83)"
    (define dir (make-temporary-directory))
    (dynamic-wind
     void
     (lambda ()
       (define mono-path (build-path dir "mono.wav"))
       (define stereo-path (build-path dir "stereo.wav"))
       (define mono (tensor '(0.0 0.25 -0.25 0.5 -1.0 0.999969482421875)))
       (write-wav mono-path mono 8000)
       (define-values (mono* rate) (load-wav mono-path))
       (check-equal? rate 8000)
       (check-equal? (tensor-shape mono*) '(1 6))
       (check-equal? (tensor->list mono*) (tensor->list mono))
       (define stereo (tensor '((0.0 0.5 -0.5) (0.25 -0.25 0.75))))
       (write-wav stereo-path stereo 22050)
       (define-values (stereo* rate*) (load-wav stereo-path))
       (check-equal? rate* 22050)
       (check-equal? (tensor-shape stereo*) '(2 3))
       (check-equal? (tensor->list stereo*) (tensor->list stereo)))
     (lambda () (delete-directory/files dir))))

  (test-case "chunk skipping, padding, and malformed frames (#83)"
    (call-with-wav-bytes
     (riff-file (riff-chunk #"fmt " (fmt-payload 1 8000))
                (riff-chunk #"LIST" (bytes 1 2 3))
                (riff-chunk #"data"
                            (bytes-append
                             (integer->integer-bytes 100 2 #t #f)
                             (integer->integer-bytes -200 2 #t #f))))
     (lambda (p)
       (define-values (samples rate) (load-wav p))
       (check-equal? rate 8000)
       (check-equal? (tensor-shape samples) '(1 2))
       (check-equal? (map (lambda (s) (* s 32768.0))
                          (tensor->list samples))
                     '(100.0 -200.0))))
    (call-with-wav-bytes
     (riff-file (riff-chunk #"fmt " (fmt-payload 2 8000))
                (riff-chunk #"data" (make-bytes 6 0)))
     (lambda (p)
       (check-exn #rx"not whole 2-channel frames"
                  (lambda () (load-wav p)))))
    (call-with-wav-bytes
     (riff-file (riff-chunk #"fmt "
                            (bytes-append (fmt-payload 1 8000) (bytes 7)))
                (riff-chunk #"data" (integer->integer-bytes 300 2 #t #f)))
     (lambda (p)
       (define-values (samples rate) (load-wav p))
       (check-equal? (tensor-shape samples) '(1 1))
       (check-equal? (map (lambda (s) (* s 32768.0)) (tensor->list samples))
                     '(300.0))))
    (call-with-wav-bytes
     (riff-file (riff-chunk #"fmt " (fmt-payload 0 8000))
                (riff-chunk #"data" (bytes)))
     (lambda (p)
       (check-exn #rx"declares 0 channels" (lambda () (load-wav p)))))
    (call-with-wav-bytes
     (riff-file (riff-chunk #"fmt " (subbytes (fmt-payload 1 8000) 0 8))
                (riff-chunk #"data" (bytes)))
     (lambda (p)
       (check-exn #rx"too short" (lambda () (load-wav p)))))
    (call-with-wav-bytes
     (bytes-append
      (riff-file (riff-chunk #"fmt " (fmt-payload 1 8000)))
      (riff-chunk #"data" (integer->integer-bytes 300 2 #t #f)))
     (lambda (p)
       (check-exn #rx"no data chunk" (lambda () (load-wav p)))))
    (call-with-wav-bytes
     (bytes-append #"RIFF"
                   (integer->integer-bytes 4000000000 4 #f #f)
                   #"WAVE")
     (lambda (p)
       (check-exn #rx"exceeds the file" (lambda () (load-wav p)))))
    (call-with-wav-bytes
     (riff-file (riff-chunk #"fmt " (make-bytes 2000 0))
                (riff-chunk #"data" (bytes)))
     (lambda (p)
       (check-exn #rx"implausibly large" (lambda () (load-wav p)))))
    (call-with-wav-bytes
     (riff-file (riff-chunk #"fmt " (fmt-payload 1 0))
                (riff-chunk #"data" (bytes)))
     (lambda (p)
       (check-exn #rx"0 sample rate" (lambda () (load-wav p)))))
    (let ([odd-fmt (bytes-append
                    #"fmt " (integer->integer-bytes 17 4 #f #f)
                    (fmt-payload 1 8000) (bytes 9))])
      (call-with-wav-bytes
       (bytes-append #"RIFF"
                     (integer->integer-bytes (+ 4 (bytes-length odd-fmt))
                                             4 #f #f)
                     #"WAVE" odd-fmt
                     (riff-chunk #"data" (bytes)))
       (lambda (p)
         (check-exn #rx"extends past" (lambda () (load-wav p)))))))

  (test-case "wav rejection and cache paths (#83)"
    (check-exn #rx"not a RIFF/WAVE"
               (lambda ()
                 (define dir (make-temporary-directory))
                 (define p (build-path dir "bogus.wav"))
                 (call-with-output-file p
                   (lambda (out) (write-bytes (make-bytes 64 65) out)))
                 (dynamic-wind
                  void
                  (lambda () (load-wav p))
                  (lambda () (delete-directory/files dir)))))
    (check-exn #rx"rank 1 or"
               (lambda () (write-wav "unused.wav" (tensor 1.0) 8000)))
    (check-exn #rx"zero channels"
               (lambda () (write-wav "unused.wav" (zeros 0 3) 8000)))
    (check-exn #rx"sample rate"
               (lambda () (write-wav "unused.wav" (tensor '(0.0)) 0)))
    (check-exn #rx"sample rate"
               (lambda () (write-wav "unused.wav" (tensor '(0.0)) 8000.0)))
    (check-exn #rx"byte rate"
               (lambda () (write-wav "unused.wav" (zeros 3 1) (expt 2 30))))
    (check-exn #rx"do not fit the WAV header"
               (lambda () (write-wav "unused.wav" (zeros 32768 0) 1)))
    (define cache (make-temporary-directory))
    (dynamic-wind
     void
     (lambda ()
       (parameterize ([current-environment-variables
                       (environment-variables-copy
                        (current-environment-variables))])
         (putenv "RKTORCH_AUDIO_DIR" (path->string cache))
         (define src (build-path cache "src.wav"))
         (write-wav src (tensor '(0.0 0.5)) 16000)
         (define cached
           (download-audio-cached "cached.wav" (format "file://~a" src)))
         (check-equal? (file->bytes cached) (file->bytes src))
         (check-exn #rx"failed validation"
                    (lambda ()
                      (download-audio-cached "rejected.wav"
                                             (format "file://~a" src)
                                             #:valid? (lambda (p) #f))))
         (check-false (file-exists? (build-path cache "rejected.wav")))
         (define src-bytes (file->bytes src))
         (delete-file src)
         (check-equal? (file->bytes
                        (download-audio-cached "cached.wav"
                                               (format "file://~a" src)))
                       src-bytes
                       "second call must read the cache, not the source")
         (check-exn #rx"inside the cache directory"
                    (lambda ()
                      (download-audio-cached "../escape.wav"
                                             (format "file://~a" src))))
         (define nested-src (build-path cache "nested-src.wav"))
         (write-wav nested-src (tensor '(0.25)) 16000)
         (check-equal? (file->bytes
                        (download-audio-cached
                         "datasets/clip.wav"
                         (format "file://~a" nested-src)))
                       (file->bytes nested-src)
                       "nested cache names create their parent")))
     (lambda () (delete-directory/files cache)))))
