#lang racket/base

(module+ test
  ;; whole-module on purpose: the expansion needs bindings only-in would strip
  (require racket/runtime-path
           rackunit
           (only-in racket/file
                    delete-directory/files file->bytes
                    make-temporary-directory)
           (only-in racket/path path-only)
           (only-in ffi/vector make-s32vector s32vector-ref)
           (only-in "../audio/data.rkt"
                    audio-info download-audio-cached load-audio
                    load-audio-fixture load-wav save-audio write-wav)
           (only-in "../foreign/raw/audio.rkt" tr-audio-load/raw)
           (only-in "../main.rkt" tensor tensor-shape tensor->list zeros))

  (define-runtime-path flac-fixture "../audio/fixtures/sine-440-16k.flac")

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
    (let ([bad-align (bytes-copy (fmt-payload 1 8000))])
      (integer->integer-bytes 4 2 #f #f bad-align 12)
      (call-with-wav-bytes
       (riff-file (riff-chunk #"fmt " bad-align)
                  (riff-chunk #"data" (make-bytes 4 0)))
       (lambda (p)
         (check-exn #rx"block alignment" (lambda () (load-wav p))))))
    (let ([bad-rate (bytes-copy (fmt-payload 1 8000))])
      (integer->integer-bytes 8000 4 #f #f bad-rate 8)
      (call-with-wav-bytes
       (riff-file (riff-chunk #"fmt " bad-rate)
                  (riff-chunk #"data" (make-bytes 4 0)))
       (lambda (p)
         (check-exn #rx"byte rate" (lambda () (load-wav p))))))
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

  (test-case "libsndfile facade: load-audio, audio-info, save-audio (#83)"
    (define-values (wav-samples wav-rate) (load-audio-fixture))
    (define wav-path
      (build-path (path-only flac-fixture) "sine-440-16k.wav"))
    (define-values (native-samples native-rate) (load-audio wav-path))
    (check-equal? native-rate wav-rate)
    (check-equal? (tensor->list native-samples) (tensor->list wav-samples)
                  "libsndfile and load-wav agree on the wav fixture")
    (define-values (flac-samples flac-rate) (load-audio flac-fixture))
    (check-equal? flac-rate 16000)
    (check-equal? (tensor-shape flac-samples) '(1 1600))
    (check-equal? (tensor->list flac-samples) (tensor->list wav-samples)
                  "flac-CLI-encoded fixture decodes to the pinned samples")
    (define-values (frames rate channels) (audio-info flac-fixture))
    (check-equal? (list frames rate channels) '(1600 16000 1))
    (define-values (windowed _wr)
      (load-audio wav-path #:frame-offset 100 #:num-frames 4))
    (check-equal? (tensor->list windowed)
                  (for/list ([v (in-list (tensor->list wav-samples))]
                             [i (in-naturals)]
                             #:when (and (>= i 100) (< i 104)))
                    v))
    (define dir (make-temporary-directory))
    (dynamic-wind
     void
     (lambda ()
       (define stereo (tensor '((0.5 -0.25 0.125) (-1.0 0.0 0.75))))
       (for ([name (in-list '("rt.wav" "rt.flac"))])
         (define p (build-path dir name))
         (save-audio p stereo 22050)
         (define-values (f r c) (audio-info p))
         (check-equal? (list f r c) '(3 22050 2) name)
         (define-values (back rate*) (load-audio p))
         (check-equal? rate* 22050)
         (check-equal? (tensor->list back) (tensor->list stereo) name))
       (define mono (tensor '(0.25 0.5)))
       (save-audio (build-path dir "mono.wav") mono 8000)
       (define-values (m _mr) (load-audio (build-path dir "mono.wav")))
       (check-equal? (tensor-shape m) '(1 2)
                     "rank-1 saves as a mono channel")
       (check-exn #rx"unsupported audio extension"
                  (lambda () (save-audio (build-path dir "x.mp3") mono 8000)))
       (check-exn #rx"sample-rate"
                  (lambda () (save-audio (build-path dir "x.wav") mono 0)))
       (check-exn #rx"sample-rate"
                  (lambda ()
                    (save-audio (build-path dir "x.wav") mono 8000.0)))
       (check-exn #rx"frame-offset"
                  (lambda ()
                    (load-audio (build-path dir "mono.wav")
                                #:frame-offset -1)))
       (check-exn #rx"num-frames"
                  (lambda ()
                    (load-audio (build-path dir "mono.wav")
                                #:num-frames -2)))
       (check-exn #rx"expected: tensor"
                  (lambda () (save-audio (build-path dir "x.wav") '(1.0) 8000)))
       (check-exn #rx"rank 1 or"
                  (lambda () (save-audio (build-path dir "x.wav")
                                         (tensor 0.5) 8000)))
       (check-exn #rx"rank 1 or"
                  (lambda () (save-audio (build-path dir "x.wav")
                                         (zeros 1 2 1) 8000)))
       (check-exn #rx"cannot open"
                  (lambda () (load-audio (build-path dir "missing.wav"))))
       (check-exn #rx"frame offset"
                  (lambda ()
                    (load-audio (build-path dir "mono.wav")
                                #:frame-offset 9)))
       (define written (make-s32vector 1 -7))
       (tr-audio-load/raw (build-path dir "mono.wav") 0 -1 written)
       (check-equal? (s32vector-ref written 0) 8000
                     "the rate out-slot is written through")
       (define untouched (make-s32vector 1 -7))
       (tr-audio-load/raw (build-path dir "mono.wav") 9 -1 untouched)
       (check-equal? (s32vector-ref untouched 0) -7
                     "a failed load leaves the out-slot untouched"))
     (lambda () (delete-directory/files dir))))

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
    (check-exn #rx"sample-rate"
               (lambda () (write-wav "unused.wav" (tensor '(0.0)) 0)))
    (check-exn #rx"sample-rate"
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
         (check-exn #rx"cache-name-inside-the-cache-directory"
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
                       "nested cache names create their parent")
         (define outside (make-temporary-directory))
         (make-file-or-directory-link outside (build-path cache "evil"))
         (check-exn #rx"inside the cache directory"
                    (lambda ()
                      (download-audio-cached
                       "evil/clip.wav"
                       (format "file://~a" nested-src))))
         (check-false (file-exists? (build-path outside "clip.wav")))
         (check-exn #rx"inside the cache directory"
                    (lambda ()
                      (download-audio-cached
                       "evil/new/clip.wav"
                       (format "file://~a" nested-src))))
         (check-false (directory-exists? (build-path outside "new")))
         (call-with-output-file (build-path outside "leak.wav")
           (lambda (out) (write-bytes (bytes 0) out)))
         (check-exn #rx"inside the cache directory"
                    (lambda ()
                      (download-audio-cached
                       "evil/leak.wav"
                       (format "file://~a" nested-src))))
         (make-file-or-directory-link (build-path outside "leak.wav")
                                      (build-path cache "direct.wav"))
         (check-exn #rx"inside the cache directory"
                    (lambda ()
                      (download-audio-cached
                       "direct.wav"
                       (format "file://~a" nested-src))))
         (parameterize ([current-directory cache])
           (putenv "RKTORCH_AUDIO_DIR" "relative-cache")
           (check-equal? (file->bytes
                          (download-audio-cached
                           "clip.wav" (format "file://~a" nested-src)))
                         (file->bytes nested-src)
                         "relative override completes against cwd"))
         (delete-directory/files outside)))
     (lambda () (delete-directory/files cache)))))
