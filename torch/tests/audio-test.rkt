#lang racket/base

(module+ test
  (require rackunit
           racket/file
           (only-in "../data/audio.rkt"
                    download-audio-cached load-audio-fixture load-wav
                    write-wav)
           (only-in "../main.rkt" tensor tensor->list tensor-shape))

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
    (check-equal? (tensor->list stereo*) (tensor->list stereo))
    (delete-directory/files dir))

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
         (check-false (file-exists? (build-path cache "rejected.wav")))))
     (lambda () (delete-directory/files cache)))))
