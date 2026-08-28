#lang racket/base

(module+ test
  ;; whole-module on purpose: the expansion needs bindings only-in would strip
  (require racket/runtime-path
           rackunit
           (only-in file/tar tar-gzip)
           (only-in racket/file delete-directory/files make-directory*)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "../audio/librispeech.rkt"
                    librispeech-utterances load-librispeech-fixture
                    load-utterance parse-trans-line utterance-id
                    utterance-path utterance-transcript)
           (only-in "../main.rkt" tensor-shape))

  (define-runtime-path fixture-flac
    "../audio/fixtures/librispeech-1272-128104-0000.flac")

  (test-case "trans.txt lines parse to id + transcript (#83)"
    (check-equal? (parse-trans-line "1272-128104-0000 MISTER QUILTER IS")
                  '("1272-128104-0000" . "MISTER QUILTER IS"))
    (check-equal? (parse-trans-line "  84-121123-0001  A  B  ")
                  '("84-121123-0001" . "A  B"))
    (check-false (parse-trans-line ""))
    (check-false (parse-trans-line "   "))
    (check-exn #rx"no transcript"
               (lambda () (parse-trans-line "1272-128104-0000"))))

  (test-case "the committed utterance fixture loads with its transcript (#83)"
    (define-values (samples rate transcript) (load-librispeech-fixture))
    (check-equal? rate 16000)
    (check-equal? (car (tensor-shape samples)) 1)
    (check-true (> (cadr (tensor-shape samples)) 16000)
                "at least a second of speech")
    (check-equal?
     transcript
     "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"))

  (test-case "extraction publishes atomically; corrupt archives fail clean (#83)"
    (with-temporary-directory (scratch)
      (parameterize ([current-environment-variables
                      (environment-variables-copy
                       (current-environment-variables))])
         (putenv "RKTORCH_AUDIO_DIR" (path->string scratch))
         (define build (build-path scratch "build"))
         (define chapter (build-path build "LibriSpeech" "dev-clean" "9" "9"))
         (make-directory* chapter)
         (copy-file fixture-flac (build-path chapter "9-9-0000.flac"))
         (call-with-output-file (build-path chapter "9-9.trans.txt")
           (lambda (out) (displayln "9-9-0000 HELLO WORLD" out)))
         (define cache-dir (build-path scratch "librispeech"))
         (make-directory* cache-dir)
         (parameterize ([current-directory build])
           (tar-gzip (build-path cache-dir "dev-clean.tar.gz")
                     "LibriSpeech"))
         (define us (librispeech-utterances "dev-clean"))
         (check-equal? (length us) 1)
         (check-equal? (utterance-id (car us)) "9-9-0000")
         (check-equal? (utterance-transcript (car us)) "HELLO WORLD")
         (define-values (samples rate) (load-utterance (car us)))
         (check-equal? rate 16000)
         (check-equal? (car (tensor-shape samples)) 1)
         (delete-directory/files (build-path cache-dir "dev-clean"))
         (call-with-output-file (build-path cache-dir "dev-clean.tar.gz")
           #:exists 'truncate
           (lambda (out)
             (write-bytes (bytes-append #"\037\213" (make-bytes 64 7)) out)))
         (check-exn exn:fail?
                    (lambda () (librispeech-utterances "dev-clean")))
         (check-false (directory-exists? (build-path cache-dir "dev-clean"))
                      "a corrupt archive must not publish a corpus")
         (check-false (file-exists?
                       (build-path cache-dir "dev-clean.tar.gz"))
                      "a corrupt archive is evicted")
         (define wrong (build-path scratch "wrong"))
         (make-directory* (build-path wrong "Nothing"))
         (parameterize ([current-directory wrong])
           (tar-gzip (build-path cache-dir "dev-clean.tar.gz") "Nothing"))
         (check-exn #rx"did not contain"
                    (lambda () (librispeech-utterances "dev-clean")))
         (check-false (file-exists?
                       (build-path cache-dir "dev-clean.tar.gz"))
                      "a wrong-tree archive is evicted")
         (check-false (directory-exists?
                       (build-path cache-dir "dev-clean")))
         (check-true
          (for/and ([p (in-list (directory-list cache-dir))])
            (not (regexp-match? #rx"librispeech-extract"
                                (path->string p))))
          "failed staging directories are cleaned up"))))

  (when (getenv "RKTORCH_LIBRISPEECH_LIVE")
    (test-case "dev-clean enumerates and loads (live)"
      (define us (librispeech-utterances "dev-clean"))
      (check-true (> (length us) 2000))
      (define u (car us))
      (check-regexp-match #rx"^[0-9]+-[0-9]+-[0-9]+$" (utterance-id u))
      (check-true (file-exists? (utterance-path u)))
      (check-true (positive? (string-length (utterance-transcript u))))
      (define-values (samples rate) (load-utterance u))
      (check-equal? rate 16000)
      (check-equal? (car (tensor-shape samples)) 1))))
