#lang racket/base

(module+ test
  (require (only-in file/tar tar-gzip)
           (only-in racket/file make-directory*)
           (only-in racket/list flatten)
           rackunit
           (only-in "../audio/data.rkt" load-audio save-audio)
           (only-in "../audio/librispeech.rkt"
                    librispeech librispeech-fixture-utterance librispeech-root
                    librispeech-utterances utterance-id utterance-path)
           (only-in "../data/loader.rkt"
                    dataset-batch dataset-length dataset-ref default-collate)
           (only-in "../main.rkt" narrow shape tensor->list)
           (only-in "../private/util.rkt" with-temporary-directory)
           "private/python-env.rkt")

  (define cuts
    '(("19" "198" "0000" 4000 "THE FIRST CUT")
      ("19" "198" "0001" 6000 "THE SECOND CUT")
      ("84" "121" "0000" 5000 "A THIRD")))

  (define (build-tree! scratch)
    (define-values (samples rate)
      (load-audio (utterance-path (librispeech-fixture-utterance))))
    (define build (build-path scratch "build"))
    (for ([cut (in-list cuts)])
      (define-values (speaker chapter utt frames transcript) (apply values cut))
      (define dir (build-path build "LibriSpeech" "dev-clean" speaker chapter))
      (make-directory* dir)
      (define id (format "~a-~a-~a" speaker chapter utt))
      (save-audio (build-path dir (format "~a.flac" id)) (narrow samples 1 0 frames) rate)
      (call-with-output-file (build-path dir (format "~a-~a.trans.txt" speaker chapter))
        #:exists 'append
        (lambda (out) (fprintf out "~a ~a\n" id transcript))))
    (define cache-dir (build-path scratch "librispeech"))
    (make-directory* cache-dir)
    (parameterize ([current-directory build])
      (tar-gzip (build-path cache-dir "dev-clean.tar.gz") "LibriSpeech")))

  (define (values-of t)
    (map exact->inexact (flatten (tensor->list t))))

  (cond
    [(not (python-module-available? "torchaudio"))
     (printf "[librispeech-parity-test] skipped: python3 torchaudio ~a\n"
             "not available (run inside `nix develop`)")]
    [else
     (test-case "L18: librispeech = torchaudio LIBRISPEECH + pad_sequence"
       (with-temporary-directory (scratch)
         (parameterize ([current-environment-variables
                         (environment-variables-copy (current-environment-variables))])
           (putenv "RKTORCH_AUDIO_DIR" (path->string scratch))
           (build-tree! scratch)
           (define root (librispeech-root "dev-clean"))
           (define us (librispeech-utterances "dev-clean"))
           (define ds (librispeech us))
           (define j
             (call-with-python-env
              #:env (list (cons "RKTORCH_LIBRISPEECH_ROOT" (path->string root)))
              (lambda () (python-result "python/librispeech_dataset.py"))))
           (check-equal? (dataset-length ds) (hash-ref j 'length))
           (check-equal? (map utterance-id us) (hash-ref j 'ids))
           (define items
             (for/list ([i (in-range (dataset-length ds))])
               (call-with-values (lambda () (dataset-ref ds i)) list)))
           (check-equal? (map cadr items) (hash-ref j 'transcripts))
           (for ([it (in-list items)] [theirs (in-list (hash-ref j 'items))] [i (in-naturals)])
             (check-equal? (shape (car it)) (hash-ref theirs 'shape)
                           (format "item ~a shape" i))
             (check-equal? (values-of (car it)) (hash-ref theirs 'values)
                           (format "item ~a values" i)))
           (define-values (x lengths _transcripts)
             (dataset-batch ds '(0 1 2) default-collate))
           (check-equal? (shape x) (hash-ref j 'x_shape))
           (check-equal? (values-of x) (hash-ref j 'x_values))
           (check-equal? lengths (hash-ref j 'lengths)))))]))
