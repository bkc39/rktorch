#lang racket/base

;; Run: raco test torch/tests/audio-parity-test.rkt (inside `nix develop`;
;; SKIPS the torchaudio/soundfile batteries when those imports fail).

(module+ test
  ;; whole-module on purpose: the expansion needs bindings only-in would strip
  (require racket/runtime-path
           rackunit
           (only-in racket/file
                    delete-directory/files make-temporary-directory)
           (only-in "../audio/functional.rkt"
                    hann-window log-mel-spectrogram mel-filterbank stft)
           (only-in "../audio/librispeech.rkt" load-librispeech-fixture)
           (only-in "../audio/data.rkt"
                    load-audio load-audio-fixture load-wav save-audio
                    write-wav)
           (only-in "../main.rkt" ref tensor tensor-shape tensor->list)
           "private/python-env.rkt")

  (define-runtime-path wav-fixture "../audio/fixtures/sine-440-16k.wav")

  ;; the .#cuda python carries only torch-bin, so both batteries gate on
  ;; import
  (when (python-module-available? "torchaudio")
    (test-case "load-wav and load-audio against torchaudio.load"
      (define j (python-result "python/audio_parity.py"))
      (define-values (samples rate) (load-audio-fixture))
      (check-equal? (tensor-shape samples) (hash-ref j 'shape)
                    "torchaudio.load shape parity")
      (check-equal? rate (hash-ref j 'rate) "torchaudio.load rate parity")
      (check-equal? (for/list ([v (in-list (tensor->list samples))])
                      (exact->inexact v))
                    (hash-ref j 'values)
                    "torchaudio.load sample parity")
      (define-values (native native-rate) (load-audio wav-fixture))
      (check-equal? (tensor-shape native) (hash-ref j 'shape)
                    "load-audio shape parity")
      (check-equal? native-rate (hash-ref j 'rate) "load-audio rate parity")
      (check-equal? (for/list ([v (in-list (tensor->list native))])
                      (exact->inexact v))
                    (hash-ref j 'values)
                    "load-audio sample parity")))

  (when (python-module-available? "torchaudio")
    (test-case "spectral front-end against torch.stft and torchaudio fbanks"
      (define j (python-result "python/spectral_parity.py"))
      (define-values (samples rate _t) (load-librispeech-fixture))
      (define x (ref samples 0))
      (define frames
        (stft x #:n-fft 400 #:hop-length 160 #:window (hann-window 400)))
      (check-equal? (tensor-shape frames) (hash-ref j 'stft_shape))
      (for ([mine (in-list (tensor->list frames))]
            [theirs (in-list (hash-ref j 'stft_head))]
            [i (in-naturals)])
        (check-= mine theirs tol (format "stft element ~a" i)))
      (define fb (mel-filterbank #:n-freqs 201 #:n-mels 80
                                 #:sample-rate rate))
      (check-equal? (tensor-shape fb) (hash-ref j 'fbank_shape))
      (for ([mine (in-list (tensor->list fb))]
            [theirs (in-list (hash-ref j 'fbank))]
            [i (in-naturals)])
        (check-= mine theirs tol (format "fbank element ~a" i)))
      (define log-mel (log-mel-spectrogram x #:sample-rate rate))
      (check-equal? (tensor-shape log-mel) (hash-ref j 'log_mel_shape))
      (for ([mine (in-list (tensor->list log-mel))]
            [theirs (in-list (hash-ref j 'log_mel_head))]
            [i (in-naturals)]
            [_ (in-range 64)])
        (check-= mine theirs 1e-3 (format "log-mel element ~a" i)))))

  (when (python-module-available? "soundfile")
    (test-case "write-wav and save-audio against soundfile"
      (define dir (make-temporary-directory))
      (dynamic-wind
       void
       (lambda ()
         (define p (build-path dir "written.wav"))
         (write-wav p (tensor '((0.5 -0.25 0.125) (-1.0 0.0 0.75))) 22050)
         (define-values (mine rate) (load-wav p))
         (define j
           (call-with-python-env
            (lambda () (python-result "python/audio_write_parity.py"))
            #:env (list (cons "RKTORCH_WAV_UNDER_TEST" (path->string p)))))
         (check-equal? (tensor-shape mine) (hash-ref j 'shape)
                       "soundfile shape parity")
         (check-equal? rate (hash-ref j 'rate) "soundfile rate parity")
         (check-equal? (for/list ([v (in-list (tensor->list mine))])
                         (exact->inexact v))
                       (hash-ref j 'values)
                       "soundfile sample parity")
         (for ([name (in-list '("native.wav" "native.flac"))])
           (define np (build-path dir name))
           (save-audio np (tensor '((0.5 -0.25 0.125) (-1.0 0.0 0.75)))
                       22050)
           (define-values (nmine nrate) (load-audio np))
           (define nj
             (call-with-python-env
              (lambda () (python-result "python/audio_write_parity.py"))
              #:env (list (cons "RKTORCH_WAV_UNDER_TEST"
                                (path->string np)))))
           (check-equal? (tensor-shape nmine) (hash-ref nj 'shape) name)
           (check-equal? nrate (hash-ref nj 'rate) name)
           (check-equal? (for/list ([v (in-list (tensor->list nmine))])
                           (exact->inexact v))
                         (hash-ref nj 'values)
                         (format "save-audio ~a soundfile parity" name))))
       (lambda () (delete-directory/files dir))))))
