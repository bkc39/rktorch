#lang racket/base

(module+ test
  (require (only-in racket/file file->bytes)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../main.rkt" copy! dtype full-like with-no-grad)
           (only-in "../nn.rkt" BatchNorm1d Linear Sequential load-state! save-state! state-dict)
           (only-in "../nn/safetensors.rkt"
                    call-with-atomic-output-file read-safetensors safetensors-header
                    safetensors-metadata write-safetensors!)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "private/comparable.rkt" layer-bytes))

  (define-runtime-path master-file "master-save-state.safetensors")

  (define (fixture-model)
    (define net (Sequential (Linear 3 2) (BatchNorm1d 2)))
    (with-no-grad
      (for ([e (in-list (state-dict net))] [k (in-naturals 1)])
        (copy! (cdr e) (full-like (cdr e) (if (eq? (dtype (cdr e)) 'int64) k (* 1.5 k))))))
    net)

  (test-case "a file written by master's save-state! loads"
    (define net (Sequential (Linear 3 2) (BatchNorm1d 2)))
    (load-state! net master-file)
    (check-equal? (layer-bytes net) (layer-bytes (fixture-model))))

  (test-case "save-state! writes master's bytes, so master's load-state! reads them"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define path (build-path dir "m.safetensors"))
      (save-state! (fixture-model) path)
      (check-equal? (file->bytes path) (file->bytes master-file))))

  (test-case "__metadata__ is a string map, and load-state! reads past it"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define path (build-path dir "m.safetensors"))
      (define meta (hash "format" "rktorch-checkpoint" "version" "1"))
      (write-safetensors! 'test path (state-dict (fixture-model)) #:metadata meta)
      (define file (read-safetensors path))
      (check-equal? (safetensors-metadata file) meta)
      (check-false (hash-has-key? (safetensors-header file) '__metadata__))
      (check-false (safetensors-metadata (read-safetensors master-file)))
      (define net (Sequential (Linear 3 2) (BatchNorm1d 2)))
      (load-state! net path)
      (check-equal? (layer-bytes net) (layer-bytes (fixture-model)))))

  (define (directory-names dir)
    (sort (map path->string (directory-list dir)) string<?))

  (test-case "a write that raises leaves the target as it was and no temporary file"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define path (build-path dir "m.safetensors"))
      (save-state! (fixture-model) path)
      (check-exn #rx"half-way"
                 (lambda ()
                   (call-with-atomic-output-file
                    path
                    (lambda (out)
                      (write-bytes #"partial" out)
                      (error 'test "half-way")))))
      (check-equal? (file->bytes path) (file->bytes master-file))
      (check-equal? (directory-names dir) '("m.safetensors")))))
