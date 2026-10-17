#lang racket/base
(require (only-in racket/file make-temporary-file)
         rackunit
         (only-in torch shape tensor->list)
         (only-in torch/audio/data load-audio save-audio))

(define fixture (collection-file-path "sine-440-16k.flac" "torch/audio/fixtures"))
(define copy (make-temporary-file "round-trip-~a.flac"))
(define-values (samples rate) (load-audio fixture))
(save-audio copy samples rate)
(define-values (copied copied-rate) (load-audio copy))
(delete-file copy)
(printf "~a Hz ~a -> ~a Hz ~a\n" rate (shape samples) copied-rate (shape copied))
(check-equal? copied-rate rate)
(check-equal? (tensor->list copied) (tensor->list samples))
