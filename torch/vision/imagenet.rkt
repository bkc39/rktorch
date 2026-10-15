#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/contract/base listof)
         (only-in racket/file file->lines)
         (only-in racket/runtime-path define-runtime-path)
         (only-in "../private/contract.rkt" define/contract-out))

(define-runtime-path classes-file "imagenet-classes.txt")

(define/contract-out imagenet-classes (listof string?) ;; noqa
  (file->lines classes-file))
