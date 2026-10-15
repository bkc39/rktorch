#lang racket/base

(require (only-in racket/contract/base listof)
         (only-in racket/file file->lines)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../private/contract.rkt" define/contract-out))

(define-runtime-path classes-file "imagenet-classes.txt")

(define/contract-out imagenet-classes (listof string?) ;; noqa
  (file->lines classes-file))
