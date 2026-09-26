#lang racket/base

(require (only-in racket/contract/base listof)
         (only-in racket/file file->lines)
         ;; whole-module: define-runtime-path needs phase-1 bindings only-in
         ;; strips
         racket/runtime-path
         (only-in "../private/contract.rkt" define/contract-out))

(define-runtime-path classes-file "imagenet-classes.txt")

(define/contract-out imagenet-classes (listof string?) ;; noqa
  (file->lines classes-file))
