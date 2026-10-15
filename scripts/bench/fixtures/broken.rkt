#lang racket/base

(module+ main
  (raise-user-error 'broken "fails before its first step"))
