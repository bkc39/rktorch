#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/contract/base -> any)
         (only-in racket/list append-map)
         ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt"
                  layer-mode layer-named-children layer-set-mode! layer?
                  mode/c registry? set-registry-mode! training?))

(provide in-mode
         in-eval-mode)

(define/contract-out (train! m) ;; noqa
  (-> layer? layer?)
  (layer-set-mode! m 'train)
  m)

(define/contract-out (eval! m) ;; noqa
  (-> layer? layer?)
  (layer-set-mode! m 'eval)
  m)

(define/contract-out (set-mode! m mode) ;; noqa
  (-> layer? mode/c layer?)
  (layer-set-mode! m mode)
  m)

(define/contract-out (layer-training? m) ;; noqa
  (-> layer? boolean?)
  (training? (layer-mode m)))

(define (mode-snapshot m)
  (define seen (make-hasheq))
  (let walk ([m m])
    (cond
      [(hash-ref seen m #f) '()]
      [else
       (hash-set! seen m #t)
       (cons (cons m (layer-mode m))
             (append-map (lambda (c) (walk (cdr c)))
                         (layer-named-children m)))])))

(define (restore-modes! before)
  (for ([e (in-list before)] #:unless (registry? (car e)))
    (layer-set-mode! (car e) (cdr e)))
  (for ([e (in-list before)] #:when (registry? (car e)))
    (set-registry-mode! (car e) (cdr e))))

(define/contract-out (call-with-mode m mode thunk) ;; noqa
  (-> layer? mode/c (-> any) any)
  (define before (mode-snapshot m))
  (dynamic-wind (lambda () (layer-set-mode! m mode))
                thunk
                (lambda () (restore-modes! before))))

(define/contract-out (call-with-eval-mode m thunk) ;; noqa
  (-> layer? (-> any) any)
  (call-with-mode m 'eval thunk))

(define-syntax-parse-rule (in-mode m:expr mode:expr body:expr ...+)
  (call-with-mode m mode (lambda () body ...)))

(define-syntax-parse-rule (in-eval-mode m:expr body:expr ...+)
  (call-with-mode m 'eval (lambda () body ...)))
