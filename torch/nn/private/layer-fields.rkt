#lang racket/base

(require (only-in "../buffer.rkt" Buffer?)
         (only-in "../parameter.rkt" Parameter?)
         (only-in "layer-containers.rkt"
                  Children%? Parameters%? check-names classify)
         (only-in "layer-core.rkt" layer?))

(provide assemble
         check-entry-arity
         entry-procedure
         layer-methods-of
         layer-methods?
         no-forward
         prop:layer-methods
         write-field!)

(define-values (prop:layer-methods layer-methods? layer-methods-of)
  (make-struct-type-property 'layer-methods))

(define (registered? v)
  (or (Parameter? v) (Parameters%? v) (Buffer? v) (Children%? v) (layer? v)))

;; immutable containers found to reach nothing registered; their contents
;; cannot change, so neither can the verdict
(define clean (make-weak-hasheq))

;; the walk answers #t (reaches a registered value), 'clean (never can), or #f
;; (reaches none now, through a mutable container or a cycle)
(define (holds-registered? v)
  (define seen #f)
  (define (join acc x)
    (cond
      [(eq? acc #t) #t]
      [(eq? acc 'clean) (walk x)]
      [else (eq? (walk x) #t)]))
  (define (walk v)
    (cond
      [(registered? v) #t]
      [(not (or (pair? v) (vector? v) (box? v) (hash? v))) 'clean]
      [(hash-ref clean v #f) 'clean]
      [(and seen (hash-ref seen v #f)) #f]
      [else
       (set! seen (or seen (make-hasheq)))
       (hash-set! seen v #t)
       (settle v (walk-in v))]))
  (define (walk-in v)
    (cond
      [(pair? v) (join (walk (car v)) (cdr v))]
      [(vector? v)
       (for/fold ([acc 'clean]) ([x (in-vector v)] #:break (eq? acc #t))
         (join acc x))]
      [(box? v) (walk (unbox v))]
      [else
       (for/fold ([acc 'clean]) ([(k x) (in-hash v)] #:break (eq? acc #t))
         (join (join acc k) x))]))
  (define (settle v verdict)
    (cond
      [(not (eq? verdict 'clean)) verdict]
      [(or (pair? v) (immutable? v)) (hash-set! clean v #t) 'clean]
      [else #f]))
  (eq? (walk v) #t))

(define (assemble who make forward mode names vals)
  (define-values (params buffers children) (classify names vals))
  (check-names who (apply make forward params buffers children mode vals)))

(define (write-field! who self field read write v)
  (define problem
    (cond
      [(registered? (read self))
       "set! cannot replace a field that held a layer, parameter or buffer at construction"]
      [(holds-registered? v)
       "set! cannot put a layer, parameter or buffer in a field that held none at construction"]
      [else #f]))
  (when problem
    (raise-arguments-error who problem "field" field))
  (write self v))

(define (check-entry-arity who n rest? hidden inputs)
  (define given (length inputs))
  (unless (if rest? (>= given n) (= given n))
    (define expected (- n hidden))
    (apply raise-arity-error who (if rest? (arity-at-least expected) expected)
           (list-tail inputs (min hidden given)))))

(define (entry-procedure run)
  (make-keyword-procedure
   (lambda (kws kw-args self . inputs) (run self kws kw-args inputs))
   (lambda (self . inputs) (run self '() '() inputs))))

(define (no-forward who)
  (entry-procedure
   (lambda (self _kws _kw-args _inputs)
     (raise-arguments-error who "this layer has no forward, so it cannot be applied"
                            "layer" self))))
