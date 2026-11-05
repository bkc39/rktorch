#lang racket/base

(require (only-in racket/match match match-lambda)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "layer-containers.rkt" Children% Parameters% check-names classify)
         (only-in "layer-core.rkt"
                  layer? set-registry-buffers! set-registry-children! set-registry-params!)
         (only-in "memo.rkt"
                  after-walk! allocated copy-of memo-who rebuilt-immutable rekey-after-walk!))

(provide rebuilt-layer)

(define (rebuilt-layer seen self child tensor who names vals allocate fill!)
  (define (filled! copy)
    (define copies
      (for/list ([name (in-list names)] [v (in-list vals)])
        (rebuilt-value seen name v child tensor)))
    (apply fill! copy copies)
    (define-values (params buffers children) (classify names copies))
    (set-registry-params! copy params)
    (set-registry-buffers! copy buffers)
    (set-registry-children! copy children)
    (after-walk! seen (lambda () (check-names who copy))))
  (allocated seen self allocate filled!))

(define (rebuilt-value seen name v child tensor)
  (let walk ([v v])
    (match v
      [(? layer?) (child name v)]
      [(? tensor?) (tensor v)]
      [(Parameters% alist)
       (Parameters% (map (match-lambda [(cons k t) (cons k (tensor t))]) alist))]
      [(Children% alist)
       (Children% (map (match-lambda [(cons k c) (cons k (child k c))]) alist))]
      [(? mutable-container?)
       (allocated seen v (lambda () (emptied v)) (lambda (c) (filled! seen c v walk)))]
      [(? immutable-container?)
       (rebuilt-immutable seen v (lambda () (reassembled seen v walk)))]
      [_ v])))

(define (mutable-container? v)
  (or (mpair? v)
      (and (or (box? v) (vector? v) (hash? v)) (not (immutable? v)))))

(define (immutable-container? v)
  (or (pair? v) (vector? v) (box? v) (hash? v)))

(define (emptied v)
  (match v
    [(? mpair?) (mcons #f #f)]
    [(? box?) (box #f)]
    [(? vector?) (make-vector (vector-length v) #f)]
    [(? hash?) (hash-copy-clear v)]))

(define (filled! seen c v walk)
  (match v
    [(mcons a d)
     (set-mcar! c (walk a))
     (set-mcdr! c (walk d))]
    [(box x) (set-box! c (walk x))]
    [(? vector?)
     (for ([x (in-vector v)] [i (in-naturals)])
       (vector-set! c i (walk x)))]
    [(? hash?)
     (define weak? (weakly-keyed? v))
     (for ([(k x) (in-hash v)])
       (hash-set! c (if weak? k (walk k)) (walk x)))
     (when (and (hash-equal? c) (not weak?))
       (rekey-after-walk! seen c))]))

(define (reassembled seen v walk)
  (match v
    [(cons a d) (cons (walk a) (walk d))]
    [(? vector?) (vector->immutable-vector (for/vector ([x (in-vector v)]) (walk x)))]
    [(box x) (box-immutable (walk x))]
    [(? hash?)
     (define entries
       (for/list ([(k x) (in-hash v)])
         (cons (walk k) (walk x))))
     (when (hash-equal? v)
       (after-walk! seen (lambda () (check-keys seen v))))
     (for/fold ([h (hash-copy-clear v)]) ([e (in-list entries)])
       (hash-set h (car e) (cdr e)))]))

(define (check-keys seen h)
  (define copy (copy-of seen h h))
  (for ([k (in-hash-keys h)])
    (unless (eq? (hash-has-key? h k) (hash-has-key? copy (copy-of seen k k)))
      (raise-arguments-error
       (memo-who seen)
       (string-append "an immutable equal?-based hash was copied while one of its keys"
                      " was still being built, so the copy cannot find that key")
       "key" k))))

(define (weakly-keyed? h)
  (or (hash-weak? h) (hash-ephemeron? h)))
