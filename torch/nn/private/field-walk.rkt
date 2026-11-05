#lang racket/base

(require (only-in racket/match match match-lambda)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "hash-keys.rkt" check-hashable hash-parts rekeyed!)
         (only-in "layer-containers.rkt" Children% Parameters% check-names classify)
         (only-in "layer-core.rkt"
                  layer? set-registry-buffers! set-registry-children! set-registry-params!)
         (only-in "memo.rkt" after-walk! allocated copy-of memo-who rebuilt-immutable))

(provide rebuilt-layer)

(define (rebuilt-layer seen self child tensor who names vals #:allocate allocate #:fill fill!)
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
  (define (immutable v parts make)
    (define (built)
      (define copies (map walk parts))
      (if (andmap eq? parts copies) v (apply make copies)))
    (rebuilt-immutable seen name v built))
  (define (walk v)
    (match v
      [(? layer?) (child name v)]
      [(? tensor?) (tensor v)]
      [(Parameters% alist)
       (Parameters% (map (match-lambda [(cons k t) (cons k (tensor t))]) alist))]
      [(Children% alist)
       (Children% (map (match-lambda [(cons k c) (cons k (child k c))]) alist))]
      [(mcons a d)
       (allocated seen v
                  (lambda () (mcons #f #f))
                  (lambda (c) (set-mcar! c (walk a)) (set-mcdr! c (walk d))))]
      [(cons a d) (immutable v (list a d) cons)]
      [(? immutable? (box x)) (immutable v (list x) box-immutable)]
      [(box x) (allocated seen v (lambda () (box #f)) (lambda (c) (set-box! c (walk x))))]
      [(? immutable? (? vector?)) (immutable v (vector->list v) vector-immutable)]
      [(? vector?)
       (allocated seen v
                  (lambda () (make-vector (vector-length v) #f))
                  (lambda (c)
                    (for ([x (in-vector v)] [i (in-naturals)])
                      (vector-set! c i (walk x)))))]
      [(? immutable? (? hash?)) (immutable v (hash-parts v) (lambda copies (rehashed seen v copies)))]
      [(? hash?)
       (allocated seen v (lambda () (hash-copy-clear v)) (lambda (c) (filled-hash! seen c v walk)))]
      [_ v]))
  (walk v))

(define (rehashed seen h copies)
  (when (hash-equal? h)
    (after-walk! seen (lambda () (check-keys seen h))))
  (hash-from (hash-copy-clear h) copies))

(define (hash-from h kxs)
  (match kxs
    ['() h]
    [(list* k x more) (hash-from (hash-set h k x) more)]))

(define (filled-hash! seen c v walk)
  (define weak? (or (hash-weak? v) (hash-ephemeron? v)))
  (define entries
    (for/list ([(k x) (in-hash v)])
      (define k2 (if weak? k (walk k)))
      (when (hash-equal? c)
        (check-hashable (memo-who seen) k2 c))
      (define x2 (walk x))
      (hash-set! c k2 x2)
      (cons k2 x2)))
  (when (and (hash-equal? c) (not weak?))
    (after-walk! seen (lambda () (rekeyed! (memo-who seen) c entries)))))

(define (check-keys seen h)
  (define copy (copy-of seen h h))
  (unless (for/and ([k (in-hash-keys h)])
           (eq? (hash-has-key? h k) (hash-has-key? copy (copy-of seen k k))))
    (raise-arguments-error
     (memo-who seen)
     (string-append "an immutable equal?-based hash was copied while one of its keys"
                    " was still being built, so the copy cannot find that key"))))
