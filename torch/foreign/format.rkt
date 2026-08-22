#lang racket/base

;; TODO: dtype suffixes for non-float32 non-empty tensors.

(require (only-in racket/format ~r)
         (only-in racket/list append-map drop take)
         (only-in racket/math infinite? nan?)
         (only-in racket/string string-join))

(provide tensor->pytorch-repr
         tensor-tree->pytorch-repr)

(define precision 4)
(define linewidth 80)
(define repr-opener "tensor(")

(define (finite-real? x)
  (and (rational? x) (not (nan? x))))

(define (fmt-special x)
  (cond
    [(nan? x) "nan"]
    [(and (infinite? x) (positive? x)) "inf"]
    [(infinite? x) "-inf"]
    [else #f]))

(define (fmt-fixed x)
  (or (fmt-special x) (~r x #:precision (list '= precision))))

(define (fmt-sci x)
  (or (fmt-special x)
      (~r x
          #:notation 'exponential
          #:precision (list '= precision)
          #:format-exponent
          (lambda (e)
            (format "e~a~a"
                    (if (negative? e) "-" "+")
                    (~r (abs e) #:min-width 2 #:pad-string "0"))))))

(define (fmt-int x)
  (or (fmt-special x) (~r x #:precision '(= 0))))

(define (all-integral? flat)
  (for/and ([x (in-list flat)] #:when (finite-real? x))
    (= x (round x))))

(define sci-ratio-bound 1000.0)
(define sci-upper-bound 1e8)
(define sci-lower-bound 1e-4)

(define (needs-sci-notation? flat)
  (define mags
    (for/list ([x (in-list flat)] #:when (and (finite-real? x) (not (zero? x))))
      (abs x)))
  (cond
    [(null? mags) #f]
    [else
     (define mx (apply max mags))
     (define mn (apply min mags))
     (or (> (/ mx mn) sci-ratio-bound)
         (> mx sci-upper-bound)
         (< mn sci-lower-bound))]))

(define (fmt-exact x)
  (number->string x))

(define (fmt-bool x)
  (if (zero? x) "False" "True"))

(define (every-value-max-width fmt flat)
  (for/fold ([w 0]) ([x (in-list flat)])
    (max w (string-length (fmt x)))))

(define (nonzero-finite-max-width fmt flat)
  (for/fold ([w 1]) ([x (in-list flat)]
                     #:when (and (finite-real? x) (not (zero? x))))
    (max w (string-length (fmt x)))))

(define (make-formatter flat mode)
  (define any-finite? (for/or ([x (in-list flat)]) (finite-real? x)))
  (define sci? (and (not mode) (needs-sci-notation? flat)))
  (define int-mode?
    (and (not mode) (not sci?) any-finite? (all-integral? flat)))
  (define fmt
    (cond
      [(eq? mode 'exact-integers) fmt-exact]
      [(eq? mode 'booleans) fmt-bool]
      [sci? fmt-sci]
      [int-mode? fmt-int]
      [else fmt-fixed]))
  (define width
    (if (memq mode '(exact-integers booleans))
        (every-value-max-width fmt flat)
        (nonzero-finite-max-width fmt flat)))
  (values fmt width))

(define (pad s width)
  (string-append (make-string (max 0 (- width (string-length s))) #\space) s))

(define (nest flat dims)
  (cond
    [(null? dims) (car flat)]
    [(null? (cdr dims)) (take flat (car dims))]
    [else
     (define block (apply * (cdr dims)))
     (for/list ([i (in-range (car dims))])
       (nest (take (drop flat (* i block)) block) (cdr dims)))]))

(define (format-nested node dims indent fmt max-width)
  (cond
    [(null? dims) (pad (fmt node) max-width)]
    [(null? (cdr dims))
     (define rendered
       (for/list ([v (in-list node)])
         (if (eq? v 'ellipsis) " ..." (pad (fmt v) max-width))))
     (define per-line
       (max 1 (quotient (- linewidth indent) (+ max-width 2))))
     (define lines
       (let loop ([xs rendered])
         (if (<= (length xs) per-line)
             (list (string-join xs ", "))
             (cons (string-join (take xs per-line) ", ")
                   (loop (drop xs per-line))))))
     (string-join lines
                  (string-append ",
" (make-string (add1 indent) #\space))
                  #:before-first "[" #:after-last "]")]
    [else
     (define sep
       (string-append "," (make-string (sub1 (length dims)) #\newline)
                      (make-string (add1 indent) #\space)))
     (string-join (for/list ([sub (in-list node)])
                    (if (eq? sub 'ellipsis)
                        "..."
                        (format-nested sub (cdr dims) (add1 indent) fmt
                                       max-width)))
                  sep #:before-first "[" #:after-last "]")]))

(define (tensor->pytorch-repr flat dims #:mode [mode #f])
  (define-values (fmt max-width) (make-formatter flat mode))
  (string-append repr-opener
                 (format-nested (nest flat dims) dims
                                (string-length repr-opener) fmt max-width)
                 ")"))

(define (tree-values node)
  (cond
    [(eq? node 'ellipsis) '()]
    [(list? node) (append-map tree-values node)]
    [else (list node)]))

(define (tensor-tree->pytorch-repr tree dims #:mode [mode #f])
  (define-values (fmt max-width) (make-formatter (tree-values tree) mode))
  (string-append repr-opener
                 (format-nested tree dims
                                (string-length repr-opener) fmt max-width)
                 ")"))
