#lang racket/base

;; Reproduce PyTorch's tensor `repr` -- the form shown in the Python REPL --
;; from a flat, row-major list of values plus a shape.  (The `tensor([[...]])`
;; framing lives in Python's torch._tensor_str, not in libtorch, so we rebuild
;; it here rather than getting it from the C++ printer.)
;;
;; Faithful for the common case: CPU float32, fixed-point notation, the default
;; precision of 4, with PyTorch's per-tensor column alignment and the `tensor(`
;; continuation indent.  Integer-valued tensors print in int-mode (`2.`).  When
;; PyTorch would switch to scientific notation (very large/small magnitudes or a
;; wide dynamic range), the caller falls back to ATen's own printer instead
;; (see structs.rkt) rather than emitting a subtly-wrong repr.
;;
;; Not yet reproduced (v0 TODO): scientific notation, line wrapping of long
;; rows, large-tensor "..." summarization, and dtype suffixes for non-float32.

(require racket/format
         racket/list
         racket/math
         racket/string)

(provide tensor->pytorch-repr
         needs-sci-notation?)

(define precision 4)

(define (finite-real? x)
  (and (rational? x) (not (nan? x))))

;; PyTorch prints non-finite values as bare "inf"/"-inf"/"nan".
(define (fmt-special x)
  (cond
    [(nan? x) "nan"]
    [(and (infinite? x) (positive? x)) "inf"]
    [(infinite? x) "-inf"]
    [else #f]))

(define (fmt-fixed x)
  (or (fmt-special x) (~r x #:precision (list '= precision))))

(define (fmt-int x)
  (or (fmt-special x) (string-append (~r x #:precision '(= 0)) ".")))

(define (all-integral? flat)
  (for/and ([x (in-list flat)] #:when (finite-real? x))
    (= x (round x))))

;; PyTorch's heuristic for switching to scientific notation.
(define (needs-sci-notation? flat)
  (define mags
    (for/list ([x (in-list flat)] #:when (and (finite-real? x) (not (zero? x))))
      (abs x)))
  (cond
    [(null? mags) #f]
    [else
     (define mx (apply max mags))
     (define mn (apply min mags))
     (or (> (/ mx mn) 1000.0)
         (>= mx 1e8)
         (< mn 1e-4))]))

;; Returns (values format-proc max-element-width).  PyTorch picks one format for
;; the whole tensor, then right-justifies every element to a common width.
(define (make-formatter flat)
  (define any-finite? (for/or ([x (in-list flat)]) (finite-real? x)))
  (define fmt (if (and any-finite? (all-integral? flat)) fmt-int fmt-fixed))
  (define max-width
    (for/fold ([w 0]) ([x (in-list flat)])
      (max w (string-length (fmt x)))))
  (values fmt max-width))

(define (pad s width)
  (string-append (make-string (max 0 (- width (string-length s))) #\space) s))

;; Rebuild nested lists from the flat row-major data, per the shape.
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
     (string-join (for/list ([v (in-list node)]) (pad (fmt v) max-width))
                  ", " #:before-first "[" #:after-last "]")]
    [else
     ;; Separator between sub-blocks: a comma, (rank-1) newlines, then enough
     ;; spaces to align the next sub-block under this one.
     (define sep
       (string-append "," (make-string (sub1 (length dims)) #\newline)
                      (make-string (add1 indent) #\space)))
     (string-join (for/list ([sub (in-list node)])
                    (format-nested sub (cdr dims) (add1 indent) fmt max-width))
                  sep #:before-first "[" #:after-last "]")]))

;; "tensor(" + data + ")", with continuation lines aligned under the data by the
;; width of "tensor(" (7) -- exactly PyTorch's layout.
(define (tensor->pytorch-repr flat dims)
  (define-values (fmt max-width) (make-formatter flat))
  (string-append "tensor("
                 (format-nested (nest flat dims) dims 7 fmt max-width)
                 ")"))
