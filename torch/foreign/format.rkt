#lang racket/base

;; Reproduce PyTorch's tensor `repr` from a flat, row-major list of values
;; plus a shape.  The `tensor([[...]])` framing lives in Python's
;; torch._tensor_str, not libtorch, so it is rebuilt here; fidelity rules
;; below are pinned from the _tensor_str/_Formatter source.
;;
;; Summarized tensors arrive as a TREE with 'ellipsis markers where a
;; dimension was elided to its edge items: the last dimension renders the
;; marker as " ..." joined by ", " (PyTorch's double-space), higher
;; dimensions as a "..." block between the standard separators.
;;
;; Not yet reproduced (TODO): dtype suffixes for non-float32 non-empty
;; tensors.

(require (only-in racket/format ~r)
         (only-in racket/list append-map drop take)
         (only-in racket/math infinite? nan?)
         (only-in racket/string string-join))

(provide tensor->pytorch-repr
         tensor-tree->pytorch-repr)

(define precision 4)
(define linewidth 80)

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

;; PyTorch's scientific form: '{:.4e}' — 4-decimal mantissa, sign on the
;; exponent, exponent zero-padded to at least two digits.
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

;; ~r with '(= 0) keeps the trailing decimal point, matching PyTorch's
;; int-mode "2." exactly.
(define (fmt-int x)
  (or (fmt-special x) (~r x #:precision '(= 0))))

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
     ;; strict bounds, exactly _Formatter's: 1e8 itself still prints
     ;; fixed (tensor([100000000.]))
     (or (> (/ mx mn) 1000.0)
         (> mx 1e8)
         (< mn 1e-4))]))

;; int64 tensors print bare integers ("1", never "1."); the values arrive
;; exact from the int64 copy-out, so number->string is already faithful.
(define (fmt-exact x)
  (number->string x))

;; bool tensors print True/False (values arrive as the float 0/1 mask
;; from the copy path).
(define (fmt-bool x)
  (if (zero? x) "False" "True"))

;; Returns (values format-proc max-element-width).  PyTorch picks one format for
;; the whole tensor, then right-justifies every element to a common width.
(define (make-formatter flat mode)
  (define any-finite? (for/or ([x (in-list flat)]) (finite-real? x)))
  ;; sci wins before int-mode: torch.tensor([1e10]) is integral AND
  ;; large, and prints 1.0000e+10
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
  ;; ONE width governs padding and the wrap budget (_Formatter's
  ;; max_width): floating tensors fold over the NONZERO FINITE values only
  ;; (default 1), so "0."/"nan" can exceed the width and go unpadded and
  ;; all-zero rows may overflow 80 columns; integer/bool fold over all.
  (define width
    (if (memq mode '(exact-integers booleans))
        (for/fold ([w 0]) ([x (in-list flat)])
          (max w (string-length (fmt x))))
        (for/fold ([w 1]) ([x (in-list flat)]
                           #:when (and (finite-real? x) (not (zero? x))))
          (max w (string-length (fmt x))))))
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
     ;; PyTorch wraps rows at floor((80 - indent) / (width + 2)) elements
     ;; per line (the ellipsis occupies one slot), continuation lines
     ;; indented one past the opening bracket.
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

;; "tensor(" + data + ")", with continuation lines aligned under the data by the
;; width of "tensor(" (7) -- exactly PyTorch's layout.
(define (tensor->pytorch-repr flat dims #:mode [mode #f])
  (define-values (fmt max-width) (make-formatter flat mode))
  (string-append "tensor("
                 (format-nested (nest flat dims) dims 7 fmt max-width)
                 ")"))

;; The population the formatter and sci heuristic run over — PyTorch
;; formats from the SELECTED elements, 'ellipsis markers skipped.
(define (tree-values node)
  (cond
    [(eq? node 'ellipsis) '()]
    [(list? node) (append-map tree-values node)]
    [else (list node)]))

;; Summarized entry point: `dims` supplies depth (indent and last-dim
;; detection), not lengths — elided dimensions hold only edge items.
(define (tensor-tree->pytorch-repr tree dims #:mode [mode #f])
  (define-values (fmt max-width) (make-formatter (tree-values tree) mode))
  (string-append "tensor("
                 (format-nested tree dims 7 fmt max-width)
                 ")"))
