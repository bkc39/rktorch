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
;; Large tensors (numel > 1000) summarize exactly like PyTorch (#45):
;; callers hand this module a TREE with 'ellipsis markers where a
;; dimension was elided to its edge items; the last dimension renders the
;; marker as the literal " ..." joined by ", " (PyTorch's double-space),
;; higher dimensions as a "..." block between the standard separators.
;;
;; Not yet reproduced (v0 TODO): scientific notation and dtype suffixes
;; for non-float32 non-empty tensors.

(require (only-in racket/format ~r)
         (only-in racket/list append-map drop take)
         (only-in racket/math infinite? nan?)
         (only-in racket/string string-join))

(provide tensor->pytorch-repr
         tensor-tree->pytorch-repr
         tree-values
         needs-sci-notation?)

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
     (or (> (/ mx mn) 1000.0)
         (>= mx 1e8)
         (< mn 1e-4))]))

;; int64 tensors print bare integers ("1", never "1.") — exactly
;; torch.tensor([1, 2, 3])'s repr. The values arrive as exact integers
;; from the int64 copy-out, so number->string is already faithful.
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
  (define int-mode? (and (not mode) any-finite? (all-integral? flat)))
  (define fmt
    (cond
      [(eq? mode 'exact-integers) fmt-exact]
      [(eq? mode 'booleans) fmt-bool]
      [int-mode? fmt-int]
      [else fmt-fixed]))
  (define max-width
    (for/fold ([w 0]) ([x (in-list flat)])
      (max w (string-length (fmt x)))))
  ;; PyTorch's _Formatter.width() EXCLUDES the trailing "." in int-mode
  ;; ("0." reports width 1), so its line budget undercounts by one per
  ;; element and int-mode rows legitimately overflow 80 columns —
  ;; reproduced here or wide zero rows wrap where Python's don't.
  (define wrap-width (if int-mode? (sub1 max-width) max-width))
  (values fmt max-width wrap-width))

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

(define (format-nested node dims indent fmt max-width wrap-width)
  (cond
    [(null? dims) (pad (fmt node) max-width)]
    [(null? (cdr dims))
     ;; PyTorch wraps rows at a fixed floor((80 - indent) / (wrap-width
     ;; + 2)) elements per line (the ellipsis occupies one slot),
     ;; continuation lines indented one past the opening bracket;
     ;; wrap-width carries the int-mode dot quirk (see make-formatter).
     (define rendered
       (for/list ([v (in-list node)])
         (if (eq? v 'ellipsis) " ..." (pad (fmt v) max-width))))
     (define per-line
       (max 1 (quotient (- linewidth indent) (+ wrap-width 2))))
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
     ;; Separator between sub-blocks: a comma, (rank-1) newlines, then enough
     ;; spaces to align the next sub-block under this one.
     (define sep
       (string-append "," (make-string (sub1 (length dims)) #\newline)
                      (make-string (add1 indent) #\space)))
     (string-join (for/list ([sub (in-list node)])
                    (if (eq? sub 'ellipsis)
                        "..."
                        (format-nested sub (cdr dims) (add1 indent) fmt
                                       max-width wrap-width)))
                  sep #:before-first "[" #:after-last "]")]))

;; "tensor(" + data + ")", with continuation lines aligned under the data by the
;; width of "tensor(" (7) -- exactly PyTorch's layout.
(define (tensor->pytorch-repr flat dims #:mode [mode #f])
  (define-values (fmt max-width wrap-width) (make-formatter flat mode))
  (string-append "tensor("
                 (format-nested (nest flat dims) dims 7 fmt max-width
                                wrap-width)
                 ")"))

;; The values present in a summarized tree, in order ('ellipsis markers
;; skipped) — the population the formatter and the sci-notation heuristic
;; run over, exactly PyTorch's behavior of formatting from the SELECTED
;; elements.
(define (tree-values node)
  (cond
    [(eq? node 'ellipsis) '()]
    [(list? node) (append-map tree-values node)]
    [else (list node)]))

;; The summarized entry point: `tree` is nested per `dims`' structure but
;; with elided dimensions holding only edge items around an 'ellipsis
;; marker; `dims` supplies depth (for indent and last-dim detection), not
;; lengths.
(define (tensor-tree->pytorch-repr tree dims #:mode [mode #f])
  (define-values (fmt max-width wrap-width)
    (make-formatter (tree-values tree) mode))
  (string-append "tensor("
                 (format-nested tree dims 7 fmt max-width wrap-width)
                 ")"))
