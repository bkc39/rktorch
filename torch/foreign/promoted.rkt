#lang racket/base

(require (only-in racket/base [abs base:abs] [cos base:cos] [sin base:sin])
         (only-in racket/contract/base
                  -> ->* ->i and/c listof none/c or/c unsupplied-arg? vectorof)
         (only-in racket/list
                  append* drop [flatten list-flatten] make-list
                  [take list-take])
         (prefix-in g: (only-in "../generated.rkt"
                                abs-tensor
                                broadcast-to
                                copy!
                                cos-tensor
                                eq-scalar eq-tensor
                                fill-scalar!
                                gather
                                ge-scalar ge-tensor
                                gt-scalar gt-tensor
                                index-add! index-copy!
                                index-fill-int-scalar!
                                index-fill-int-tensor!
                                index-select
                                le-scalar le-tensor
                                lt-scalar lt-tensor
                                masked-fill-scalar!
                                masked-fill-tensor!
                                masked-scatter!
                                masked-select
                                narrow
                                ne-scalar ne-tensor
                                nonzero
                                scatter-add! scatter-src! scatter-value!
                                select-int
                                sin-tensor
                                slice-tensor
                                take
                                take-along-dim
                                where-scalar
                                where-scalarother
                                where-scalarself
                                where-self))
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "contracts.rkt"
                  bool-tensor/c compare/c flatten/c index-spec/c index-vector/c
                  index/c int64-tensor/c unary-numeric/c unary-real/c)
         (only-in "device-type.rkt" device-type)
         (only-in "ops.rkt"
                  item tensor-device tensor-dtype tensor-shape tensor->list
                  to-device to-dtype)
         (only-in "slice.rkt" :: slice-end slice-start slice-step slice?)
         (only-in "structs.rkt" tensor?)
         (only-in "tensor-ops.rkt" add mul reshape sum tensor unsqueeze))

(define/contract-out narrow ;; noqa
  (-> tensor? index/c index/c exact-positive-integer? tensor?)
  g:narrow)
(define/contract-out select (-> tensor? index/c index/c tensor?) g:select-int) ;; noqa
(define/contract-out index-select ;; noqa
  (-> tensor? index/c index-vector/c tensor?)
  g:index-select)
(define/contract-out masked-select ;; noqa
  (-> tensor? bool-tensor/c tensor?)
  g:masked-select)
(define/contract-out nonzero (-> tensor? tensor?) g:nonzero) ;; noqa

(define/contract-out (abs x) unary-real/c ;; noqa
  (if (tensor? x) (g:abs-tensor x) (base:abs x)))
(define/contract-out (sin x) unary-numeric/c ;; noqa
  (if (tensor? x) (g:sin-tensor x) (base:sin x)))
(define/contract-out (cos x) unary-numeric/c ;; noqa
  (if (tensor? x) (g:cos-tensor x) (base:cos x)))

(define (bool-mask? s)
  (and (tensor? s) (eq? (tensor-dtype s) 'bool)))

(define (spec-consumed-dims s)
  (cond
    [(or (eq? s '...) (eq? s #f)) 0]
    [(bool-mask? s) (length (tensor-shape s))]
    [else 1]))

(define (expand-ellipsis specs rank who)
  (define consuming
    (for/sum ([s (in-list specs)]) (spec-consumed-dims s)))
  (unless (<= consuming rank)
    (error who "too many indices for a ~a-d tensor: ~e" rank specs))
  (define fills (- rank consuming))
  (define ellipses (for/sum ([s (in-list specs)]) (if (eq? s '...) 1 0)))
  (case ellipses
    [(0) specs]
    [(1) (append* (for/list ([s (in-list specs)])
                    (if (eq? s '...) (build-list fills (lambda (_) (::)))
                        (list s))))]
    [else (error who "at most one '... allowed: ~e" specs)]))

;; Deviation from numpy advanced indexing: multiple index tensors
;; apply per-dim sequentially, not broadcast-combined, and index
;; tensors are rank-1 (contract).
(define (index-tensor positions v)
  (tensor positions #:dtype 'int64 #:device (tensor-device v)))

(define (mask-spec->index-tensor m)
  (reshape (g:nonzero (reshape m -1)) -1))

(define (wrap-negative-positions s v d)
  (define n (list-ref (tensor-shape v) d))
  (cond
    [(not (tensor? s))
     (define positions (if (vector? s) (vector->list s) s))
     (index-tensor (for/list ([i (in-list positions)])
                     (if (< i 0) (+ i n) i))
                   v)]
    [else
     ;; python indexing accepts a CPU index for a CUDA tensor;
     ;; index_select does not, so align devices first. Adding n*(s<0)
     ;; wraps negatives and is the identity elsewhere — no host sync.
     (define s* (to-device s (tensor-device v)))
     (add s* (mul (to-dtype (g:lt-scalar s* 0.0) 'int64)
                  (tensor n #:dtype 'int64
                          #:device (tensor-device v))))]))

(define (apply-mask-spec v d m0)
  (define m (to-device m0 (tensor-device v)))
  (define vdims (tensor-shape v))
  (define mdims (tensor-shape m))
  (define n (length mdims))
  (unless (and (<= (+ d n) (length vdims))
               (equal? mdims (list-take (list-tail vdims d) n)))
    (error 'tensor-ref "mask shape ~a does not match dims ~a at dim ~a"
           mdims vdims d))
  (define collapsed
    (if (= n 1)
        v
        (apply reshape v (append (list-take vdims d)
                                 (list (apply * mdims))
                                 (list-tail vdims (+ d n))))))
  (g:index-select collapsed d (mask-spec->index-tensor m)))

(define/checked-out (tensor-ref t . specs)
  (-> tensor? index-spec/c ... (or/c tensor? number? boolean?))
  (cond
    [(and (pair? specs) (null? (cdr specs))
          (bool-mask? (car specs))
          (equal? (tensor-shape (car specs)) (tensor-shape t)))
     (g:masked-select t (to-device (car specs) (tensor-device t)))]
    [else
     (define expanded
       (expand-ellipsis specs (length (tensor-shape t)) 'tensor-ref))
     (define result
       (for/fold ([v t] [d 0] #:result v) ([s (in-list expanded)])
         (cond
           [(exact-integer? s) (values (g:select-int v d s) d)]
           [(slice? s)
            (values (g:slice-tensor v d (slice-start s) (slice-end s)
                                    (slice-step s))
                    (add1 d))]
           [(eq? s #f) (values (unsqueeze v d) (add1 d))]
           [(bool-mask? s) (values (apply-mask-spec v d s) (add1 d))]
           [(or (tensor? s) (list? s) (vector? s))
            (values (g:index-select v d (wrap-negative-positions s v d))
                    (add1 d))]
           [else (error 'tensor-ref "not an index spec: ~e" s)])))
     (cond
       [(pair? (tensor-shape result)) result]
       [(eq? (tensor-dtype result) 'bool) (not (zero? (item result)))]
       [else (item result)])]))

(define/contract-out (take v n)
  (->i ([v (or/c tensor? list?)]
        [n (v) (if (tensor? v)
                   (or/c int64-tensor/c
                         (listof exact-integer?)
                         (vectorof exact-integer?))
                   exact-nonnegative-integer?)])
       [result (v) (if (tensor? v) tensor? list?)])
  (cond
    [(not (tensor? v)) (list-take v n)]
    [else
     (define idx (if (tensor? n) n (index-tensor n v)))
     (cond
       [(eq? (device-type (tensor-device v)) 'mps)
        ;; the vendored schema registers at::take for CPU/CUDA only;
        ;; the device check pre-empts the wrap's silent to-device
        (unless (equal? (tensor-device idx) (tensor-device v))
          (error 'take
                 "index tensor must be on the same device as the input"))
        ;; explicit sizes: -1 is ambiguous for zero-element reshapes
        (define flat (reshape v (apply * (tensor-shape v))))
        (define idx-flat (reshape idx (apply * (tensor-shape idx))))
        (define flat-out
          (g:index-select
           flat 0 (wrap-negative-positions idx-flat flat 0)))
        (apply reshape flat-out (tensor-shape idx))]
       [else (g:take v idx)])]))

(define/contract-out (gather t dim index) ;; noqa
  (->i ([t tensor?]
        [dim index/c]
        [index (t dim)
               (and/c int64-tensor/c
                      (lambda (x)
                        (define td (tensor-shape t))
                        (define xd (tensor-shape x))
                        (define rank (length td))
                        (define g (if (negative? dim) (+ dim rank) dim))
                        (and (= (length xd) rank)
                             ;; ATen skips extent checks for empty
                             ;; indices (verified)
                             (or (zero? (apply * xd))
                                 (for/and ([xi (in-list xd)]
                                           [ti (in-list td)]
                                           [i (in-naturals)])
                                   (or (= i g) (<= xi ti)))))))])
       [result tensor?])
  (g:gather t dim index #f))

(define/contract-out (take-along-dim t indices [dim #f]) ;; noqa
  (->i ([t tensor?]
        [indices (t dim)
                 (and/c int64-tensor/c
                        (lambda (x)
                          (or (unsupplied-arg? dim)
                              (not dim)
                              (= (length (tensor-shape x))
                                 (length (tensor-shape t))))))])
       ([dim (or/c #f index/c)])
       [result tensor?])
  (g:take-along-dim t indices dim))

(define/contract-out where
  (->i ([c bool-tensor/c])
       ([a (or/c tensor? real?)]
        [b (a) (if (unsupplied-arg? a) none/c (or/c tensor? real?))])
       #:pre/name (a b) "a condition alone, or condition + both arms"
       (eq? (unsupplied-arg? a) (unsupplied-arg? b))
       [result (a) (if (unsupplied-arg? a) (listof tensor?) tensor?)])
  (case-lambda
    [(mask)
     (cond
       [(null? (tensor-shape mask))
        (list (tensor (if (zero? (car (tensor->list mask))) '() '(0))
                      #:dtype 'int64 #:device (tensor-device mask)))]
       [else
        (define coords (g:nonzero mask))
        (for/list ([d (in-range (length (tensor-shape mask)))])
          (g:select-int coords 1 d))])]
    [(mask a) (error 'where "expected both value arms, got one: ~e" a)]
    [(mask a b)
     (cond
       [(and (tensor? a) (tensor? b)) (g:where-self mask a b)]
       [(tensor? a)
        (cond
          [(and (exact-integer? b) (memq (tensor-dtype a) '(bool int64)))
           ;; a double scalar would float-promote integral results and
           ;; round past 2^53 — same guard as the comparison combinator
           (g:where-self mask a (tensor b #:device (tensor-device a)))]
          [else (g:where-scalarother mask a (exact->inexact b))])]
       [(and (tensor? b) (exact-integer? a)
             (memq (tensor-dtype b) '(bool int64)))
        (g:where-self mask (tensor a #:device (tensor-device b)) b)]
       [(tensor? b) (g:where-scalarself mask (exact->inexact a) b)]
       [(and (exact-integer? a) (exact-integer? b))
        (define dev (tensor-device mask))
        (g:where-self mask (tensor a #:device dev) (tensor b #:device dev))]
       [else (g:where-scalar mask (exact->inexact a) (exact->inexact b))])]))


(define (double-roundtrips? v)
  (define d (exact->inexact v))
  (and (rational? d) (= (inexact->exact d) v)))

(define (int64-dtype? t)
  (eq? (tensor-dtype t) 'int64))

(define (scalar->value-tensor v t)
  (tensor v #:dtype 'int64 #:device (tensor-device t)))

(define/contract-out (index-copy! t dim index source) ;; noqa
  (-> tensor? index/c index-vector/c tensor? void?)
  (void (g:index-copy! t dim index source)))

(define/contract-out (index-add! t dim index source #:alpha [alpha 1]) ;; noqa
  (->* [tensor? index/c index-vector/c tensor?] [#:alpha real?] void?)
  (void
   (if (and (exact-integer? alpha) (int64-dtype? t)
            (not (double-roundtrips? alpha)))
       (g:index-add! t dim index (mul source (scalar->value-tensor alpha t))
                     1.0)
       (g:index-add! t dim index source (exact->inexact alpha)))))

(define/contract-out (index-fill! t dim index v) ;; noqa
  (-> tensor? index/c index-vector/c real? void?)
  (void
   (if (and (exact-integer? v) (int64-dtype? t)
            (not (double-roundtrips? v)))
       (g:index-fill-int-tensor! t dim index (scalar->value-tensor v t))
       (g:index-fill-int-scalar! t dim index (exact->inexact v)))))

(define (same-rank-index/c t)
  (and/c int64-tensor/c
         (lambda (x)
           (= (length (tensor-shape x)) (length (tensor-shape t))))))

(define/contract-out (scatter! t dim index v) ;; noqa
  (->i ([t tensor?]
        [dim index/c]
        [index (t) (same-rank-index/c t)]
        [v (or/c tensor? real?)])
       [result void?])
  (void
   (cond
     [(tensor? v) (g:scatter-src! t dim index v)]
     [(and (exact-integer? v) (int64-dtype? t)
           (not (double-roundtrips? v)))
      ;; broadcast v to the index shape without a double transit
      (g:scatter-src! t dim index
                      (add (mul index (scalar->value-tensor 0 t))
                           (scalar->value-tensor v t)))]
     [else (g:scatter-value! t dim index (exact->inexact v))])))

(define/contract-out (scatter-add! t dim index src) ;; noqa
  (->i ([t tensor?]
        [dim index/c]
        [index (t) (same-rank-index/c t)]
        [src tensor?])
       [result void?])
  (void (g:scatter-add! t dim index src)))

(define/contract-out (masked-fill! t mask v) ;; noqa
  (-> tensor? bool-tensor/c real? void?)
  (void
   (if (and (exact-integer? v) (int64-dtype? t)
            (not (double-roundtrips? v)))
       (g:masked-fill-tensor! t mask (scalar->value-tensor v t))
       (g:masked-fill-scalar! t mask (exact->inexact v)))))

(define/contract-out (masked-scatter! t mask source) ;; noqa
  (-> tensor? bool-tensor/c tensor? void?)
  (void (g:masked-scatter! t mask source)))

(define (write-mask-target! t mask0 v)
  (define moved (to-device mask0 (tensor-device t)))
  (define mdims (tensor-shape moved))
  (define tdims (tensor-shape t))
  (unless (and (<= (length mdims) (length tdims))
               (equal? mdims (list-take tdims (length mdims))))
    (error 'tensor-ref! "mask shape ~a does not match leading dims of ~a"
           mdims tdims))
  ;; ATen broadcasts masks right-aligned; a leading-dims mask needs
  ;; trailing width-1 dims or it lands on the wrong axes
  (define trailing (list-tail tdims (length mdims)))
  (define mask
    (if (null? trailing)
        moved
        (apply reshape moved
               (append mdims (make-list (length trailing) 1)))))
  (cond
    [(not (tensor? v))
     (if (and (exact-integer? v) (int64-dtype? t)
              (not (double-roundtrips? v)))
         (g:masked-fill-tensor! t mask (scalar->value-tensor v t))
         (g:masked-fill-scalar! t mask (exact->inexact v)))]
    [(null? (tensor-shape v))
     ;; python broadcasts a 0-d source; masked_scatter_ would not
     (g:masked-fill-tensor! t mask v)]
    [else
     ;; python broadcasts the source to (nnz . trailing); validation is
     ;; required because masked_scatter_ silently ignores surplus
     (define sel-shape
       (cons (item (sum (to-dtype mask 'int64))) trailing))
     ;; python strips leading singleton source dims before broadcasting
     (define vdims
       (let loop ([d (tensor-shape v)])
         (if (and (> (length d) (length sel-shape)) (= (car d) 1))
             (loop (cdr d))
             d)))
     (unless (and (<= (length vdims) (length sel-shape))
                  (for/and ([f (in-list (reverse vdims))]
                            [s (in-list (reverse sel-shape))])
                    (or (= f 1) (= f s))))
       (error 'tensor-ref!
              "source shape ~a cannot broadcast to the true mask positions ~a"
              (tensor-shape v) sel-shape))
     (g:masked-scatter!
      t mask
      (if (= (apply * vdims) (apply * sel-shape))
          v
          (g:broadcast-to (apply reshape v vdims) sel-shape)))]))

(define/checked-out (tensor-ref! t v . specs)
  (-> tensor? (or/c tensor? real?) index-spec/c ... void?)
  (void
   (cond
    [(and (pair? specs) (null? (cdr specs)) (bool-mask? (car specs)))
     (write-mask-target! t (car specs) v)]
    [(for/or ([s (in-list specs)])
       (or (tensor? s) (list? s) (vector? s)))
     (error 'tensor-ref!
            "index-vector write targets need index_put_ (issue #67): ~e"
            specs)]
    [else
     ;; ATen clamps slices where select would raise, hence the bounds
     ;; check before the width-1 rebuild
     (define rank (length (tensor-shape t)))
     (define expanded (expand-ellipsis specs rank 'tensor-ref!))
     (define dropped
       (for/sum ([s (in-list expanded)])
         (if (exact-integer? s) 1 0)))
     (define added
       (for/sum ([s (in-list expanded)]) (if (eq? s #f) 1 0)))
     (define result-rank (+ (- rank dropped) added))
     (define view
       (cond
         [(positive? result-rank) (apply tensor-ref t expanded)]
         [(zero? rank) (reshape t 1)]
         [else
          (for ([s (in-list expanded)]
                [n (in-list (tensor-shape t))])
            (unless (and (exact-integer? s) (< (- (add1 n)) s n))
              (error 'tensor-ref!
                     "index ~a out of range for dimension of size ~a"
                     s n)))
          (apply tensor-ref t
                 (for/list ([s (in-list expanded)])
                   (if (exact-integer? s)
                       (if (= s -1) (:: s #f) (:: s (add1 s)))
                       s)))]))
     (cond
       [(tensor? v)
        ;; python strips leading singleton source dims; copy_ does not
        (define vshape (tensor-shape view))
        (define vdims
          (let loop ([d (tensor-shape v)])
            (if (and (> (length d) (length vshape)) (= (car d) 1))
                (loop (cdr d))
                d)))
        (g:copy! view
                 (if (equal? vdims (tensor-shape v))
                     v
                     (apply reshape v vdims))
                 #f)]
       [(and (exact-integer? v) (int64-dtype? t)
             (not (double-roundtrips? v)))
        (g:copy! view (scalar->value-tensor v t) #f)]
       ;; fill_'s C double keeps full float precision (a float32
       ;; ingestion transit would round float64 destinations)
       [else (g:fill-scalar! view (exact->inexact v))])])))

(define/contract-out (flatten v [start-dim 0] [end-dim -1])
  flatten/c
  (cond
    [(tensor? v)
     (define shp (tensor-shape v))
     (define n (length shp))
     (cond
       [(zero? n)
        (unless (and (memv start-dim '(0 -1)) (memv end-dim '(0 -1)))
          (error 'flatten
                 "invalid dim range [~a, ~a] for a 0-d tensor"
                 start-dim end-dim))
        (reshape v 1)]
       [else
        (define s (if (negative? start-dim) (+ n start-dim) start-dim))
        (define e (if (negative? end-dim) (+ n end-dim) end-dim))
        (unless (and (<= 0 s e) (< e n))
          (error 'flatten
                 "invalid dim range [~a, ~a] for a ~a-d tensor"
                 start-dim end-dim n))
        ;; not for/product — that would raise the racket/base version floor
        (define collapsed
          (apply * (for/list ([d (in-list shp)] [i (in-naturals)]
                                                 #:when (<= s i e))
                     d)))
        (apply reshape v (append (list-take shp s)
                                 (list collapsed)
                                 (drop shp (add1 e))))])]
    [else (list-flatten v)]))

;; --------------------------------------------------- comparison dispatchers

(define ((comparison t-op s-op) a b)
  (cond
    [(tensor? b) (t-op a b)]
    [(and (exact-integer? b) (eq? (tensor-dtype a) 'int64))
     ;; the scalar path transits a C double, which rounds past 2^53; the
     ;; rhs tensor must live on a's device, not the default
     (t-op a (tensor b #:device (tensor-device a)))]
    [else (s-op a (exact->inexact b))]))

(define/contract-out eq compare/c (comparison g:eq-tensor g:eq-scalar)) ;; noqa
(define/contract-out ne compare/c (comparison g:ne-tensor g:ne-scalar)) ;; noqa
(define/contract-out lt compare/c (comparison g:lt-tensor g:lt-scalar)) ;; noqa
(define/contract-out le compare/c (comparison g:le-tensor g:le-scalar)) ;; noqa
(define/contract-out gt compare/c (comparison g:gt-tensor g:gt-scalar)) ;; noqa
(define/contract-out ge compare/c (comparison g:ge-tensor g:ge-scalar)) ;; noqa
