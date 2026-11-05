#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/match match match*)
         (only-in racket/unsafe/ops unsafe-set-immutable-car! unsafe-set-immutable-cdr!)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide after-walk!
         allocated
         call-with-memo
         copy-of
         memo-who
         rebuilt-immutable
         rekey-after-walk!
         remembered
         remembering?
         remembering-memo
         with-remembering)

(struct memo (who done open lock frames
              [anchors #:mutable] [frame-count #:mutable] [low #:mutable] [pending #:mutable]
              [rekeyed #:mutable] [afters #:mutable])
  #:constructor-name make-memo
  #:omit-define-syntaxes)

(struct frame (node id anchors [shell #:mutable] [placeholder #:mutable] [copy #:mutable]))

(struct remembering ([seen #:mutable] proc)
  #:property prop:procedure
  (lambda (self name c)
    (remembering-memo self)
    ((remembering-proc self) self name c)))

(define (memo who)
  (make-memo who (make-hasheq) (make-hasheq) (make-semaphore 1) (make-hasheq) 0 0 #f '() '() '()))

(define (call-with-memo who body)
  (define seen (memo who))
  (define (walked)
    (begin0 (body seen)
            (finish! seen)))
  (define (cleared)
    (hash-clear! (memo-done seen))
    (hash-clear! (memo-open seen))
    (hash-clear! (memo-frames seen)))
  (dynamic-wind void walked cleared))

(define (copy-of seen v default)
  (hash-ref (memo-done seen) v default))

(define (after-walk! seen thunk)
  (set-memo-afters! seen (cons thunk (memo-afters seen))))

(define (rekey-after-walk! seen h)
  (set-memo-rekeyed! seen (cons h (memo-rekeyed seen))))

(define (finish! seen)
  (for ([h (in-list (reverse (memo-rekeyed seen)))])
    (define entries (hash->list h))
    (hash-clear! h)
    (for ([e (in-list entries)])
      (hash-set! h (car e) (cdr e))))
  (for ([after (in-list (reverse (memo-afters seen)))])
    (after)))

(define (remembering-memo r)
  (or (remembering-seen r)
      (raise-arguments-error 'layer-copy
                             "a child callback used after layer-copy returned")))

(define (call-with-remembering seen proc body)
  (define r (remembering seen proc))
  (dynamic-wind void
                (lambda () (body r))
                (lambda () (set-remembering-seen! r #f))))

(define-syntax-parse-rule (with-remembering ([child:id seen:expr proc:expr]) body:expr ...+)
  (call-with-remembering seen proc (lambda (child) body ...)))

(define (claim seen v)
  (define open (memo-open seen))
  (define (decide)
    (define owner (hash-ref open v #f))
    (cond
      [(and owner (not (eq? owner (current-thread))) (not (thread-dead? owner))) 'elsewhere]
      [(hash-has-key? (memo-done seen) v) 'done]
      [(eq? owner (current-thread)) 'reentered]
      [else
       (hash-set! open v (current-thread))
       'claimed]))
  (call-with-semaphore (memo-lock seen) decide))

(define (released seen v body)
  (dynamic-wind void body (lambda () (hash-remove! (memo-open seen) v))))

(define (claimed seen v build reentered)
  (case (claim seen v)
    [(done) (hash-ref (memo-done seen) v)]
    [(claimed) (released seen v build)]
    [(elsewhere)
     (raise-arguments-error (memo-who seen) "a child callback used from two threads at once")]
    [else (reentered)]))

(define (remembered seen v build [reentered (lambda () (reentered-value seen v))])
  (claimed seen v
           (lambda ()
             (define made (build))
             (hash-set! (memo-done seen) v made)
             made)
           reentered))

(define (allocated seen v allocate fill!)
  (claimed seen v
           (lambda ()
             (define made (allocate))
             (hash-set! (memo-done seen) v made)
             (anchored! seen)
             (fill! made)
             made)
           (lambda () (reentered-value seen v))))

(define (reentered-value seen v)
  (raise-arguments-error (memo-who seen) "a cycle re-enters a value still being copied"
                         "value" v))

(define (anchored! seen)
  (set-memo-anchors! seen (add1 (memo-anchors seen))))

(define (rebuilt-immutable seen v build)
  (define frames
    (call-with-semaphore (memo-lock seen)
                         (lambda () (hash-ref (memo-frames seen) v '()))))
  (define latest (and (pair? frames) (car frames)))
  (cond
    [(hash-has-key? (memo-done seen) v) (hash-ref (memo-done seen) v)]
    [(not latest) (entered seen v build)]
    [(frame-shell latest) => values]
    [(pair? v)
     (define shell (cons #f #f))
     (set-frame-shell! latest shell)
     shell]
    [(> (memo-anchors seen) (frame-anchors latest)) (entered seen v build)]
    [else
     (lower! seen (frame-id latest))
     (placeholder-of latest)]))

(define (lower! seen id)
  (define low (memo-low seen))
  (set-memo-low! seen (if low (min low id) id)))

(define (placeholder-of f)
  (or (frame-placeholder f)
      (let ([p (make-placeholder #f)])
        (set-frame-placeholder! f p)
        p)))

(define (frames-update! seen v update)
  (define frames (memo-frames seen))
  (define (updated)
    (define fs (update (hash-ref frames v '())))
    (if (null? fs) (hash-remove! frames v) (hash-set! frames v fs)))
  (call-with-semaphore (memo-lock seen) updated))

(define (pushed-since seen outer)
  (let loop ([fs (memo-pending seen)])
    (if (eq? fs outer) '() (cons (car fs) (loop (cdr fs))))))

(define (dropped! seen fs)
  (for ([f (in-list fs)])
    (frames-update! seen (frame-node f) (lambda (all) (remq f all)))))

(define (sealed f c)
  (match* ((frame-shell f) c)
    [(#f _) c]
    [(shell (cons a d))
     (unsafe-set-immutable-car! shell a)
     (unsafe-set-immutable-cdr! shell d)
     shell]))

(define (entered seen v build)
  (define id (memo-frame-count seen))
  (define f (frame v id (memo-anchors seen) #f #f #f))
  (define outer-low (memo-low seen))
  (define outer-pending (memo-pending seen))
  (define finished? #f)
  (define (restore!)
    (set-memo-pending! seen outer-pending)
    (set-memo-low! seen outer-low))
  (define (walked)
    (set-frame-copy! f (sealed f (build)))
    (define low (memo-low seen))
    (define result
      (cond
        [(and low (< low id))
         (set-memo-pending! seen (cons f (memo-pending seen)))
         (set-memo-low! seen (if outer-low (min outer-low low) low))
         (placeholder-of f)]
        [else
         (define scc (reverse (cons f (pushed-since seen outer-pending))))
         (restore!)
         (resolved! seen scc)
         (hash-ref (memo-done seen) v)]))
    (set! finished? #t)
    result)
  (define (abandoned)
    (unless finished?
      (dropped! seen (cons f (pushed-since seen outer-pending)))
      (restore!)))
  (set-memo-frame-count! seen (add1 id))
  (when (pair? v)
    (anchored! seen))
  (frames-update! seen v (lambda (fs) (cons f fs)))
  (set-memo-low! seen #f)
  (dynamic-wind void walked abandoned))

(define (resolved! seen scc)
  (define done (memo-done seen))
  (define chosen (make-hasheq))
  (for ([f (in-list scc)])
    (define v (frame-node f))
    (define target (hash-ref done v (lambda () (hash-ref! chosen v (frame-copy f)))))
    (when (frame-placeholder f)
      (placeholder-set! (frame-placeholder f) target)))
  (define nodes (hash-keys chosen))
  (define targets (map (lambda (v) (hash-ref chosen v)) nodes))
  (define graph? (ormap frame-placeholder scc))
  (define finals (if graph? (make-reader-graph targets) targets))
  (for ([v (in-list nodes)] [final (in-list finals)])
    (hash-set! done v final))
  (when graph?
    (for ([v (in-list nodes)] [c (in-list targets)] [final (in-list finals)])
      (check-survived seen v c final (placeholder-nodes scc))))
  (dropped! seen scc))

(define (placeholder-nodes scc)
  (for/hasheq ([f (in-list scc)] #:when (frame-placeholder f))
    (values (frame-placeholder f) (frame-node f))))

(define (check-survived seen v c final nodes)
  (define (expected x)
    (if (placeholder? x) (hash-ref (memo-done seen) (hash-ref nodes x)) x))
  (define (kept? x y)
    (eq? (expected x) y))
  (define survived?
    (match c
      [(box x) (and (immutable? final) (kept? x (unbox final)))]
      [(? vector?) (for/and ([x (in-vector c)] [y (in-vector final)]) (kept? x y))]
      [(? hash?)
       (for/and ([(k x) (in-hash c)])
         (and (not (placeholder? k))
              (memq k (hash-keys final))
              (kept? x (hash-ref final k))))]))
  (unless survived?
    (raise-arguments-error
     (memo-who seen)
     (string-append "a cycle of immutable vectors, boxes or hashes reaches another cycle"
                    " or keys a hash by the cycle, and make-reader-graph cannot rebuild it"
                    " without copying that too")
     "value" v)))
