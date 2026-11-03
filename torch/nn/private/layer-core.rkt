#lang racket/base

(require (for-syntax racket/base
                     (only-in syntax/parse/pre expr))
         (only-in racket/generic define-generics)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/stxparam define-syntax-parameter)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../foreign.rkt" device dtype prop:to)
         (only-in (submod "../../foreign.rkt" unsafe) to!)
         (only-in "../../foreign/autograd-ops.rkt" collect-at-forward-trough!))

;; the noqa'd exports are macro expansions raco review cannot see
(provide gen:layer
         layer?
         layer-parameters ;; noqa
         layer-named-parameters ;; noqa
         layer-buffers ;; noqa
         layer-named-buffers ;; noqa
         layer-named-children ;; noqa
         layer-mode ;; noqa
         layer-set-mode! ;; noqa
         layer-rebuild ;; noqa
         fields-of
         fields?
         prop:fields
         prop:on-move
         prop:rebuilder
         (struct-out registry)
         with-mode)

(define-generics layer
  (layer-forward layer . inputs)
  (layer-parameters layer)
  (layer-named-parameters layer prefix)
  (layer-buffers layer)
  (layer-named-buffers layer prefix)
  (layer-named-children layer)
  (layer-own-tensors layer)
  (layer-mode layer)
  (layer-set-mode! layer mode)
  (layer-rebuild layer child tensor)
  #:derive-property prop:to (lambda (m dev dtype) (move-layer! m dev dtype))
  #:derive-property prop:procedure
  (make-keyword-procedure
   (lambda (kws kw-args m . inputs) (call-layer/keywords m kws kw-args inputs))
   (lambda (m . inputs) (apply call-layer m inputs)))
  #:fallbacks
  [(define (layer-parameters self) '()) ;; noqa
   (define (layer-named-parameters self prefix) '()) ;; noqa
   (define (layer-buffers self) '()) ;; noqa
   (define (layer-named-buffers self prefix) '()) ;; noqa
   (define (layer-named-children self) '()) ;; noqa
   (define (layer-own-tensors self) (own-tensors self)) ;; noqa
   (define (layer-mode self) 'train) ;; noqa
   (define (layer-set-mode! self mode) (void)) ;; noqa
   (define (layer-rebuild self child tensor) ;; noqa
     (raise-arguments-error
      'layer-copy
      "a hand-written gen:layer has no layer-rebuild method to copy it with"
      "layer" self))])

;; PyTorch: "nn.Module.to only accepts floating point or complex dtypes",
;; and its convert forwards the dtype only to floating-point tensors — an
;; int64 or bool buffer keeps its dtype and changes device alone
(define floating-dtypes '(float32 float64 float16 bfloat16))

(define (floating? t)
  (and (memq (dtype t) floating-dtypes) #t))

(define-values (prop:on-move on-move? on-move-ref)
  (make-struct-type-property 'on-move))

(define (placement t)
  (cons (device t) (dtype t)))

(define (move-tensor! t dev dtype)
  (define dt (and dtype (floating? t) dtype))
  (cond
    [(and dev dt) (to! t dev dt)]
    [dev (to! t dev)]
    [dt (to! t dt)]
    [else (void)]))

;; The fallback for a layer that does not declare its own tensors apart from
;; its children's: both lists here are recursive, so this is the expensive
;; way round and `registry` overrides it.
(define (own-tensors m)
  (define theirs (make-hasheq))
  (for* ([c (in-list (layer-named-children m))]
         [t (in-list (append (layer-parameters (cdr c))
                             (layer-buffers (cdr c))))])
    (hash-set! theirs t #t))
  (define mine (remove-duplicates (append (layer-parameters m) (layer-buffers m)) eq?))
  (for/list ([t (in-list mine)]
             #:unless (hash-ref theirs t #f))
    t))

(define (tree-placements m)
  (define seen (make-hasheq))
  (define before (make-hasheq))
  (let walk ([m m])
    (cond
      [(hash-has-key? seen m) (void)]
      [else
       (hash-set! seen m #t)
       (for ([c (in-list (layer-named-children m))])
         (walk (cdr c)))
       (for ([t (in-list (layer-own-tensors m))])
         (hash-set! before t (placement t)))]))
  before)

;; a tensor tied across siblings moves with the first, so read placements first
(define (move-layer! m dev dtype)
  (when (and dtype (not (memq dtype floating-dtypes)))
    (raise-arguments-error 'to "a layer only moves to a floating-point dtype"
                           "dtype" dtype))
  (move-tree! m dev dtype (and (hooked? m) (tree-placements m)) (make-hasheq))
  m)

(define (hooked? m)
  (define seen (make-hasheq))
  (let walk ([m m])
    (cond
      [(hash-has-key? seen m) #f]
      [else
       (hash-set! seen m #t)
       (or (on-move? m)
           (for/or ([c (in-list (layer-named-children m))])
             (walk (cdr c))))])))

(define (move-tree! m dev dtype before seen)
  (cond
    [(hash-has-key? seen m) (hash-ref seen m)]
    [else
     (define children-moved?
       (for/fold ([moved? #f]) ([c (in-list (layer-named-children m))])
         (or (move-tree! (cdr c) dev dtype before seen) moved?)))
     (define own (layer-own-tensors m))
     (for ([t (in-list own)])
       (move-tensor! t dev dtype))
     (define moved?
       (and before
            (or children-moved?
                (for/or ([t (in-list own)])
                  (not (equal? (hash-ref before t) (placement t)))))))
     (hash-set! seen m moved?)
     (when (and moved? (on-move? m))
       ((on-move-ref m) m))
     moved?]))

(define-syntax-parameter with-mode
  (lambda (stx)
    (raise-syntax-error
     #f (string-append "only allowed inside a define-layer #:forward body"
                       " or a define-trainer or define-callback hook")
     stx)))

(define layer-call-key (make-continuation-mark-key 'layer-call))

;; Applying any layer reaches the method through here, since gen:layer
;; derives prop:procedure from it, so a layer written by hand gets the
;; trough too. The mark tells a nested call from the outermost one, whose
;; return is where a no-grad loop's memory is at its lowest.
(define-syntax-parse-rule (at-forward-trough call:expr)
  (if (continuation-mark-set-first #f layer-call-key)
      call
      (begin0 (with-continuation-mark layer-call-key #t call)
              (collect-at-forward-trough!))))

(define (call-layer m . inputs) ;; noqa
  (at-forward-trough (apply layer-forward m inputs)))

;; Only define-layer's forward and procedure->Layer's take keywords; a
;; hand-written layer-forward is a generic method, which cannot, so the
;; application reports that method as the one refusing them.
(define (call-layer/keywords m kws kw-args inputs) ;; noqa
  (define forward (if (registry? m) (registry-forward m) layer-forward))
  (at-forward-trough (keyword-apply forward kws kw-args m inputs)))

(struct registry (forward params buffers children [mode #:mutable])
  #:methods gen:layer
  [(define (layer-forward self . inputs)
     (apply (registry-forward self) self inputs))
   (define (layer-parameters self)
     (append (map cdr (registry-params self))
             (append-map child-parameters (registry-children self))))
   (define (layer-named-parameters self prefix)
     (append (for/list ([p (in-list (registry-params self))])
               (cons (string-append prefix (car p)) (cdr p)))
             (append-map (lambda (c) (child-named-parameters c prefix))
                         (registry-children self))))
   (define (layer-buffers self)
     (append (map cdr (registry-buffers self))
             (append-map child-buffers (registry-children self))))
   (define (layer-named-buffers self prefix)
     (append (for/list ([b (in-list (registry-buffers self))])
               (cons (string-append prefix (car b)) (cdr b)))
             (append-map (lambda (c) (child-named-buffers c prefix))
                         (registry-children self))))
   (define (layer-named-children self)
     (registry-children self))
   (define (layer-own-tensors self)
     (append (map cdr (registry-params self))
             (map cdr (registry-buffers self))))
   (define (layer-set-mode! self mode)
     (set-registry-mode! self mode)
     (for ([c (in-list (registry-children self))])
       (child-set-mode! c mode)))
   (define (layer-mode self)
     (registry-mode self))
   (define (layer-rebuild self child tensor)
     ((rebuilder-of self) self child tensor))])

(define-values (prop:rebuilder rebuilder? rebuilder-of) ;; noqa
  (make-struct-type-property 'rebuilder))

(define-values (prop:fields fields? fields-of)
  (make-struct-type-property 'fields))

(define (child-parameters c)
  (layer-parameters (cdr c)))

(define (child-buffers c)
  (layer-buffers (cdr c)))

(define (child-set-mode! c mode)
  (layer-set-mode! (cdr c) mode))

(define (child-prefix c prefix)
  (if (string=? (car c) "") prefix (string-append prefix (car c) ".")))

(define (child-named-parameters c prefix)
  (layer-named-parameters (cdr c) (child-prefix c prefix)))

(define (child-named-buffers c prefix)
  (layer-named-buffers (cdr c) (child-prefix c prefix)))
