#lang racket/base

(require (only-in racket/list check-duplicates)
         (only-in "../buffer.rkt" Buffer?)
         (only-in "../parameter.rkt" Parameter?)
         (only-in "layer-core.rkt"
                  layer-named-buffers layer-named-children layer-named-parameters layer?
                  prop:rebuilder registry registry-buffers registry-children
                  registry-mode registry-params))

(provide (struct-out Children%)
         (struct-out Fn%)
         (struct-out Parameters%)
         check-names
         classify
         fn-forward)

(struct Fn% registry (proc)
  #:reflection-name 'Fn
  #:property prop:rebuilder
  (lambda (self _child _tensor _seen)
    (unless (null? (append (registry-params self) (registry-buffers self)
                           (registry-children self)))
      (raise-arguments-error
       'layer-copy
       (string-append "a procedure->Layer layer with parameters, buffers or"
                      " children cannot be copied: its procedure would"
                      " still read the original's")
       "layer" self))
    (Fn% fn-forward '() '() '() (registry-mode self) (Fn%-proc self))))

(define fn-forward
  (make-keyword-procedure
   (lambda (kws kw-args self . inputs)
     (keyword-apply (Fn%-proc self) kws kw-args inputs))
   (lambda (self . inputs)
     (apply (Fn%-proc self) inputs))))

(struct Parameters% (alist)
  #:reflection-name 'Parameters)

(struct Children% (alist)
  #:reflection-name 'Children)

(define (classify names vals) ;; noqa
  (for/fold ([params '()] [buffers '()] [children '()]
             #:result (values (reverse params)
                              (reverse buffers)
                              (reverse children)))
            ([name (in-list names)] [v (in-list vals)])
    (cond
      [(Parameter? v) (values (cons (cons name v) params) buffers children)]
      [(Parameters%? v)
       (values (append (reverse (Parameters%-alist v)) params)
               buffers children)]
      [(Buffer? v) (values params (cons (cons name v) buffers) children)]
      [(Children%? v)
       (values params buffers
               (append (reverse (Children%-alist v)) children))]
      [(layer? v) (values params buffers (cons (cons name v) children))]
      [else (values params buffers children)])))

(define (check-names who m) ;; noqa
  (define child-clash (check-duplicates (map car (layer-named-children m))))
  (when child-clash
    (raise-arguments-error who "two children would share a name"
                           "name" child-clash))
  (define params (map car (layer-named-parameters m "")))
  (define clash (check-duplicates params))
  (when clash
    (raise-arguments-error who "two parameters would share a name"
                           "name" clash))
  (define entry-clash
    (check-duplicates (append params (map car (layer-named-buffers m "")))))
  (when entry-clash
    (raise-arguments-error who "two state-dict entries would share a name"
                           "name" entry-clash))
  m)
