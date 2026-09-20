import os, re
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/nn/layer.rkt'
s = open(p).read()

start = s.index('<<<<<<< HEAD')
end = s.index('>>>>>>> origin/master')
end = s.index('\n', end) + 1

merged = r"""              (~optional (~seq #:predicate pred:id))
              (~optional (~seq #:on-move moved-body:expr ...+))) ...
        #:forward (~or* (input:forward-formal ...)
                        (input:forward-formal ... . restarg:id))
        body:expr ...+)
     #:fail-when
     (and (attribute init)
          (for/or ([f (in-list (syntax->list #'(field.id ...)))]
                   [bare? (in-list (attribute field.bare?))])
            (and (not bare?) f)))
     "with #:init, a field is a bare identifier; defaults and keywords belong to the #:init formals"
     #:do [(define fields (syntax->list #'(field.id ...)))
           (define init? (and (attribute init) #t))
           (define init-ids (if init? (syntax->list #'(init.id ...)) '()))
           (define struct-id (generate-temporary #'name))
           (define input-ids (syntax->list #'(input.id ...)))
           (define arity (length input-ids))
           (define input-ctcs (attribute input.ctc))
           (define checker-ids
             (for/list ([c (in-list input-ctcs)])
               (and c (generate-temporary #'check))))]
     #:with (checker-def ...)
     (for/list ([c (in-list input-ctcs)]
                [cid (in-list checker-ids)]
                #:when c)
       #`(define #,cid
           (contract (-> #,c any) values 'name 'caller 'name
                     (quote-syntax name))))
     #:with (checked-binding ...)
     (for/list ([c (in-list input-ctcs)]
                [cid (in-list checker-ids)]
                [i (in-list input-ids)]
                #:when c)
       #`[#,i (#,cid #,i)])
     #:with sid struct-id
     #:with sid? (format-id struct-id "~a?" struct-id)
     #:with name? (format-id #'name "~a?" #'name)
     #:with reflect-name (or (attribute reflect) #'(quote name))
     #:with formals (if init? #'init.formals #'((~@ field.decl ...) ...))
     #:with (assign ...) (if init? #'(init-body ...) #'())
     #:with (absent ...) (unassigned-fields (if init? fields '()) init-ids)
     #:with (field-name ...) (field-strings fields)
     #:with (field-acc ...) (accessor-ids struct-id fields)
     #:with n-inputs #`#,arity
     #:with enough? (if (attribute restarg) #'>= #'=)
     #:with expected (if (attribute restarg)
                         #`(arity-at-least #,arity)
                         #`#,arity)
     #:with forward-lambda
     (if (attribute restarg)
         #'(lambda (input.id ... . restarg)
             (let (checked-binding ...) body ...))
         #'(lambda (input.id ...)
             (let (checked-binding ...) body ...)))
     #:with (moved-defn ...)
     (if (attribute moved-body)
         #'((define (moved-proc self dev dtype)
              (define before (placement-of self))
              (begin0 (move-layer! self dev dtype)
                      (unless (equal? before (placement-of self))
                        (let ([field.id (field-acc self)] ...)
                          moved-body ...)))))
         #'())
     #:with (moved-clause ...)
     (if (attribute moved-body) #'(#:property prop:to moved-proc) #'())
     #:with export (contract-export stx #'name #'name?
                                    (attribute ctc) (attribute pred))
     #'(begin
         moved-defn ...
         (struct sid registry (field.id ...)
           #:reflection-name reflect-name
           moved-clause ...)
         (define name? sid?)
         checker-def ...
         (define (forward-proc self . inputs)
           (unless (enough? (length inputs) n-inputs)
             (apply raise-arity-error 'name expected inputs))
           (let ([field.id (field-acc self)] ...)
             (syntax-parameterize ([with-mode (with-mode-transformer #'self)])
               (apply forward-lambda inputs))))
         (define (name . formals)
           (let ([absent #f] ...)
             assign ...
             (let-values ([(params buffers children)
                           (classify '(field-name ...) (list field.id ...))])
               (check-names
                'name
                (sid forward-proc params buffers children 'train
                     field.id ...)))))
         export)]))
"""

open(p, 'w').write(s[:start] + merged + s[end:])
print("layer.rkt resolved")
