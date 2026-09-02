#lang racket/base

(module lib racket/base
  (require (only-in racket/contract/base -> ->i not/c)
           (only-in "../private/contract.rkt" define/contract-out))
  (provide call-across-no-boundary)

  (define/contract-out (safe-div a b)
    (-> number? (not/c zero?) number?)
    (/ a b))

  (define/contract-out (split-heads n-embd n-head)
    (->i ([n-embd exact-positive-integer?]
          [n-head exact-positive-integer?])
         #:pre (n-embd n-head) (zero? (remainder n-embd n-head))
         [result exact-positive-integer?]
         #:post (result) (positive? result))
    (quotient n-embd n-head))

  (define/contract-out answer exact-integer? 42)

  ;; nat? must stay below nat-add1 -- the forward reference is the thing
  ;; under test, and hoisting it would silently retire this case
  (define/contract-out (nat-add1 n) (-> nat? nat?) (add1 n))
  (define (nat? v) (and (exact-integer? v) (>= v 0)))

  (define (call-across-no-boundary) (split-heads 32 5)))

(module+ test
  (require (only-in racket/list append*)
           rackunit
           (only-in syntax/location quote-module-path)
           (only-in syntax/macro-testing convert-compile-time-error)
           (submod ".." lib)
           (only-in "../private/contract.rkt" define/contract-out))

  (check-equal? (safe-div 6 3) 2)
  (check-equal? (split-heads 32 4) 8)
  (check-equal? answer 42)
  (check-equal? (nat-add1 1) 2)

  (check-exn #rx"^safe-div: contract violation"
             (lambda () (safe-div 1 0)))

  (define ((message-matching pattern) e)
    (and (exn:fail? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))
  (define blames-the-caller
    (message-matching
     #rx"blaming: [(][^)]*contract-out-test\\.rkt test[)]"))
  (define contract-comes-from-lib
    (message-matching
     #rx"contract from: [(][^)]*contract-out-test\\.rkt lib[)]"))
  (check-exn blames-the-caller (lambda () (safe-div 1 0)))
  (check-exn contract-comes-from-lib (lambda () (safe-div 1 0)))

  (check-exn #rx"#:pre condition violation"
             (lambda () (split-heads 32 5)))
  (check-exn #rx"n-head: 5"
             (lambda () (split-heads 32 5)))

  (check-equal? (call-across-no-boundary) 6)

  ;; contract-out exports its names as syntax bindings, so both lists count
  (let-values ([(vars stxs)
                (module->exports (quote-module-path ".." lib))])
    (check-equal? (sort (map (lambda (e) (symbol->string (car e)))
                             (append* (map cdr (append vars stxs))))
                        string<?)
                  '("answer" "call-across-no-boundary" "nat-add1" "safe-div"
                    "split-heads")))

  (check-exn #rx"only allowed at module level"
             (lambda ()
               (convert-compile-time-error
                (let ()
                  (define/contract-out (nope x) (-> number? number?) x)
                  (nope 1))))))

(module checked-lib racket/base
  (require (only-in racket/contract/base -> not/c)
           (only-in "../private/contract.rkt" define/checked-out))
  (provide sibling-call)

  (define/checked-out (safe-div a b)
    (-> number? (not/c zero?) number?)
    (/ a b))

  (define (sibling-call) (safe-div 1 0)))

(module+ test
  (require (prefix-in plain: (submod ".." checked-lib))
           (prefix-in checked: (submod ".." checked-lib checked)))

  (check-equal? (checked:safe-div 6 3) 2)
  (check-equal? (plain:safe-div 6 3) 2)

  (check-exn #rx"safe-div: contract violation"
             (lambda () (checked:safe-div 1 0)))
  (check-exn (message-matching
              #rx"blaming: [(][^)]*contract-out-test\\.rkt test[)]")
             (lambda () (checked:safe-div 1 0)))

  (check-exn #rx"^/: division by zero"
             (lambda () (plain:safe-div 1 0)))
  (check-exn #rx"^/: division by zero" plain:sibling-call))

(module+ test
  (require (only-in racket/file file->string)
           racket/runtime-path
           (only-in racket/sequence sequence->list))

  (define-runtime-path foreign-dir "../foreign")

  (for ([f (in-list (sequence->list (in-directory foreign-dir)))]
        #:when (regexp-match? #rx"[.]rkt$" (path->string f)))
    (check-false (regexp-match? #rx"submod[^)]*checked" (file->string f))
                 (format "~a requires a checked submodule" f))))
