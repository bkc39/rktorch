#lang racket/base

(module lib racket/base
  (require (only-in racket/contract/base -> ->i not/c)
           (only-in "../private/contract.rkt" define/contract-out))
  (provide internal-caller)

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

  ;; the contract names a predicate defined below it
  (define/contract-out (nat-add1 n) (-> nat? nat?) (add1 n))
  (define (nat? v) (and (exact-integer? v) (>= v 0)))

  (define (internal-caller) (split-heads 32 5)))

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

  ;; the caller is blamed, not the module that defines safe-div
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

  ;; boundary-only: the same arguments that violate #:pre across the boundary
  ;; are unchecked from inside the defining module
  (check-equal? (internal-caller) 6)

  ;; contract-out exports its names as syntax bindings, so both lists count
  (let-values ([(vars stxs)
                (module->exports (quote-module-path ".." lib))])
    (check-equal? (sort (map (lambda (e) (symbol->string (car e)))
                             (append* (map cdr (append vars stxs))))
                        string<?)
                  '("answer" "internal-caller" "nat-add1" "safe-div"
                    "split-heads")))

  (check-exn #rx"only allowed at module level"
             (lambda ()
               (convert-compile-time-error
                (let ()
                  (define/contract-out (nope x) (-> number? number?) x)
                  (nope 1))))))
