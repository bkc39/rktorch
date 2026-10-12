#lang racket/base

(module+ test
  (require (only-in racket/port call-with-output-string)
           (only-in rackunit test-case)
           resyntax/test/private/rackunit
           "resyntax-rules.rkt")

  (define (program . forms)
    (code-block
     (call-with-output-string
      (lambda (out)
        (displayln "#lang racket/base" out)
        (for ([form (in-list forms)])
          (writeln form out))))))

  (parameterize ([current-suite-under-test rktorch-rules])
    (test-case "ge on a tensor becomes >="
      (check-suite-refactors
       (program '(require torch) '(ge (tensor (list 1 2 3)) 2))
       (program '(require torch) '(>= (tensor (list 1 2 3)) 2))))

    (test-case "gt, le and lt become >, <= and <"
      (check-suite-refactors
       (program '(require torch)
                '(let ([t (tensor (list 1 2 3))])
                   (list (gt t 1) (le t 2) (lt t (tensor (list 3 2 1))))))
       (program '(require torch)
                '(let ([t (tensor (list 1 2 3))])
                   (list (> t 1) (<= t 2) (< t (tensor (list 3 2 1))))))))

    (test-case "ge stays where >= is racket/base's"
      (check-suite-does-not-refactor
       (program '(require (only-in torch ge tensor))
                '(ge (tensor (list 1 2 3)) 2))))

    (test-case "a ge that is not torch's stays"
      (check-suite-does-not-refactor
       (program '(require torch)
                '(define (ge a b) (>= a b))
                '(ge 2 1))))))
