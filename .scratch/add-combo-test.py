import os
os.chdir('/home/bkc/dev/rkt/rktorch/.claude/worktrees/vision-152')

p = 'torch/tests/forward-arity-test.rkt'
s = open(p).read()

old = """  (require (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear Sequential define-layer forward
                    layer-forward))

  (define-layer Pair ()
    #:forward (x y)
    (list x y))
"""
new = """  (require (only-in racket/contract/base flat-named-contract)
           (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" ones tensor-shape)
           (only-in "../nn.rkt" Linear Sequential define-layer forward
                    layer-forward))

  (define-layer Pair ()
    #:forward (x y)
    (list x y))

  (define rank2/c
    (flat-named-contract 'rank2 (lambda (t) (= 2 (length (tensor-shape t))))))

  (define-layer Stateful ()
    #:forward ([x : rank2/c] . state)
    (cons x state))
"""
assert old in s
s = s.replace(old, new)

old2 = """    (check-equal? (length (p x x)) 2)))"""
new2 = """    (check-equal? (length (p x x)) 2))

  (test-case "a rest formal and a contracted one compose"
    (define s (Stateful))
    (define x (ones 1 2))
    (check-equal? (length (s x)) 1 "the rest may be empty")
    (check-equal? (length (s x 'h 'c)) 3)
    (check-exn #rx"^Stateful: contract violation"
               (lambda () (s (ones 1 2 3) 'h)))
    (check-exn #rx"expected: rank2" (lambda () (s (ones 1 2 3))))
    (check-exn #rx"^Stateful: arity mismatch" (lambda () (s)))))"""
assert old2 in s
open(p, 'w').write(s.replace(old2, new2))
print("combination test added")
