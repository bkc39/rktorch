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

    (test-case "add, sub, mul and div become +, -, * and /"
      (check-suite-refactors
       (program '(require torch)
                '(let ([t (tensor (list 1.0 2.0))])
                   (list (add t 1.0) (sub 1.0 t) (mul t t) (div t 2.0))))
       (program '(require torch)
                '(let ([t (tensor (list 1.0 2.0))])
                   (list (+ t 1.0) (- 1.0 t) (* t t) (/ t 2.0))))))

    (test-case "matmul becomes @"
      (check-suite-refactors
       (program '(require torch)
                '(let ([m (eye 2)]) (matmul m m)))
       (program '(require torch)
                '(let ([m (eye 2)]) (@ m m)))))

    (test-case "a named operator in a ~> step becomes the symbol"
      (check-suite-refactors
       (program '(require torch) '(~> (tensor (list 1.0 2.0)) (mul 2.0)))
       (program '(require torch) '(~> (tensor (list 1.0 2.0)) (* 2.0)))))

    (test-case "a ~>> step keeps its operand first"
      (check-suite-refactors
       (program '(require torch) '(~>> (tensor (list 1.0 2.0)) (sub 1.0)))
       (program '(require torch) '(~>> (tensor (list 1.0 2.0)) (- 1.0)))))

    (test-case "mul stays where * is racket/base's"
      (check-suite-does-not-refactor
       (program '(require (only-in torch mul tensor))
                '(mul (tensor (list 1.0 2.0)) 2.0))))

    (test-case "L9: mul stays under torch/safe alone"
      (check-suite-does-not-refactor
       (program '(require torch/safe)
                '(let ([t (tensor (list 1.0 2.0))]) (mul t 2.0)))))

    (test-case "L9: mul becomes * under torch/safe and torch/operators"
      (check-suite-refactors
       (program '(require torch/operators torch/safe)
                '(let ([t (tensor (list 1.0 2.0))]) (mul t 2.0)))
       (program '(require torch/operators torch/safe)
                '(let ([t (tensor (list 1.0 2.0))]) (* t 2.0)))))

    (test-case "ge stays where >= is racket/base's"
      (check-suite-does-not-refactor
       (program '(require (only-in torch ge tensor))
                '(ge (tensor (list 1 2 3)) 2))))

    (test-case "a ge that is not torch's stays"
      (check-suite-does-not-refactor
       (program '(require torch)
                '(define (ge a b) (>= a b))
                '(ge 2 1)))))

  (parameterize ([current-suite-under-test rktorch-rules])
    (test-case "tensor-device and tensor-dtype become device and dtype"
      (check-suite-refactors
       (program '(require torch)
                '(let ([t (tensor (list 1 2))])
                   (list (tensor-device t) (tensor-dtype t))))
       (program '(require torch)
                '(let ([t (tensor (list 1 2))])
                   (list (device t) (dtype t))))))

    (test-case "a long getter passed as a value becomes the short one"
      (check-suite-refactors
       (program '(require torch)
                '(map tensor-dtype (list (tensor (list 1 2)))))
       (program '(require torch)
                '(map dtype (list (tensor (list 1 2)))))))

    (test-case "a long getter as a ~> step becomes the short one"
      (check-suite-refactors
       (program '(require torch) '(~> (tensor (list 1.0)) tensor-device))
       (program '(require torch) '(~> (tensor (list 1.0)) device))))

    (test-case "a long getter call as a ~>> step becomes the short one"
      (check-suite-refactors
       (program '(require torch)
                '(~>> (tensor (list 1.0)) (tensor-dtype) (list 1)))
       (program '(require torch)
                '(~>> (tensor (list 1.0)) (dtype) (list 1)))))

    (test-case "tensor-device stays where device names a local"
      (check-suite-does-not-refactor
       (program '(require torch)
                '(define (place xs)
                   (define device (tensor-device xs))
                   device))))

    (test-case "tensor-dtype stays where dtype is not imported"
      (check-suite-does-not-refactor
       (program '(require (only-in torch tensor tensor-dtype))
                '(tensor-dtype (tensor (list 1 2))))))

    (test-case "a tensor-dtype that is not torch's stays"
      (check-suite-does-not-refactor
       (program '(require torch)
                '(define (tensor-dtype t) 'float32)
                '(tensor-dtype 1))))

    (test-case "a renamed import of tensor-dtype stays"
      (check-suite-does-not-refactor
       (program '(require (rename-in torch [tensor-dtype long-dtype]))
                '(long-dtype (tensor (list 1 2))))))

    (test-case "a require naming both getters stays"
      (check-suite-does-not-refactor
       (program '(require (only-in torch dtype tensor tensor-dtype))
                '(dtype (tensor (list 1 2)))))))

  (define calls
    '((define (f . xs) (car xs))
      (define (g . xs) (car xs))
      (define (h . xs) (car xs))
      (define (k #:y [y 0] . xs) (cons y xs))))

  (define (threaded . forms)
    (apply program '(require torch) (append calls forms)))

  (parameterize ([current-suite-under-test rktorch-rules])
    (test-case "three nested calls thread from the innermost call"
      (check-suite-refactors
       (threaded '(f (g (h 1))))
       (threaded '(~> (h 1) g f))))

    (test-case "steps keep their trailing arguments"
      (check-suite-refactors
       (threaded '(f (g (h 1 2) 3) 4))
       (threaded '(~> (h 1 2) (g 3) (f 4)))))

    (test-case "a residual sum threads through its second operand"
      (check-suite-refactors
       (threaded '(lambda (x) (f (+ x (g (h x))) 2)))
       (threaded '(lambda (x) (~> (h x) g (+ x) (f 2))))))

    (test-case "a product threads through either operand"
      (check-suite-refactors
       (threaded '(list (* 2 (g (h 1))) (* (g (h 1)) 2)))
       (threaded '(list (~> (h 1) g (* 2)) (~> (h 1) g (* 2))))))

    (test-case "values and list collect peers rather than take a step"
      (check-suite-refactors
       (threaded '(values (f (g (h 1))) (list (g (h 1)) 2)))
       (threaded '(values (~> (h 1) g f) (list (g (h 1)) 2)))))

    (test-case "racket/base's + threads the same way"
      (check-suite-refactors
       (program '(require threading) '(+ 1 (add1 (abs 1))))
       (program '(require threading) '(~> (abs 1) add1 (+ 1)))))

    (test-case "a ~> starting from nested calls starts from the innermost"
      (check-suite-refactors
       (threaded '(~> (f (g (h 1)) 2) h))
       (threaded '(~> (h 1) g (f 2) h))))

    (test-case "quoted data rides along as an argument"
      (check-suite-refactors
       (threaded '(f (g (h 1) 'v)))
       (threaded '(~> (h 1) (g 'v) f))))

    (test-case "a call joining two computations starts the chain"
      (check-suite-refactors
       (threaded '(f (g (- (h 1) (h 2)))))
       (threaded '(~> (- (h 1) (h 2)) g f))))

    (test-case "a keyword call threads like any other"
      (check-suite-refactors
       (threaded '(k (g (h 1)) #:y 2))
       (threaded '(~> (h 1) g (k #:y 2)))))

    (test-case "a ~> step already takes the previous result"
      (check-suite-does-not-refactor (threaded '(~> 1 (f (g (h 2)))))))

    (test-case "a ~>> step takes it last"
      (check-suite-does-not-refactor (threaded '(~>> 1 (f (g (h 2)))))))

    (test-case "a keyword step takes it too"
      (check-suite-does-not-refactor (threaded '(~> 1 (k (g (h 2)) #:y 3)))))

    (test-case "a contracted torch step takes it too"
      (check-suite-does-not-refactor
       (threaded '(~> (tensor (list 1.0 2.0)) (reshape (g (h -1)))))))

    (test-case "a computed argument beside the chain stops it"
      (check-suite-does-not-refactor (threaded '(f (g (h 1)) (h 2)))))

    (test-case "two calls stay"
      (check-suite-does-not-refactor (threaded '(f (g 1)))))

    (test-case "an operand that runs code keeps its place"
      (check-suite-does-not-refactor (threaded '(+ (f 1) (g (h 1))))))

    (test-case "a non-commutative operator keeps its operands"
      (check-suite-does-not-refactor
       (threaded '(lambda (x) (f (- x (g (h x))))))))

    (test-case "a sum of three operands stays"
      (check-suite-does-not-refactor (threaded '(+ 1 2 (g (h 1))))))

    (test-case "a core form breaks the chain"
      (check-suite-does-not-refactor (threaded '(f (g (if (h 1) 2 3))))))

    (test-case "a with- form keeps its setting in place"
      (check-suite-refactors
       (threaded '(with-default-device (f (g (h 'cpu))) 1))
       (threaded '(with-default-device (~> (h 'cpu) g f) 1))))

    (test-case "an assertion is not a step"
      (check-suite-does-not-refactor
       (threaded '(define (check-true x) x) '(check-true (f (g 1))))))

    (test-case "without threading's ~> nothing threads"
      (check-suite-does-not-refactor
       (program '(add1 (sub1 (abs 1))))))))
