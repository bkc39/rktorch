#lang racket/base

(module+ test
  (require (for-syntax racket/base
                       (only-in syntax/parse/pre ~var attribute syntax-parse)
                       (only-in "../private/definer.rkt" field-of layer-fields))
           rackunit
           (only-in syntax/macro-testing convert-compile-time-error)
           "../main.rkt"
           "../nn.rkt"
           (only-in (submod "../nn/layer.rkt" private)
                    define-layer* layer-method layer-methods))

  (define-values (prop:tag tagged? tag-of) (make-struct-type-property 'tag))

  (define-layer* Counter (net count history)
    #:init (width)
    (set! net (Linear width width))
    (set! count 0)
    (set! history '())
    #:property prop:tag 'counter
    #:method (bump! [by : exact-nonnegative-integer?])
    (set! count (+ count by))
    count
    #:method (remember! x)
    (set! history (cons (net x) history))
    (length history)
    #:method (net-of) net
    #:method (swap!) (set! net (Linear 2 2))
    #:method (total #:scale [scale 1] . xs) (* scale (apply + count xs))
    #:forward (x)
    (set! count (add1 count))
    (net x))

  (define-layer* Registered (w shift k)
    #:init ()
    (set! w (Parameter (ones 2)))
    (set! shift (Buffer (zeros 2)))
    (set! k 1)
    #:method (set-w!) (set! w (Parameter (zeros 2)))
    #:method (set-shift!) (set! shift (Buffer (ones 2)))
    #:method (set-k! v) (set! k v)
    #:method (k-of) k)

  (define-layer* Holder (slot)
    #:init ()
    (set! slot #f)
    #:method (hold! m) (set! slot m)
    #:method (held) slot)

  (define-layer* Teacher (student)
    #:init ()
    (set! student #f)
    #:method (attach! s) (set! student s))

  (define-layer* Student (net teacher)
    #:init ()
    (set! net (Linear 2 2))
    (set! teacher (Teacher)))

  (define-layer* Stepper (calls)
    #:init ()
    (set! calls 0)
    #:method #:hidden-arguments 1 (step run x y) (list run x y)
    #:method #:hidden-arguments 1 (step* run x . more) (list* run x more))

  (define-layer* Hooks (n)
    #:method (get) n
    #:method (mode-of) (with-mode mode))

  (define-layer* Moved (w moves)
    #:init ()
    (set! w (Parameter (ones 2)))
    (set! moves 0)
    #:on-move (set! moves (add1 moves))
    #:method (moves-of) moves
    #:forward (x) (* x w))

  (define-syntax (fields-of stx)
    (syntax-parse stx
      [(_ init? fields (~var h (field-of '(a b) 'Thing)) ...)
       #:with (~var f (layer-fields (and (syntax-e #'init?) "define-init"))) #'fields
       #''(f.id ...)]))

  (define (call m name . args)
    (apply (layer-method m name) m args))

  (define ((message-matching rx) e)
    (and (exn:fail:contract? e) (regexp-match? rx (exn-message e))))

  (test-case "a method sees the fields and its set! persists for later calls"
    (define c (Counter 2))
    (check-equal? (call c 'bump! 3) 3)
    (check-equal? (call c 'bump! 2) 5)
    (check-eq? (call c 'net-of) (car (children c)))
    (check-equal? ((layer-method c 'total) c 1 2 #:scale 2) 16))

  (test-case "forward reads and writes the same slots as the methods"
    (define c (Counter 2))
    (c (ones 2))
    (c (ones 2))
    (check-equal? (call c 'bump! 0) 2)
    (call c 'bump! 10)
    (c (ones 2))
    (check-equal? (call c 'bump! 0) 13))

  (test-case "set! of a plain field leaves parameters and the state dict alone"
    (define c (Counter 2))
    (define before (parameters c))
    (check-equal? (call c 'remember! (ones 2)) 1)
    (check-equal? (call c 'remember! (ones 2)) 2)
    (check-true (andmap eq? (parameters c) before))
    (check-equal? (map car (state-dict c)) '("net.weight" "net.bias")))

  (test-case "set! of a field registered at construction raises, naming field and type"
    (define c (Counter 2))
    (define net (call c 'net-of))
    (check-exn (message-matching #rx"^Counter: set! cannot replace a field .*\n  field: 'net")
               (lambda () (call c 'swap!)))
    (check-eq? (call c 'net-of) net)
    (define r (Registered))
    (check-exn (message-matching #rx"^Registered: .*\n  field: 'w")
               (lambda () (call r 'set-w!)))
    (check-exn (message-matching #rx"^Registered: .*\n  field: 'shift")
               (lambda () (call r 'set-shift!)))
    (call r 'set-k! 3)
    (check-equal? (call r 'k-of) 3 "a plain field takes plain values"))

  (test-case "set! of a layer, parameter or buffer into a plain field raises"
    (define h (Holder))
    (for ([v (list (Linear 2 2) (Parameter (ones 2)) (Buffer (zeros 2)))])
      (check-exn (message-matching
                  #rx"^Holder: set! cannot put a layer, parameter or buffer .*\n  field: 'slot")
                 (lambda () (call h 'hold! v))))
    (check-false (call h 'held) "the slot keeps its value")
    (check-equal? (children h) '()))

  (test-case "layer-copy passes a plain field's value through"
    (define h (Holder))
    (define inner (list (Linear 2 2)))
    (call h 'hold! inner)
    (define copy (layer-copy h))
    (check-eq? (call copy 'held) inner)
    (check-equal? (children copy) '())
    (call copy 'hold! 5)
    (check-eq? (call h 'held) inner "the copy's slot is its own"))

  (test-case "a layer cannot be set! into a field of its own child, so layer-copy ends"
    (define s (Student))
    (define t (car (filter (lambda (c) (layer-method c 'attach!)) (children s))))
    (check-exn (message-matching #rx"^Teacher: set! cannot put .*\n  field: 'student")
               (lambda () (call t 'attach! s)))
    (define c (make-custodian))
    (custodian-limit-memory c (* 256 1024 1024) c)
    (define done
      (parameterize ([current-custodian c])
        (thread (lambda () (layer-copy s)))))
    (define finished (sync/timeout 30 done))
    (custodian-shutdown-all c)
    (check-not-false finished "layer-copy returned"))

  (test-case "layer-copy keeps which fields were registered at construction"
    (define c (Counter 2))
    (call c 'bump! 4)
    (define copy (layer-copy c))
    (check-equal? (call copy 'bump! 0) 4)
    (call copy 'bump! 1)
    (check-equal? (call c 'bump! 0) 4)
    (check-false (eq? (call copy 'net-of) (call c 'net-of)))
    (check-exn (message-matching #rx"field: 'net") (lambda () (call copy 'swap!))))

  (test-case "a define-layer* without a forward cannot be applied"
    (define h (Hooks 7))
    (check-equal? (call h 'get) 7)
    (check-exn (message-matching #rx"^Hooks: this layer has no forward, so it cannot be applied")
               (lambda () (h 1)))
    (check-exn (message-matching #rx"^Hooks: this layer has no forward")
               (lambda () (h 1 #:mask 2))))

  (test-case "the methods are one table per type"
    (define a (Counter 2))
    (define b (Counter 3))
    (check-eq? (layer-methods a) (layer-methods b))
    (check-equal? (sort (hash-keys (layer-methods a)) symbol<?)
                  '(bump! net-of remember! swap! total))
    (check-false (eq? (layer-methods a) (layer-methods (Hooks 1))))
    (check-false (layer-methods (Linear 1 1)))
    (check-false (layer-method (Linear 1 1) 'get))
    (check-false (layer-method a 'missing))
    (check-exn #rx"^layer-method: contract violation" (lambda () (layer-method 5 'get))))

  (test-case "#:property attaches to the type"
    (check-true (tagged? (Counter 2)))
    (check-equal? (tag-of (Counter 2)) 'counter)
    (check-false (tagged? (Hooks 1))))

  (test-case "method inputs are checked like forward inputs"
    (define c (Counter 2))
    (check-exn (message-matching #rx"^bump!: contract violation")
               (lambda () (call c 'bump! -1)))
    (check-exn exn:fail:contract:arity? (lambda () (call c 'bump!)))
    (check-exn (message-matching #rx"^bump!: arity mismatch")
               (lambda () (call c 'bump! 1 2))))

  (test-case "hidden leading inputs are left out of an arity message"
    (define st (Stepper))
    (check-equal? (call st 'step 'run 1 2) '(run 1 2))
    (check-exn (message-matching #rx"^step: arity mismatch.*\n  expected: 2\n  given: 1\n  arguments...:\n   1$")
               (lambda () (call st 'step 'run 1)))
    (check-exn (message-matching #rx"expected: at least 1\n  given: 0$")
               (lambda () (call st 'step* 'run)))
    (check-exn (message-matching #rx"expected: 2\n  given: 0$")
               (lambda () (call st 'step))))

  (test-case "on-move and with-mode see the instance"
    (define m (Moved))
    (to m 'float64)
    (to m 'float32)
    (check-equal? (call m 'moves-of) 2)
    (check-equal? (shape (m (ones 2))) '(2))
    (define h (Hooks 1))
    (check-equal? (call h 'mode-of) 'train)
    (eval! h)
    (check-equal? (call h 'mode-of) 'eval))

  (test-case "define-layer* rejects a repeated method and a non-bare field with #:init"
    (check-exn #rx"a method is defined twice"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-layer* Twice (n)
                      #:method (get) n
                      #:method (get) n)
                    (Twice 1)))))
    (check-exn #rx"with #:init, a field is a bare identifier"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-layer* Defaulted ([n 1])
                      #:init ()
                      (set! n 2))
                    (Defaulted)))))
    (check-exn #rx"#:hidden-arguments counts more inputs than the positional formals"
               (lambda ()
                 (convert-compile-time-error
                  (let ()
                    (define-layer* Hidden ()
                      #:method #:hidden-arguments 2 (get x) x)
                    (Hidden))))))

  (test-case "with-mode outside a body names every form that binds it"
    (check-exn #rx"only allowed inside a define-layer #:forward body or a define-trainer or define-callback hook"
               (lambda () (convert-compile-time-error (with-mode 1)))))

  (test-case "the shared field checks take the definer's init clause and type"
    (check-equal? (fields-of #f ([a 1] #:k b) a b) '(a b))
    (check-equal? (fields-of #t (a b)) '(a b))
    (check-exn #rx"with define-init, a field is a bare identifier; defaults and keywords belong to the define-init formals"
               (lambda () (convert-compile-time-error (fields-of #t (a [b 1])))))
    (check-exn #rx"c is not a field of Thing"
               (lambda () (convert-compile-time-error (fields-of #f (a) c))))
    (check-exn #rx"a field name may not contain a dot"
               (lambda () (convert-compile-time-error (fields-of #f (a.b)))))))
