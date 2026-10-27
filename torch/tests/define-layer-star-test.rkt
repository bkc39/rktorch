#lang racket/base

(module+ test
  (require rackunit
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
    (call r 'set-k! (Linear 1 1))
    (call r 'set-k! 3)
    (check-equal? (call r 'k-of) 3 "a plain field stays plain whatever it held"))

  (test-case "a plain field set! to a layer after construction is not registered"
    (define h (Holder))
    (define inner (Linear 2 2))
    (call h 'hold! inner)
    (check-eq? (call h 'held) inner)
    (check-equal? (parameters h) '())
    (check-equal? (children h) '())
    (check-equal? (state-dict h) '())
    (define copy (layer-copy h))
    (check-false (eq? (call copy 'held) inner) "layer-copy copies what the field holds")
    (check-equal? (children copy) '() "and still does not register it")
    (call copy 'hold! 5)
    (check-eq? (call h 'held) inner "the copy's slot is its own"))

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
                    (Defaulted)))))))
