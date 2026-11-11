#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/contract/base contract-name value-contract)
           (only-in racket/list make-list)
           (only-in racket/match match)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in scribble/reader read-inside)
           (only-in threading ~>)
           (prefix-in g: (only-in "../generated.rkt" dropout))
           (only-in "../main.rkt"
                    arange backward! current-generator draw-seed dtype eq grad item
                    make-generator manual-seed! numel ones reshape sum tensor
                    tensor->list to-dtype zeros)
           (only-in "../nn.rkt" Dropout dropout eval!)
           (only-in "private/blame.rkt" blamed contract-violation?)
           (only-in "private/python-env.rkt" python-check python-torch-available?)
           (only-in "private/sandbox.rkt" with-fresh-eval)
           (only-in "private/within.rkt" bounded-test))

  (define caller (blamed "dropout-generator-test.rkt"))

  (define ((raised-by who rx) e)
    (and (exn:fail:contract? e)
         (regexp-match? (regexp (format "^~a: contract violation" who)) (exn-message e))
         (regexp-match? rx (exn-message e))))

  (define input (~> (arange 1 257) (reshape 4 64)))

  (define unchanged-input (tensor->list input))

  (define (functional [generator #f])
    (tensor->list (dropout input #:p 0.3 #:generator generator)))

  (define (layer [generator #f])
    (tensor->list ((Dropout #:p 0.3) input #:generator generator)))

  (define (under-current-generator seed draw)
    (parameterize ([current-generator (make-generator seed)])
      (draw)))

  (define (unchanged? generator seed)
    (equal? (draw-seed #:generator generator)
            (draw-seed #:generator (make-generator seed))))

  (define (dropped out)
    (~> (eq out 0) (to-dtype 'float64) sum item))

  (define (output-mean out)
    (/ (~> out (to-dtype 'float64) sum item) (numel out)))

  (bounded-test "one seed through current-generator draws one mask"
    (for ([draw (list functional layer)] [name '(dropout Dropout)])
      (check-equal? (under-current-generator 3 draw) (under-current-generator 3 draw)
                    (format "~a" name))
      (check-equal? (under-current-generator 3 draw) (draw (make-generator 3))
                    (format "~a: #:generator and current-generator agree" name))))

  (bounded-test "different seeds draw different masks"
    (for ([draw (list functional layer)] [name '(dropout Dropout)])
      (check-not-equal? (under-current-generator 3 draw) (under-current-generator 4 draw)
                        (format "~a" name))
      (check-not-equal? (draw (make-generator 3)) (draw (make-generator 4))
                        (format "~a, passed" name))))

  (bounded-test "the functional form and the layer draw the same mask"
    (check-equal? (functional (make-generator 9)) (layer (make-generator 9))))

  (bounded-test "an explicit #:generator wins over current-generator"
    (define parameterized (make-generator 1))
    (check-equal? (parameterize ([current-generator parameterized])
                    (functional (make-generator 2)))
                  (functional (make-generator 2)))
    (check-true (unchanged? parameterized 1)))

  (bounded-test "each float dtype drops at rate p and keeps the mean at 1"
    (define n 1000000)
    (define p 0.3)
    (define keep (- 1 p))
    (define rate-sigma (~> (* p keep) (/ n) sqrt))
    (for ([type '(float32 float64 float16 bfloat16)]
          [scale-ulp (list (expt 2 -23) (expt 2 -52) (expt 2 -10) (expt 2 -7))])
      (define out (dropout (ones n #:dtype type) #:p p #:generator (make-generator 5)))
      (check-equal? (dtype out) type)
      (check-= (/ (dropped out) n) p (* 5 rate-sigma) (format "~a: drop rate" type))
      (check-= (output-mean out) 1.0 (+ (* 5 (/ rate-sigma keep)) scale-ulp)
               (format "~a: mean" type))))

  (bounded-test "half and bfloat16 masks are drawn at full precision"
    (for ([type '(bfloat16 float16)] [p '(0.001 0.0002)])
      (define out (dropout (ones 1000000 #:dtype type) #:p p #:generator (make-generator 1)))
      (check-true (> (dropped out) 0) (format "~a at p = ~a drops some" type p))))

  (bounded-test "p = 1 zeroes everything, without a NaN or a draw"
    (define generator (make-generator 12))
    (define zeroed (make-list 256 0.0))
    (check-equal? (tensor->list (reshape (dropout input #:p 1) -1)) zeroed)
    (check-equal? (tensor->list (reshape (dropout input #:p 1 #:generator generator) -1)) zeroed)
    (check-equal? (tensor->list (reshape ((Dropout #:p 1) input #:generator generator) -1))
                  zeroed)
    (check-true (unchanged? generator 12))
    (for ([source (list #f (make-generator 2))])
      (define leaf (ones 3 #:requires-grad? #t))
      (~> (dropout leaf #:p 1 #:generator source) sum backward!)
      (check-equal? (tensor->list (grad leaf)) '(0.0 0.0 0.0))))

  (bounded-test "p = 1 leaves the global stream where it was"
    (manual-seed! 12)
    (define next (draw-seed))
    (manual-seed! 12)
    (void (dropout input #:p 1))
    (void ((Dropout #:p 1) input))
    (check-equal? (draw-seed) next))

  (bounded-test "an evaluating layer is the identity and draws nothing"
    (define generator (make-generator 6))
    (define evaluating (eval! (Dropout #:p 0.5)))
    (check-equal? (tensor->list (evaluating input #:generator generator)) unchanged-input)
    (check-equal? (tensor->list (parameterize ([current-generator generator])
                                  (evaluating input)))
                  unchanged-input)
    (check-true (unchanged? generator 6)))

  (bounded-test "#:training? #f and #:p 0 draw nothing"
    (define generator (make-generator 8))
    (check-equal? (tensor->list (dropout input #:training? #f #:generator generator))
                  unchanged-input)
    (check-equal? (tensor->list (dropout input #:p 0 #:generator generator)) unchanged-input)
    (check-true (unchanged? generator 8)))

  (bounded-test "without a generator dropout is libtorch's, on the global stream"
    (manual-seed! 11)
    (define ours (tensor->list (dropout input #:p 0.3)))
    (define layer-drawn (begin (manual-seed! 11) (tensor->list ((Dropout #:p 0.3) input))))
    (manual-seed! 11)
    (check-equal? ours (tensor->list (g:dropout input 0.3 #t)))
    (check-equal? layer-drawn ours))

  (bounded-test "the gradient through a generator's mask is the mask"
    (define leaf (ones 4 64 #:requires-grad? #t))
    (define out (dropout leaf #:p 0.3 #:generator (make-generator 5)))
    (backward! (sum out))
    (check-equal? (tensor->list (grad leaf)) (tensor->list out)))

  (bounded-test "dropout's contract is a literal ->*, so contract-out takes its fast path"
    (check-equal? (car (contract-name (value-contract dropout))) '->*))

  (bounded-test "an argument outside the contract blames the caller"
    (check-exn (contract-violation? caller #rx"generator")
               (lambda () (dropout input #:generator 5)))
    (check-exn (contract-violation? caller #rx"#:p")
               (lambda () (dropout input #:p 1.5)))
    (check-exn (contract-violation? caller #rx"#:p")
               (lambda () (Dropout #:p -0.1))))

  (bounded-test "drawing for an integer tensor is the caller's error"
    (define ints (tensor '(1 2 3) #:dtype 'int64))
    (check-exn (raised-by 'dropout #rx"floating-point")
               (lambda () (dropout ints #:generator (make-generator 1))))
    (check-exn (raised-by 'dropout #rx"floating-point")
               (lambda ()
                 (parameterize ([current-generator (make-generator 1)])
                   (dropout ints))))
    (check-exn (raised-by 'Dropout #rx"floating-point")
               (lambda () ((Dropout) ints #:generator (make-generator 1)))))

  (bounded-test "an integer tensor passes untouched without a draw"
    (define ints (tensor '(1 2 3) #:dtype 'int64))
    (check-equal? (tensor->list (dropout ints #:training? #f #:generator (make-generator 1)))
                  '(1 2 3))
    (check-equal? (dtype (dropout ints #:training? #f)) 'int64)
    (check-equal? (tensor->list (dropout (zeros 2 #:dtype 'int64) #:p 1)) '(0 0)))

  (bounded-test "the layer rejects a #:generator that is none, drawing or not"
    (for ([apply-layer (list (lambda () ((Dropout) input #:generator 'g))
                             (lambda () ((eval! (Dropout)) (ones 2) #:generator 'g))
                             (lambda () ((Dropout #:p 0) input #:generator 'g))
                             (lambda () ((Dropout #:p 1) input #:generator 'g)))]
          [label '(training evaluating p=0 p=1)])
      (check-exn (raised-by 'Dropout #rx"generator") apply-layer (format "~a" label))))

  (bounded-test "the layer rejects an input that is no tensor, with or without a generator"
    (check-exn (raised-by 'Dropout #rx"tensor")
               (lambda ()
                 (parameterize ([current-generator (make-generator 1)])
                   ((Dropout) 5))))
    (check-exn (raised-by 'Dropout #rx"tensor") (lambda () ((Dropout) 5))))

  (define-runtime-path nn-scrbl "../scribblings/nn.scrbl")

  (define (dropout-example form)
    (match form
      [(list 'racketblock (and example (list 'define-layer 'Dropout _ ...))) example]
      [(cons head tail) (or (dropout-example head) (dropout-example tail))]
      [_ #f]))

  (bounded-test "the define-layer example in the manual draws the library's mask"
    (define example
      (dropout-example
       (call-with-input-file nn-scrbl (lambda (in) (read-line in 'any) (read-inside in)))))
    (check-not-false example "nn.scrbl has the define-layer Dropout example")
    (with-fresh-eval (ev #:time-limit 120)
      (ev `(module example racket/base
             (require racket/contract torch torch/nn)
             ,example))
      (ev '(require torch (only-in torch/nn eval!) 'example))
      (check-equal? (ev '(~> (arange 1 257)
                             (reshape 4 64)
                             ((Dropout #:p 0.3) #:generator (make-generator 9))
                             tensor->list))
                    (layer (make-generator 9)))
      (check-equal? (ev '(~> (arange 3) ((eval! (Dropout))) tensor->list))
                    (tensor->list (arange 3)))))

  (if (python-torch-available?)
      (bounded-test "the global stream draws F.dropout's mask, a generator torch.rand's"
        (define masks (python-check "dropout_generator.py"))
        (manual-seed! 7)
        (check-equal? (tensor->list (dropout input #:p 0.3)) (hash-ref masks 'global)
                      "F.dropout after torch.manual_seed")
        (check-equal? (functional (make-generator 7)) (hash-ref masks 'generator)
                      "torch.Generator().manual_seed(7) through torch.rand"))
      (printf "[dropout-generator-test] skipped parity: python3 `torch` not available ~a\n"
              "(run inside the default `nix develop`)")))
