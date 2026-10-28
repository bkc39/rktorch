#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/list check-duplicates last)
           (only-in racket/match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-prefix?)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * + cat cuda-available? cuda-device in-flattened-tensor
                    item manual-seed! mps-available? mps-device narrow
                    native-memory-use numel permute randn
                    reclaim-native-memory! reshape shape sum tensor->list to
                    to-dtype upsample-bicubic2d)
           (only-in "../generated.rkt" broadcast-to)
           (only-in "../nn.rkt"
                    child-ref eval! in-layers layer-copy layer-mode
                    layer-rebuild linear? named-parameters
                    parameters state-dict train!)
           (only-in "../vision/vit.rkt"
                    PatchEmbed VisionTransformer huggingface-vit-key
                    patch-embed? vision-transformer?)
           (only-in (submod "../vision/vit.rkt" private) positions)
           (only-in "private/gradcheck.rkt" gradcheck)
           (only-in "private/op-calls.rkt" op-calls-of)
           (only-in "private/within.rkt" bounded-test))

  (define-runtime-path vit-keys "../vision/fixtures/vit-keys.rktd")

  (define devices
    (append (list 'cpu)
            (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define (small #:classes [classes 10])
    (VisionTransformer #:image-size 32 #:patch 4 #:width 24 #:depth 2
                       #:heads 3 #:classes classes))

  (define (same? a b)
    (and (equal? (shape a) (shape b))
         (equal? (tensor->list (to a 'cpu)) (tensor->list (to b 'cpu)))))

  (define (parameter m name)
    (cdr (assoc name (named-parameters m))))

  (define (count-of m)
    (for/sum ([p (in-list (parameters m))]) (numel p)))

  (define ((message-matching pattern) e)
    (and (exn:fail:contract? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (bounded-test "V1: each output has its documented shape"
    (define vit (small))
    (define x32 (randn 2 3 32 32))
    (check-true (vision-transformer? vit))
    (check-true (patch-embed? (child-ref vit "patch-embed")))
    (check-equal? (shape (vit x32)) '(2 10))
    (check-equal? (shape (vit x32 #:output 'cls)) '(2 24))
    (check-equal? (shape (vit x32 #:output 'tokens)) '(2 65 24))
    (check-equal? (shape (vit (randn 2 3 16 16) #:output 'tokens)) '(2 17 24))
    (check-equal? (shape (vit x32 #:output 'attention)) '(2 3 65 65))
    (check-equal? (shape (vit (randn 2 3 32 48) #:output 'tokens)) '(2 97 24))
    (define headless (small #:classes #f))
    (check-true (same? (headless x32) (headless x32 #:output 'cls)))
    (check-equal? (shape ((PatchEmbed 3 24 4) x32)) '(2 64 24)))

  (define b16 (VisionTransformer #:layer-norm-eps 1e-12 #:classes 1000))
  (define b16-headless (VisionTransformer #:layer-norm-eps 1e-12))

  (bounded-test "V2: parameter counts match the references'"
    (check-equal? (count-of (VisionTransformer #:image-size 32 #:patch 4
                                               #:width 192 #:depth 9
                                               #:heads 3))
                  4026240)
    (check-equal? (count-of b16-headless) 85798656)
    (check-equal? (count-of b16) 86567656)
    (check-equal? (count-of (VisionTransformer #:width 192 #:heads 3
                                               #:classes 1000))
                  5717416))

  (bounded-test "V3: state-dict keys follow the name map"
    (check-equal?
     (map car (state-dict (small)))
     '("cls-token" "pos-embed"
       "patch-embed.proj.weight" "patch-embed.proj.bias"
       "blocks.0.self-attn.query.weight" "blocks.0.self-attn.query.bias"
       "blocks.0.self-attn.key.weight" "blocks.0.self-attn.key.bias"
       "blocks.0.self-attn.value.weight" "blocks.0.self-attn.value.bias"
       "blocks.0.self-attn.out.weight" "blocks.0.self-attn.out.bias"
       "blocks.0.linear1.weight" "blocks.0.linear1.bias"
       "blocks.0.linear2.weight" "blocks.0.linear2.bias"
       "blocks.0.norm1.weight" "blocks.0.norm1.bias"
       "blocks.0.norm2.weight" "blocks.0.norm2.bias"
       "blocks.1.self-attn.query.weight" "blocks.1.self-attn.query.bias"
       "blocks.1.self-attn.key.weight" "blocks.1.self-attn.key.bias"
       "blocks.1.self-attn.value.weight" "blocks.1.self-attn.value.bias"
       "blocks.1.self-attn.out.weight" "blocks.1.self-attn.out.bias"
       "blocks.1.linear1.weight" "blocks.1.linear1.bias"
       "blocks.1.linear2.weight" "blocks.1.linear2.bias"
       "blocks.1.norm1.weight" "blocks.1.norm1.bias"
       "blocks.1.norm2.weight" "blocks.1.norm2.bias"
       "norm.weight" "norm.bias" "head.weight" "head.bias")))

  (bounded-test "V4: huggingface-vit-key maps recorded checkpoints onto the model"
    (define revisions (call-with-input-file vit-keys read))
    (check-equal? (length revisions) 2)
    (for ([revision (in-list revisions)])
      (match-define (list repository _ classes entries) revision)
      (define model-shapes
        (for/hash ([e (in-list (state-dict (if classes b16 b16-headless)))])
          (values (car e) (shape (cdr e)))))
      (define mapped
        (for/list ([e (in-list entries)])
          (cons (huggingface-vit-key (car e)) e)))
      (define kept (filter car mapped))
      (check-false (check-duplicates (map car kept)) repository)
      (check-equal? (sort (map car kept) string<?)
                    (sort (hash-keys model-shapes) string<?)
                    repository)
      (for ([m (in-list kept)])
        (match-define (list ours key file-shape) m)
        (check-equal? (hash-ref model-shapes ours #f) file-shape key))
      (for ([m (in-list mapped)] #:unless (car m))
        (check-true (string-prefix? (cadr m) "pooler.") (cadr m)))))

  (bounded-test "huggingface-vit-key renames, drops the pooler, keeps the unknown"
    (check-equal? (huggingface-vit-key
                   "vit.encoder.layer.3.attention.attention.query.weight")
                  "blocks.3.self-attn.query.weight")
    (check-equal? (huggingface-vit-key "encoder.layer.11.output.dense.bias")
                  "blocks.11.linear2.bias")
    (check-equal? (huggingface-vit-key "embeddings.cls_token") "cls-token")
    (check-equal? (huggingface-vit-key "classifier.weight") "head.weight")
    (check-false (huggingface-vit-key "pooler.dense.bias"))
    (check-equal? (huggingface-vit-key "vit.mystery") "vit.mystery"))

  (define (reference-tokens vit x)
    (define tokens ((child-ref vit "patch-embed") x))
    (match-define (list n _ width) (shape tokens))
    (define cls (broadcast-to (parameter vit "cls-token") (list n 1 width)))
    (define z (+ (cat (list cls tokens) 1) (parameter vit "pos-embed")))
    ((child-ref vit "norm")
     (for/fold ([z z]) ([block (in-layers (child-ref vit "blocks"))])
       (block z))))

  (define (twin-positions pos gh gw target)
    (define width (last (shape pos)))
    (cat (list (narrow pos 1 0 1)
               (~> (narrow pos 1 1 (* gh gw))
                   (reshape 1 gh gw width)
                   (permute 0 3 1 2)
                   (upsample-bicubic2d target)
                   (permute 0 2 3 1)
                   (reshape 1 -1 width)))
         1))

  (bounded-test "V5: the native grid adds pos-embed as it is"
    (for ([on (in-list devices)])
      (define vit (to (small) on))
      (define x32 (to (randn 2 3 32 32) on))
      (define x16 (to (randn 2 3 16 16) on))
      (define pos (parameter vit "pos-embed"))
      (check-true (same? (vit x32 #:output 'tokens) (reference-tokens vit x32))
                  (format "~a" on))
      (check-eq? (positions pos '(8 8) '(8 8)) pos)
      (define (calls thunk) (thunk) (op-calls-of thunk))
      (define native (calls (lambda () (vit x32 #:output 'tokens))))
      (define resized (calls (lambda () (vit x16 #:output 'tokens))))
      (define interpolation (calls (lambda () (positions pos '(8 8) '(4 4)))))
      (check-equal? (calls (lambda () (positions pos '(8 8) '(8 8)))) 0)
      (check-equal? (- resized native) interpolation
                    (format "~a: ~a native, ~a at 16 px" on native resized))
      (check-true (same? (positions pos '(8 8) '(4 4))
                         (twin-positions pos 8 8 '(4 4)))
                  (format "~a" on))))

  (bounded-test "V6: gradients reach pos-embed and cls-token, through the interpolation"
    (define vit (to (VisionTransformer #:image-size 4 #:patch 2 #:width 8
                                       #:depth 1 #:heads 2)
                    'float64))
    (define pos (parameter vit "pos-embed"))
    (define cls (parameter vit "cls-token"))
    (for ([side (in-list '(4 2))])
      (define x (to-dtype (randn 1 3 side side) 'float64))
      (check-true
       (gradcheck (lambda (p c)
                    (define swapped
                      (layer-rebuild vit
                                     (lambda (_name child) child)
                                     (lambda (t)
                                       (cond
                                         [(eq? t pos) p]
                                         [(eq? t cls) c]
                                         [else t]))))
                    (sum (swapped x #:output 'tokens)))
                  (list pos cls))
       (format "input side ~a" side))))

  (bounded-test "V7: modes, moves, copies and seeds"
    (define vit (small))
    (define x (randn 2 3 32 32))
    (define train-out (vit x))
    (eval! vit)
    (define once (vit x))
    (eval! vit)
    (check-equal? (layer-mode vit) 'eval)
    (check-true (same? (vit x) once))
    (check-true (same? train-out once))
    (train! vit)
    (define before (parameters vit))
    (to vit 'cpu)
    (for ([a (in-list before)] [b (in-list (parameters vit))])
      (check-eq? a b))
    (check-true (same? ((layer-copy vit) x) (vit x)))
    (define (seeded)
      (manual-seed! 5)
      (map (lambda (e) (tensor->list (cdr e))) (state-dict (small))))
    (check-equal? (seeded) (seeded)))

  (define (block-linears vit)
    (for*/list ([block (in-layers (child-ref vit "blocks"))]
                [path (in-list '(("self-attn" "query") ("self-attn" "key")
                                 ("self-attn" "value") ("self-attn" "out")
                                 ("linear1") ("linear2")))])
      (for/fold ([m block]) ([name (in-list path)]) (child-ref m name))))

  (bounded-test "V8: block initialisation"
    (define vit (VisionTransformer #:image-size 32 #:patch 4 #:width 192
                                   #:depth 9 #:heads 3))
    (define linears (block-linears vit))
    (check-equal? (length linears) 54)
    (for ([l (in-list linears)])
      (check-true (linear? l))
      (check-true (for/and ([v (in-flattened-tensor (parameter l "weight"))])
                    (<= -2.0 v 2.0)))
      (check-true (for/and ([v (in-flattened-tensor (parameter l "bias"))])
                    (zero? v))))
    (for* ([block (in-layers (child-ref vit "blocks"))]
           [norm (in-list (list (child-ref block "norm1")
                                (child-ref block "norm2")))])
      (check-true (andmap (lambda (v) (= v 1.0))
                          (tensor->list (parameter norm "weight"))))
      (check-true (andmap zero? (tensor->list (parameter norm "bias")))))
    (define-values (squares count)
      (for/fold ([squares 0.0] [count 0]) ([l (in-list linears)])
        (define w (parameter l "weight"))
        (values (+ squares (item (sum (* w w)))) (+ count (numel w)))))
    (check-= (sqrt (/ squares count)) 0.02 0.0005))

  (bounded-test "V9: malformed arguments are contract violations, blamed on the caller"
    (define vit (small))
    (define blames-this-test
      (message-matching #rx"blaming: [(][^)]*vit-test[.]rkt"))
    (define blames-caller (message-matching #rx"blaming: caller"))
    (for ([thunk (in-list
                  (list (lambda () (vit (randn 2 32 32)))
                        (lambda () (vit (randn 2 1 32 32)))
                        (lambda () (vit (randn 2 3 30 32)))
                        (lambda () (vit (to-dtype (randn 2 3 32 32) 'uint8)))
                        (lambda () (vit (randn 2 3 32 32) #:output 'logits))))])
      (check-exn blames-caller thunk))
    (check-exn (message-matching #rx"#:heads divides #:width")
               (lambda () (VisionTransformer #:width 10 #:heads 3)))
    (check-exn blames-this-test
               (lambda () (VisionTransformer #:width 10 #:heads 3)))
    (check-exn (message-matching #rx"#:patch divides each side of #:image-size")
               (lambda () (VisionTransformer #:image-size 30 #:patch 4)))
    (check-exn blames-this-test
               (lambda () (VisionTransformer #:image-size 30 #:patch 4)))
    (check-exn blames-this-test (lambda () (PatchEmbed 3 0 4)))
    (check-exn blames-this-test (lambda () (huggingface-vit-key 'cls))))

  (define (bytes-on on)
    (for/sum ([entry (in-list (native-memory-use))]
              #:when (equal? (car entry) on))
      (cdr entry)))

  (bounded-test "V10: a dropped forward leaves native memory at its baseline"
    (for ([on (in-list devices)])
      (define vit (to (small) on))
      (define x (to (randn 16 3 32 32) on))
      (vit x)
      (reclaim-native-memory!)
      (define before (bytes-on on))
      (vit x)
      (reclaim-native-memory!)
      (check-equal? (bytes-on on) before (format "~a" on))
      (check-equal? (shape (vit x #:output 'cls)) '(16 24)))))
