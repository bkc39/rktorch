#lang racket/base

;; The Phase-1/2 capstone smoke test: a real convnet trains on the committed
;; MNIST fixture and its cross-entropy loss falls over a few Adam steps, then
;; its parameters survive a safetensors save/load round-trip. This exercises
;; the whole stack end to end — data -> conv/pool/flatten/linear ->
;; cross-entropy -> backward! -> adam -> state-dict.
;;
;; conv2d names the nn *layer*; the functional max-pool2d/flatten/relu come
;; from `torch` (the F.* vs nn.* split), so each facade drops the names the
;; other owns — the model-file import pattern until issue #11 namespaces it.

(module+ test
  (require rackunit
           (only-in racket/list first last)
           (only-in racket/file make-temporary-file)
           (except-in "../main.rkt" conv2d)
           (except-in "../nn.rkt" max-pool2d flatten)
           (only-in "../data/mnist.rkt" load-mnist-fixture))

  ;; [N,1,28,28] -> conv(1->8,k3) -> relu -> maxpool2 -> flatten -> fc(1352,10).
  ;; 28 -3 +1 = 26 after conv; /2 = 13 after pool; 8*13*13 = 1352.
  (define-module convnet ()
    #:submodules ([c1 (conv2d 1 8 3)]
                  [fc (linear 1352 10)])
    #:forward (x)
    (fc (flatten (max-pool2d (relu (c1 x)) 2) 1)))

  (test-case "convnet trains on the MNIST fixture (loss decreases)"
    (manual-seed! 0)
    (define-values (imgs lbls) (load-mnist-fixture))
    (define net (convnet))
    (check-equal? (map car (named-parameters net))
                  '("c1.weight" "c1.bias" "fc.weight" "fc.bias"))
    (check-equal? (tensor-shape (net imgs)) '(256 10))
    (define opt (adam (parameters net) #:lr 0.01))
    (define losses
      (for/list ([_ (in-range 8)])
        (zero-grads! opt)
        (define loss (cross-entropy (net imgs) lbls))
        (backward! loss)
        (step! opt)
        (item loss)))
    (check-true (< (last losses) (first losses))
                (format "convnet loss did not decrease: ~a" losses)))

  (test-case "trained convnet round-trips through safetensors"
    (manual-seed! 1)
    (define-values (imgs lbls) (load-mnist-fixture))
    (define net (convnet))
    (define opt (adam (parameters net) #:lr 0.01))
    (for ([_ (in-range 3)])
      (zero-grads! opt)
      (backward! (cross-entropy (net imgs) lbls))
      (step! opt))
    (define path (make-temporary-file "convnet-~a.safetensors"))
    (save-state! net path)
    (define net2 (convnet))
    (load-state! net2 path)
    (for ([a (in-list (state-dict net))] [b (in-list (state-dict net2))])
      (check-equal? (car a) (car b))
      (check-equal? (tensor->list (cdr a)) (tensor->list (cdr b))))
    ;; same params => identical predictions
    (check-equal? (tensor->list (net imgs)) (tensor->list (net2 imgs)))
    (delete-file path)))
