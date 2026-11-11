;; (op specs label): optional-argument paths the default recipes in
;; generated-parity-test.rkt leave absent, or vice versa; the label names the path.
((slice-tensor
  ((tensor 6) (int64 0) (optional-int64 #f) (optional-int64 #f)
   (int64 2))
  "[open]")
 (avg-pool2d
  ((tensor 1 1 4 4) (int-array (2 2)) (int-array (2 2))
   (int-array (0 0)) (bool #f) (bool #t) (optional-int64 2))
  "[divisor=2]")
 (nll-loss
  ((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor-ones 3)
   (int64 1) (int64 -100))
  "[weight]")
 (cross-entropy-loss
  ((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor-ones 3)
   (int64 1) (int64 -100) (double 0.0))
  "[weight]")
 (nll-loss
  ((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
   (int64 0) (int64 -100))
  "[none]")
 (cross-entropy-loss
  ((tensor 4 3) (int-tensor (0 2 1 0)) (optional-tensor #f)
   (int64 0) (int64 -100) (double 0.0))
  "[none]")
 (conv2d
  ((tensor 1 1 5 5) (tensor 2 1 3 3) (optional-tensor #f)
   (int-array (1 1)) (int-array (0 0)) (int-array (1 1)) (int64 1))
  "[no-bias]")
 (sum-dim-intlist
  ((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
  "[full]")
 (mean-dim
  ((tensor 2 3) (optional-int-array #f) (bool #f) (dtype #f))
  "[full]")
 (sum-dim-intlist
  ((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
  "[keepdim]")
 (mean-dim
  ((tensor 2 3) (optional-int-array (1)) (bool #t) (dtype #f))
  "[keepdim]")
 (layer-norm
  ((tensor 2 3) (int-array (3)) (optional-tensor #f)
   (optional-tensor #f) (double 1e-5) (bool #t))
  "[no-affine]")
 (embedding
  ((tensor 5 3) (int-tensor (0 2 4 1)) (int64 2) (bool #f) (bool #f))
  "[padding-idx=2]")
 (layer-norm
  ((tensor 2 2 3) (int-array (2 3)) (optional-tensor 2 3)
   (optional-tensor 2 3) (double 1e-5) (bool #t))
  "[multi-dim]")
 (tril
  ((tensor 4 4) (int64 -1))
  "[diag=-1]")
 (triu
  ((tensor 4 4) (int64 1))
  "[diag=1]")
 (conv-transpose2d-input
  ((tensor 1 2 4 4) (tensor 2 3 3 3) (optional-tensor #f)
   (int-array (1 1)) (int-array (0 0)) (int-array (0 0)) (int64 1)
   (int-array (1 1)))
  "[no-bias]")
 (conv-transpose2d-input
  ((tensor 1 2 4 4) (tensor 2 3 3 3) (optional-tensor 3)
   (int-array (2 2)) (int-array (0 0)) (int-array (1 1)) (int64 1)
   (int-array (1 1)))
  "[stride-2+output-padding]")
 (conv-transpose2d-input
  ((tensor 1 4 4 4) (tensor 4 3 3 3) (optional-tensor 6)
   (int-array (1 1)) (int-array (0 0)) (int-array (0 0)) (int64 2)
   (int-array (2 2)))
  "[groups-2+dilation-2]")
 (group-norm
  ((tensor 2 4 3 3) (int64 2) (optional-tensor #f) (optional-tensor #f)
   (double 1e-5) (bool #t))
  "[no-affine]")
 (group-norm
  ((tensor 2 4 3 3) (int64 1) (optional-tensor 4) (optional-tensor 4)
   (double 1e-5) (bool #t))
  "[one-group]")
 (clamp
  ((tensor 2 3) (optional-scalar -0.5) (optional-scalar #f))
  "[min-only]")
 (clamp
  ((tensor 2 3) (optional-scalar #f) (optional-scalar 0.5))
  "[max-only]")
 (repeat-interleave-self-int
  ((tensor 2 3) (int64 3) (optional-int64 #f) (kwarg "output_size" none))
  "[flattened]")
 (repeat-interleave-self-int
  ((tensor 2 3) (int64 3) (optional-int64 #f) (kwarg "output_size" 18))
  "[flattened+output-size]")
 (batch-norm
  ((tensor 2 3 4 4) (optional-tensor 3) (optional-tensor 3)
   (optional-tensor 3) (optional-tensor-ones 3) (bool #t) (double 0.1)
   (double 1e-5) (bool #t))
  "[training]")
 (batch-norm
  ((tensor 2 3 4 4) (optional-tensor #f) (optional-tensor #f)
   (optional-tensor #f) (optional-tensor #f) (bool #t) (double 0.1)
   (double 1e-5) (bool #t))
  "[training+no-stats+no-affine]")
 (leaky-relu
  ((tensor 2 3) (double 0.2))
  "[slope-0.2]")
 (binary-cross-entropy-with-logits
  ((tensor 2 3) (tensor 2 3) (optional-tensor 3) (optional-tensor-ones 3)
   (int64 2))
  "[weighted+sum]")
 (huber-loss
  ((tensor 2 3) (tensor 2 3) (int64 0) (double 0.5))
  "[none+delta-0.5]")
 (l1-loss
  ((tensor 2 3) (tensor 2 3) (int64 2))
  "[sum]")
 (flip
  ((tensor 2 3) (int-array (0 1)))
  "[both-dims]")
 (linear
  ((tensor 4 3) (tensor 2 3) (optional-tensor #f))
  "[no-bias]")
 (topk
  ((tensor 3 5) (int64 3) (int64 0) (bool #f) (bool #t))
  "[smallest-along-dim-0]")
 (sort-tensor
  ((tensor 3 5) (int64 0) (bool #t))
  "[descending-dim-0]")
 (argsort
  ((tensor 3 5) (int64 0) (bool #t))
  "[descending-dim-0]")
 (multinomial
  ((uniform-tensor 3 6) (int64 8) (bool #t) (kwarg "generator" none))
  "[replacement]")
 (lstm-input
  ((tensor 2 5 3) (tensors (4 2 4) (4 2 4))
   (tensors (16 3) (16 4) (16 3) (16 4)
            (16 8) (16 4) (16 8) (16 4))
   (bool #f) (int64 2) (double 0.0) (bool #f) (bool #t) (bool #t))
  "[2-layer-bidirectional-batch-first-no-bias]")
 (gru-input
  ((tensor 2 5 3) (tensor 4 2 4)
   (tensors (12 3) (12 4) (12 3) (12 4)
            (12 8) (12 4) (12 8) (12 4))
   (bool #f) (int64 2) (double 0.0) (bool #f) (bool #t) (bool #t))
  "[2-layer-bidirectional-batch-first-no-bias]")
 (scaled-dot-product-attention
  ((tensor 2 3 4) (tensor 2 5 4) (tensor 2 5 6) (optional-tensor #f)
   (double 0.0) (bool #f) (kwarg "scale" 0.5) (kwarg "enable_gqa" #f))
  "[scale=0.5]")
 (scaled-dot-product-attention
  ((tensor 2 3 4) (tensor 2 5 4) (tensor 2 5 6) (optional-tensor 3 5)
   (double 0.0) (bool #f) (kwarg "scale" none) (kwarg "enable_gqa" #f))
  "[float-mask]")
 (scaled-dot-product-attention
  ((tensor 2 3 4) (tensor 2 5 4) (tensor 2 5 6)
   (bool-tensor (1 0 1 1 0)) (double 0.0) (bool #f) (kwarg "scale" none)
   (kwarg "enable_gqa" #f))
  "[bool-mask]")
 (scaled-dot-product-attention
  ((tensor 2 5 4) (tensor 2 5 4) (tensor 2 5 6) (optional-tensor #f)
   (double 0.0) (bool #t) (kwarg "scale" none) (kwarg "enable_gqa" #f))
  "[causal]")
 (scaled-dot-product-attention
  ((tensor 2 4 3 8) (tensor 2 2 5 8) (tensor 2 2 5 8)
   (optional-tensor #f) (double 0.0) (bool #f) (kwarg "scale" none)
   (kwarg "enable_gqa" #t))
  "[grouped-query]")
 (grid-sampler-2d
  ((tensor 2 3 4 5) (tensor 2 6 7 2) (int64 1) (int64 1) (bool #t))
  "[nearest-border-aligned]")
 (grid-sampler-2d
  ((tensor 2 3 4 5) (tensor 2 6 7 2) (int64 2) (int64 2) (bool #f))
  "[bicubic-reflection]")
 (affine-grid-generator
  ((tensor 2 2 3) (int-array (2 3 4 5)) (bool #t))
  "[aligned]")
 (upsample-bicubic2d
  ((tensor 1 2 8 8) (int-array (4 4)) (bool #t) (kwarg "scales_h" none)
   (kwarg "scales_w" none))
  "[downscale-aligned]"))
