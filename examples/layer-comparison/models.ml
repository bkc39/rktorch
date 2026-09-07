open Torch

let projection vs ~input_dim output_dim =
  let bound = 1. /. sqrt (float_of_int input_dim) in
  let weight =
    Var_store.new_var vs ~name:"weight" ~shape:[ output_dim; input_dim ]
      ~init:(Uniform (-.bound, bound))
  in
  let bias =
    Var_store.new_var vs ~name:"bias" ~shape:[ output_dim ]
      ~init:(Uniform (-.bound, bound))
  in
  Layer.of_fn (fun x ->
      Tensor.(matmul x (transpose weight ~dim0:0 ~dim1:1) + bias))

let channel_norm vs channels =
  let norm = Layer.layer_norm Var_store.(vs / "norm") channels in
  Layer.of_fn (fun x ->
      Tensor.permute x ~dims:[ 0; 2; 3; 1 ]
      |> Layer.forward norm
      |> Tensor.permute ~dims:[ 0; 3; 1; 2 ])

let residual_block vs ~input_dim output_dim ~stride =
  let conv1 =
    Layer.conv2d_
      Var_store.(vs / "conv1")
      ~input_dim output_dim ~ksize:3 ~stride ~padding:1
  in
  let norm1 = channel_norm Var_store.(vs / "norm1") output_dim in
  let conv2 =
    Layer.conv2d_
      Var_store.(vs / "conv2")
      ~input_dim:output_dim output_dim ~ksize:3 ~stride:1 ~padding:1
  in
  let norm2 = channel_norm Var_store.(vs / "norm2") output_dim in
  let shortcut =
    if stride = 1 && input_dim = output_dim then Layer.id
    else
      Layer.conv2d_
        Var_store.(vs / "shortcut")
        ~input_dim output_dim ~ksize:1 ~stride
  in
  Layer.of_fn (fun x ->
      let residual = Layer.forward shortcut x in
      let h = Layer.forward conv1 x |> Layer.forward norm1 |> Tensor.relu in
      let h = Layer.forward conv2 h |> Layer.forward norm2 in
      Tensor.(relu (h + residual)))

let residual_stage vs ~input_dim output_dim ~depth ~stride =
  let blocks =
    List.init depth (fun i ->
        residual_block
          Var_store.(vs / "blocks" // i)
          ~input_dim:(if i = 0 then input_dim else output_dim)
          output_dim
          ~stride:(if i = 0 then stride else 1))
  in
  Layer.of_fn (fun x ->
      List.fold_left (fun h block -> Layer.forward block h) x blocks)

let small_resnet vs ~classes =
  let stem =
    Layer.conv2d_
      Var_store.(vs / "stem")
      ~input_dim:3 16 ~ksize:3 ~stride:1 ~padding:1
  in
  let stage1 =
    residual_stage Var_store.(vs / "stage1") ~input_dim:16 16 ~depth:2 ~stride:1
  in
  let stage2 =
    residual_stage Var_store.(vs / "stage2") ~input_dim:16 32 ~depth:2 ~stride:2
  in
  let stage3 =
    residual_stage Var_store.(vs / "stage3") ~input_dim:32 64 ~depth:2 ~stride:2
  in
  let head = projection Var_store.(vs / "head") ~input_dim:64 classes in
  Layer.of_fn (fun images ->
      Layer.forward stem images |> Tensor.relu |> Layer.forward stage1
      |> Layer.forward stage2 |> Layer.forward stage3
      |> Tensor.mean_dim ~dim:(Some [ 2; 3 ]) ~keepdim:false ~dtype:(T Float)
      |> Layer.forward head)

let causal_self_attention vs ~width ~heads ~max_t ~dropout =
  if width <= 0 || heads <= 0 || width mod heads <> 0 then
    invalid_arg "width must be positive and divisible by heads";
  let head_dim = width / heads in
  let q = projection Var_store.(vs / "q") ~input_dim:width width in
  let k = projection Var_store.(vs / "k") ~input_dim:width width in
  let v = projection Var_store.(vs / "v") ~input_dim:width width in
  let out = projection Var_store.(vs / "out") ~input_dim:width width in
  let mask =
    Tensor.ones [ max_t; max_t ] ~device:(Var_store.device vs)
    |> Tensor.tril ~diagonal:0
    |> fun t ->
    Tensor.eq_scalar t (Scalar.int 0) |> fun src ->
    Var_store.new_var_copy vs ~name:"mask" ~trainable:false ~src
  in
  Layer.of_fn_ (fun x ~is_training ->
      let batch, time, _ = Tensor.shape3_exn x in
      let split_heads projection =
        Layer.forward projection x
        |> Tensor.reshape ~shape:[ batch; time; heads; head_dim ]
        |> Tensor.transpose ~dim0:1 ~dim1:2
      in
      let queries = split_heads q in
      let keys = split_heads k in
      let values = split_heads v in
      let scores =
        Tensor.(
          matmul queries (transpose keys ~dim0:2 ~dim1:3)
          / f (Stdlib.sqrt (float_of_int head_dim)))
      in
      let active_mask =
        Tensor.narrow mask ~dim:0 ~start:0 ~length:time
        |> Tensor.narrow ~dim:1 ~start:0 ~length:time
      in
      let weights =
        Tensor.masked_fill scores ~mask:active_mask
          ~value:(Scalar.float neg_infinity)
        |> Tensor.softmax ~dim:(-1) ~dtype:(T Float)
        |> Tensor.dropout ~p:dropout ~is_training
      in
      Tensor.matmul weights values
      |> Tensor.transpose ~dim0:1 ~dim1:2
      |> Tensor.reshape ~shape:[ batch; time; width ]
      |> Layer.forward out
      |> Tensor.dropout ~p:dropout ~is_training)

let feed_forward vs ~width ~dropout =
  let up = projection Var_store.(vs / "up") ~input_dim:width (4 * width) in
  let down = projection Var_store.(vs / "down") ~input_dim:(4 * width) width in
  Layer.of_fn_ (fun x ~is_training ->
      Layer.forward up x
      |> Tensor.gelu ~approximate:"none"
      |> Layer.forward down
      |> Tensor.dropout ~p:dropout ~is_training)

let transformer_block vs ~width ~heads ~max_t ~dropout =
  let norm1 = Layer.layer_norm Var_store.(vs / "norm1") width in
  let attention =
    causal_self_attention
      Var_store.(vs / "attention")
      ~width ~heads ~max_t ~dropout
  in
  let norm2 = Layer.layer_norm Var_store.(vs / "norm2") width in
  let mlp = feed_forward Var_store.(vs / "mlp") ~width ~dropout in
  Layer.of_fn_ (fun x ~is_training ->
      let a = Layer.forward norm1 x |> Layer.forward_ attention ~is_training in
      let h = Tensor.(x + a) in
      let m = Layer.forward norm2 h |> Layer.forward_ mlp ~is_training in
      Tensor.(h + m))

let transformer_stack vs ~width ~heads ~depth ~max_t ~dropout =
  let blocks =
    List.init depth (fun i ->
        transformer_block
          Var_store.(vs / "blocks" // i)
          ~width ~heads ~max_t ~dropout)
  in
  let norm = Layer.layer_norm Var_store.(vs / "norm") width in
  Layer.of_fn_ (fun tokens ~is_training ->
      List.fold_left
        (fun h block -> Layer.forward_ block h ~is_training)
        tokens blocks
      |> Layer.forward norm)

let demo () =
  let vision_vs = Var_store.create ~name:"vision" () in
  let vision = small_resnet vision_vs ~classes:10 in
  let images = Tensor.randn [ 2; 3; 32; 32 ] in
  let vision_opt = Optimizer.adam vision_vs ~learning_rate:1e-3 in
  let logits = Layer.forward vision images in
  Optimizer.backward_step vision_opt ~loss:Tensor.(mean (logits * logits));
  Printf.printf "SmallResNet %s\n" (Tensor.shape_str logits);
  let vs = Var_store.create ~name:"transformer" () in
  let net =
    transformer_stack vs ~width:32 ~heads:4 ~depth:2 ~max_t:16 ~dropout:0.1
  in
  let tokens = Tensor.randn [ 2; 8; 32 ] in
  let optimizer = Optimizer.adam vs ~learning_rate:1e-3 in
  let output = Layer.forward_ net tokens ~is_training:true in
  Optimizer.backward_step optimizer ~loss:Tensor.(mean (output * output));
  let prediction =
    Tensor.no_grad (fun () -> Layer.forward_ net tokens ~is_training:false)
  in
  Printf.printf "TransformerStack %s, %d parameter tensors\n"
    (Tensor.shape_str prediction)
    (Var_store.num_trainable_vars vs)
