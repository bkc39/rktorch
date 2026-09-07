open Torch

let int input = Scanf.bscanf input "%d " Fun.id
let word input = Scanf.bscanf input "%s " Fun.id
let float input = Scanf.bscanf input "%f " Fun.id

let tensor input =
  let rank = int input in
  let shape = List.init rank (fun _ -> int input) in
  let count = List.fold_left ( * ) 1 shape in
  let data = List.init count (fun _ -> float input) in
  Tensor.float_vec data |> Tensor.reshape ~shape

let check_close output expected =
  assert (Tensor.shape output = Tensor.shape expected);
  let actual = Tensor.reshape output ~shape:[ -1 ] |> Tensor.to_float1_exn in
  let reference =
    Tensor.reshape expected ~shape:[ -1 ] |> Tensor.to_float1_exn
  in
  let max_error = ref 0. in
  Array.iteri
    (fun i value ->
      let error = abs_float (value -. reference.(i)) in
      assert (
        Float.is_finite value
        && error <= 2e-5 +. (2e-4 *. abs_float reference.(i)));
      max_error := max !max_error error)
    actual;
  !max_error

let () =
  let input = Scanf.Scanning.open_in Sys.argv.(1) in
  for _ = 1 to int input do
    let name = word input in
    let vs = Var_store.create ~name () in
    let net =
      if name = "resnet" then
        Layer.with_training (Models.small_resnet vs ~classes:10)
      else
        Models.transformer_stack vs ~width:32 ~heads:4 ~depth:2 ~max_t:16
          ~dropout:0.1
    in
    let x = tensor input in
    let expected = tensor input in
    let count = int input in
    assert (Var_store.num_trainable_vars vs = count);
    let variables = Var_store.all_vars vs in
    Tensor.no_grad (fun () ->
        for _ = 1 to count do
          let path = word input in
          let src = tensor input in
          Tensor.copy_ (List.assoc path variables) ~src
        done);
    let output = Layer.forward_ net x ~is_training:false in
    let error = check_close output expected in
    let loss = Tensor.(mean (output * output)) in
    Tensor.backward loss;
    Var_store.iter_trainable_vars vs ~f:(fun _ p ->
        let gradient = Tensor.grad p in
        assert (Tensor.defined gradient);
        assert (Tensor.isfinite gradient |> Tensor.all |> Tensor.int_value = 1));
    if name = "transformer" then (
      assert (List.length variables = count + 2);
      let changed = Tensor.copy x in
      Tensor.no_grad (fun () ->
          let suffix = Tensor.narrow changed ~dim:1 ~start:4 ~length:4 in
          Tensor.copy_ suffix ~src:Tensor.(suffix + f 100.));
      let altered = Layer.forward_ net changed ~is_training:false in
      ignore
        (check_close
           (Tensor.narrow altered ~dim:1 ~start:0 ~length:4)
           (Tensor.narrow output ~dim:1 ~start:0 ~length:4)));
    let optimizer = Optimizer.adam vs ~learning_rate:1e-3 in
    let training_output = Layer.forward_ net x ~is_training:true in
    Optimizer.backward_step optimizer
      ~loss:Tensor.(mean (training_output * training_output));
    Printf.printf
      "%s OCaml/Racket output parity, gradients, Adam OK; max error %.9g\n" name
      error
  done;
  Scanf.Scanning.close_in input
