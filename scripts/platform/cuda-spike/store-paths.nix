cpp-cuda:
let
  libtorch = builtins.head cpp-cuda.buildInputs;
  cudnn = builtins.head
    (builtins.filter (input: (input.pname or "") == "cudnn") libtorch.buildInputs);
  toolkit = builtins.head
    (builtins.filter (input: (input.pname or "") == "cuda-merged") cpp-cuda.buildInputs);
in
{
  shim = "${cpp-cuda}";
  libtorch = "${libtorch.src}";
  cudnn = "${cudnn.src}";
  cudnnVersion = cudnn.version;
  nvrtc = builtins.head libtorch.appendRunpaths;
  toolkit = "${toolkit}";
}
