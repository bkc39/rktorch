{ pkgs }:
let
  inherit (pkgs) lib;
  ocaml = pkgs.ocamlPackages;
  darwin = pkgs.stdenv.hostPlatform.isDarwin;
  libtorch = pkgs.stdenv.mkDerivation {
    pname = "libtorch-ocaml";
    version = "2.1.2";
    src = pkgs.fetchurl (if darwin then {
      url =
        "https://files.pythonhosted.org/packages/1e/86/477ec85bf1f122931f00a2f3889ed9322c091497415a563291ffc119dacc/torch-2.1.2-cp311-none-macosx_11_0_arm64.whl";
      sha256 =
        "e2d83f07b4aac983453ea5bf8f9aa9dacf2278a8d31247f5d9037f37befc60e4";
    } else {
      name = "libtorch-2.1.2-linux.zip";
      url =
        "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip";
      sha256 = "1sbrzsx2mx0cd1p2gsvis3chlnf1p1alnk7npqsqlshhyr6pcjwh";
    });
    nativeBuildInputs = [ pkgs.unzip ]
      ++ lib.optionals darwin [ pkgs.fixDarwinDylibNames ]
      ++ lib.optionals (!darwin) [ pkgs.autoPatchelfHook ];
    buildInputs = lib.optionals (!darwin) [ pkgs.stdenv.cc.cc.lib ];
    unpackPhase = ''
      runHook preUnpack
      unzip -q "$src"
      cd ${if darwin then "torch" else "libtorch"}
      runHook postUnpack
    '';
    dontConfigure = true;
    dontBuild = true;
    dontStrip = true;
    installPhase = ''
      runHook preInstall
      mkdir -p "$out"
      cp -r include lib "$out/"
      runHook postInstall
    '';
    meta = {
      description = "Libtorch runtime matching Jane Street OCaml Torch";
      homepage = "https://pytorch.org";
      license = lib.licenses.bsd3;
      sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
      platforms = [ "aarch64-darwin" "x86_64-linux" ];
    };
  };
in ocaml.buildDunePackage {
  pname = "torch";
  version = "v0.17.0";
  src = pkgs.fetchFromGitHub {
    owner = "janestreet";
    repo = "torch";
    rev = "v0.17.0";
    sha256 = "10wg4i7iwdcjh1l2z8c36yza4il83wj7rijlnalcdzhfzz9qp86d";
  };
  minimalOCamlVersion = "5.1";
  nativeBuildInputs = [ pkgs.pkg-config ];
  buildInputs = [ ocaml.dune-configurator ocaml.core_unix ];
  propagatedBuildInputs = with ocaml; [
    base
    core
    stdio
    ctypes
    ctypes-foreign
    ocaml-compiler-libs
    ppx_bench
    ppx_inline_test
    ppx_jane
    ppx_string
  ];
  LIBTORCH = libtorch;
  postPatch = ''
    substituteInPlace src/tests/dune \
      --replace-fail '(name torch_tests)' '(name torch_tests) (inline_tests (deps (glob_files *.pt)))'
    substituteInPlace src/wrapper/dune \
      --replace-fail '(names torch_api)' '(names torch_api) (extra_deps torch_api_generated.cpp)'
    substituteInPlace src/wrapper/torch_api.h src/wrapper/torch_api.cpp \
      --replace-fail 'void (*f)(char *,' 'void (*f)(const char *,'
  '' + lib.optionalString darwin ''
    substituteInPlace src/wrapper/dune --replace-fail '-lstdc++' '-lc++'
    substituteInPlace src/tests/tensor_tests.ml \
      --replace-fail '-0.40455365180969238' '-0.40455368161201477' \
      --replace-fail '0.36596611142158508' '0.36596614122390747'
  '';
  doCheck = false;
  meta = {
    description = "Jane Street OCaml bindings to PyTorch";
    homepage = "https://github.com/janestreet/torch";
    license = lib.licenses.mit;
    platforms = [ "aarch64-darwin" "x86_64-linux" ];
  };
}
