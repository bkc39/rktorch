# PyTorch's own x86_64-linux wheels for the parity twins, at the libtorch
# version the shim links. The Racket side runs PyTorch's libtorch build, so the
# Python side has to as well: nixpkgs' from-source torch links a different
# BLAS, and a parity test that trains with Adam amplifies that rounding
# difference into a failure (#197).
#
# A python packageOverrides function, so every package in the set that depends
# on torch gets this one. With `cudaLibs` (library directories) it is the cu130
# build, patched against those directories: the CUDA libraries the Racket side
# loads, so a CUDA parity test runs the same ones on both sides. torchaudio's
# last release is 2.11; its wheel runs on torch 2.14 with its version pin
# unchecked.
{ lib, stdenv, fetchurl, autoPatchelfHook, ffmpeg-headless, libheif
, cudaLibs ? null }:

self: super:

let
  cuda = cudaLibs != null;
  variant = if cuda then "cu130" else "cpu";
  sha256 = {
    cpu = {
      torch = "0zhxpqn8xaipjngr33qz8nsahzmlxd8y90771pz2linwqlfz8lpi";
      torchvision = "1dq581s7wx9nf1ldq7n3c97h8r4pqf5i6af7gnzrxlgmzaamjwyx";
      torchaudio = "1m3kz24ca9p0jp8apwmr8aycvic3bwxzxd4pq00jnyp10kbxri9l";
    };
    cu130 = {
      torch = "0hi4qvm49h90gz6wgdky481s02mwx6axnv9s76b7ad4ikyg9zc0j";
      torchvision = "065fdha2vn2151i0m70b33as423jlmgnjkyhyhlvwzqla5644543";
      torchaudio = "16h1g0ffhwzmcfd6c4fzx8pqfl512a7r4h2xsai184aq3dklk2rp";
    };
  }.${variant};
  sitePackages = self.python.sitePackages;
  torchLibs = "${self.torch}/${sitePackages}/torch";
  ffmpegMajor = lib.versions.major ffmpeg-headless.version;

  wheel = { pname, version, url, sha256, dependencies, extra ? { } }:
    self.buildPythonPackage ({
      inherit pname version dependencies;
      format = "wheel";
      src = fetchurl { inherit url sha256; };
      nativeBuildInputs = [ autoPatchelfHook ];
      buildInputs = [ stdenv.cc.cc.lib ];
      preInstall = lib.optionalString (pname != "torch") ''
        addAutoPatchelfSearchPath "${torchLibs}"
      '';
      # stripping the wheels' libraries breaks their ELF load commands
      dontStrip = true;
      pythonImportsCheck = [ pname ];
    } // extra);

  # torch, torchvision and torchaudio, in the variant's build
  torchWheel = { pname, version, dependencies, extra ? { } }:
    wheel {
      inherit pname version dependencies;
      url = "https://download.pytorch.org/whl/${variant}/${pname}-${version}"
        + "%2B${variant}-cp314-cp314-manylinux_2_28_x86_64.whl";
      sha256 = sha256.${pname};
      extra = lib.optionalAttrs cuda {
        buildInputs = [ stdenv.cc.cc.lib ] ++ cudaLibs;
        # the host driver, found at run time
        autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
      } // extra;
    };
in
{
  torch = torchWheel {
    pname = "torch";
    version = "2.14.0";
    dependencies = with self; [
      filelock fsspec jinja2 networkx numpy pyyaml requests setuptools sympy
      typing-extensions
    ];
    extra = {
      postInstall = "rm -rf $out/bin";
      postFixup = ''
        addAutoPatchelfSearchPath "$out/${sitePackages}/torch/lib"
      '';
    } // lib.optionalAttrs cuda {
      # it lists the CUDA libraries as pip packages, which cudaLibs replace,
      # and triton and cuda-bindings, which only torch.compile and
      # cuda-python use
      dontCheckRuntimeDeps = true;
    };
  };

  torchvision = torchWheel {
    pname = "torchvision";
    version = "0.29.0";
    dependencies = with self; [ numpy pillow torch ];
  };

  torchaudio = torchWheel {
    pname = "torchaudio";
    version = "2.11.0";
    dependencies = with self; [ torch torchcodec ];
    # its metadata pins torch==2.11.0
    extra.dontCheckRuntimeDeps = true;
  };

  # torchaudio.load decodes through torchcodec. The wheel carries one core
  # library per FFmpeg major and loads the first that resolves; only the one
  # for nixpkgs' FFmpeg is kept, so every library left can be patched.
  torchcodec = wheel {
    pname = "torchcodec";
    version = "0.17.0";
    url = "https://download.pytorch.org/whl/cpu/torchcodec-0.17.0%2Bcpu-cp314-"
      + "cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    sha256 = "0skywsls4ffbxnn98a22y7cyf1wk6wjwr5mgaxn1zrrrqdslv5zl";
    dependencies = [ self.torch ];
    extra = {
      buildInputs = [
        stdenv.cc.cc.lib (lib.getLib ffmpeg-headless) (lib.getLib libheif)
      ];
      postInstall = ''
        for f in $out/${sitePackages}/torchcodec/libtorchcodec_{core,custom_ops}?.so; do
          case "$f" in
            *${ffmpegMajor}.so) ;;
            *) rm "$f" ;;
          esac
        done
      '';
    };
  };
}
