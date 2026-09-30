{
  description = "rktorch - Racket bindings to libtorch (PyTorch)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # Scoped ONLY to the Racket toolchain (#41): the main pin carries
    # Racket 9.2; this rev carries 9.3. cpp/libtorch/clang stay on the
    # main pin, so a Racket bump can never move the native stack.
    # Keep this rev AT LEAST as new as the main pin: the Racket binary
    # from here dlopens libtorchrkt built on the main pin, and glibc
    # symbol versioning is backward-compatible only in that direction
    # (older-built lib into newer-glibc process, never the reverse).
    nixpkgsRacket.url =
      "github:NixOS/nixpkgs/07e1d92cdc0ed416cfa11ff3ca40d17e61cfba7a";
  };

  outputs = { self, nixpkgs, nixpkgsRacket }:
    let
      # libtorch-bin ships only these two; the C++ side builds against it.
      supportedSystems = [ "aarch64-darwin" "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      version = "0.1.0";

      # The one knob that decides the PyTorch-parity story (see plans/):
      #   "bin"    -> pkgs.libtorch-bin: small prebuilt download, fast cached CI
      #               on both platforms; parity is tolerant (cross-test absorbs
      #               any patch-version drift vs the Python torch).
      #   "python" -> pkgs.python314Packages.torch: the SAME libtorch the parity
      #               script imports -> bit-exact randn, at the cost of a heavy
      #               (often uncached on darwin) from-source build.
      torchSource = "bin";

      # threading-lib and its dependency closure, prefetched as unpacked
      # source trees via a fixed-output derivation (network is permitted in
      # FODs) so the sandboxed racket build installs offline — the same
      # pattern as rkt-polars' racket-deps. Unpacked trees (not archive
      # zips) so the output hash is mtime-free and platform-stable. Bump
      # outputHash when the threading version in the catalog changes or a
      # new runtime dep lands in torch/info.rkt.
      # The Racket toolchain from the scoped pin, guarded by the ordering
      # invariant the nixpkgsRacket input comment documents: the Racket
      # binary dlopens libtorchrkt built against the MAIN pin's glibc, and
      # glibc symbol versioning only tolerates older-built-lib into
      # newer-glibc-process — so the scoped pin's glibc must be at least
      # as new. Enforced here (not just prose) so a re-pin in the wrong
      # direction fails at evaluation, symmetric with racket92's floor
      # assert. Darwin has no glibc; dyld versioning does not share the
      # constraint.
      racketFor = pkgs: pkgsRacket:
        assert pkgs.lib.assertMsg
          (!pkgs.stdenv.isLinux
           || pkgs.lib.versionAtLeast pkgsRacket.glibc.version
                pkgs.glibc.version)
          ("nixpkgsRacket glibc " + pkgsRacket.glibc.version
           + " is older than the main pin's " + pkgs.glibc.version
           + " — re-pin nixpkgsRacket at least as new as the main pin "
           + "(see the flake inputs comment)");
        pkgsRacket.racket;

      racketDepsFor = pkgs: racketPkg:
        pkgs.stdenvNoCC.mkDerivation {
          name = "torch-rkt-racket-deps";
          dontUnpack = true;
          nativeBuildInputs = [ racketPkg pkgs.cacert pkgs.unzip ];
          buildPhase = ''
            runHook preBuild
            export HOME=$TMPDIR/home
            export PLTUSERHOME=$TMPDIR/plt
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            mkdir -p "$PLTUSERHOME"
            raco pkg install --batch --auto --no-setup --scope user threading-lib
            mapfile -t deps < <(racket -e \
              '(require pkg/lib)(for ([p (installed-pkg-names #:scope (quote user))]) (displayln p))')
            raco pkg archive "$TMPDIR/archive" "''${deps[@]}"
            mkdir -p "$out"
            for z in "$TMPDIR"/archive/pkgs/*.zip; do
              name="$(basename "$z" .zip)"
              mkdir -p "$out/$name"
              unzip -q "$z" -d "$out/$name"
            done
            runHook postBuild
          '';
          dontInstall = true;
          outputHashMode = "recursive";
          outputHashAlgo = "sha256";
          outputHash = "sha256-7149ciHUXyTKKg+3KGRPKU9aTHrAA5gw5XWibaf1avw=";
        };

      # nixpkgs' libtorch-bin is still 2.9.0 (#140), so torchPackageFor points
      # its derivation at PyTorch's own 2.14.0 downloads. The darwin zip
      # bundles libomp.dylib behind @loader_path, so no Homebrew install name
      # needs rewriting any more.
      libtorchVersion = "2.14.0";
      libtorchZips = {
        aarch64-darwin-cpu = {
          name = "libtorch-macos-arm64-2.14.0.zip";
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.14.0.zip";
          hash = "sha256-cAQuRtcPkfApOlYri/pJ3oYvzKlpAt+QTbe/91J1BNk=";
        };
        x86_64-linux-cpu = {
          name = "libtorch-shared-with-deps-2.14.0-cpu.zip";
          url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.14.0%2Bcpu.zip";
          hash = "sha256-9trtRE/syydXBBkWdgYqZIkJZwzQmwr0AhE/NhBcHdw=";
        };
        x86_64-linux-cuda = {
          name = "libtorch-shared-with-deps-2.14.0-cu130.zip";
          url = "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.14.0%2Bcu130.zip";
          hash = "sha256-bFpQwpkKmPxORgCWP7rv1iTBq6p+dOys5B7ecHbLhnI=";
        };
      };

      # The 2.14 cu130 zip bundles only the libcudnn.so.9 dispatcher, none of
      # the libcudnn_* libraries it dlopens. This is the cuDNN PyTorch 2.14's
      # cu130 build expects, as one entry of NVIDIA's redistrib_9.24.0.json
      # (path and sha256 copied from it) fed to nixpkgs' own cuDNN derivation.
      cudnnFor = pkgs:
        pkgs.cudaPackages_13.cudnn.overrideAttrs (old: rec {
          passthru = old.passthru // {
            release = {
              version = "9.24.0.43";
              cuda_variant = [ "13" ];
              linux-x86_64.cuda13 = {
                relative_path =
                  "cudnn/linux-x86_64/cudnn-linux-x86_64-9.24.0.43_cuda13-archive.tar.xz";
                sha256 =
                  "63f1900222c69ee7e94583408181ccdb988dc2833531ce6bde0df43bbdd04a6d";
              };
            };
            supportedReleases.linux-x86_64 =
              passthru.release.linux-x86_64.cuda13;
          };
        });

      # The vendored ATen schema and the MPS kernels ctc-loss and GroupNorm
      # call directly both need libtorch >= libtorchVersion, so the python
      # route fails at evaluation, not on the first MPS call, while nixpkgs'
      # Python torch is older.
      torchPackageFor = pkgs:
        if torchSource == "python" then
          let torch = pkgs.python314Packages.torch;
          in
          assert pkgs.lib.assertMsg
            (pkgs.lib.versionAtLeast torch.version libtorchVersion)
            ("torchSource = \"python\" links torch " + torch.version
             + ", older than the libtorch " + libtorchVersion
             + " the vendored schema and the MPS kernels assume;"
             + " use torchSource = \"bin\" until nixpkgs' Python torch"
             + " reaches it");
          torch
        else
          let
            cuda = pkgs.config.cudaSupport;
            device = if cuda then "cuda" else "cpu";
          in
          pkgs.libtorch-bin.overrideAttrs (old: {
            version = libtorchVersion;
            src = pkgs.fetchzip
              libtorchZips."${pkgs.stdenv.hostPlatform.system}-${device}";
          } // pkgs.lib.optionalAttrs pkgs.stdenv.isDarwin {
            # The darwin zip's libomp, libshm, libtorch and libtorch_cpu carry
            # plain ad-hoc signatures (not linker-signed), which the
            # derivation's own `install_name_tool -id` postFixup invalidates
            # rather than re-signs. macOS 27 then SIGKILLs anything that loads
            # them (26 tolerated it), so re-sign every dylib after that edit.
            nativeBuildInputs = (old.nativeBuildInputs or [ ])
              ++ [ pkgs.darwin.sigtool ];
            postFixup = (old.postFixup or "") + ''
              for lib in $out/lib/*.dylib; do
                codesign -f -s - "$lib"
              done
            '';
          } // pkgs.lib.optionalAttrs cuda {
            # Drop the bundled dispatcher so autoPatchelf resolves
            # libcudnn.so.9 to cuDNN 9.24, whose own RUNPATH ($ORIGIN) finds
            # the libcudnn_* libraries it dlopens.
            buildInputs = (old.buildInputs or [ ]) ++ [ (cudnnFor pkgs) ];
            installPhase = old.installPhase + ''
              rm $out/lib/libcudnn.so.9
            '';
            # The bundled libnvrtc.so.13 is CUDA 13.0.88 and dlopens
            # libnvrtc-builtins.so.13.0, which the zip leaves out; this is the
            # matching nvrtc, appended to every library's RUNPATH.
            appendRunpaths = [
              "${pkgs.lib.getLib pkgs.cudaPackages_13_0.cuda_nvrtc}/lib"
            ];
          });

      # Stage libtorchrkt into ./torch/native-libs by temp file + rename(2).
      # `cp` opens the destination O_TRUNC, and that invalidates the page-cache
      # pages of every process still executing the old file — rewriting even
      # identical bytes faults a live REPL and wedges it, TERM-immune (#72).
      # rename swaps the directory entry and leaves the old inode alive, so
      # running processes keep the old code and new ones get the new lib.
      # Same discipline as torch/data/mnist.rkt's cache write.
      stageNativeLibs = src: ''
        _dest="$PWD/torch/native-libs"
        _stage_failed=0
        mkdir -p "$_dest"
        for _f in ${src}/lib/libtorchrkt.*; do
          _b=$(basename "$_f")
          # Chained: a shell hook has no errexit, so a partial copy must not
          # reach the rename and replace a good shim with a truncated one.
          # 0555 as the store ships it, which also makes an in-place `cp` fail
          # loudly with EACCES; no --no-preserve=mode keeps this POSIX.
          if cp -f "$_f" "$_dest/.$_b.tmp.$$" \
             && chmod 0555 "$_dest/.$_b.tmp.$$" \
             && mv -f "$_dest/.$_b.tmp.$$" "$_dest/$_b"; then
            :
          else
            rm -f "$_dest/.$_b.tmp.$$"
            echo "ERROR: staging $_b failed; leaving the existing shim in place" >&2
            _stage_failed=1
          fi
        done
      '';

      # Shell-entry staging, keyed to which shim is already staged rather than
      # to first-provision.  Without the key, `.#cuda` restaged on every entry
      # (the #72 vector, unprompted), while the default shell never restaged at
      # all — so a plain `nix develop` after a `.#cuda` visit kept running the
      # CUDA-linked shim, which needs the driver farm to even load.
      # Keyed on the bytes actually staged, so any other writer -- `nix run
      # .#copy-native-libs`, a hand copy, a deletion -- is noticed on the next
      # shell entry.  ~220 KB compare.
      stageNativeLibsIfStale = src: ''
        _stale=0
        for _f in ${src}/lib/libtorchrkt.*; do
          cmp -s "$_f" "$PWD/torch/native-libs/$(basename "$_f")" || _stale=1
        done
        if [ "$_stale" = 1 ]; then
          echo "Staging libtorchrkt (${src})..."
          ${stageNativeLibs src}
          if [ "$_stage_failed" != 0 ]; then
            echo "" >&2
            echo "  *** libtorchrkt was NOT staged.  The shim in torch/native-libs is" >&2
            echo "  *** stale or missing; racket in this shell will load the wrong one" >&2
            echo "  *** or fail to load at all.  Fix the error above and re-enter." >&2
            echo "" >&2
          fi
        else
          # Bytes match but an older checkout may have left 0644/0755 behind.
          chmod 0555 "$PWD"/torch/native-libs/libtorchrkt.* 2>/dev/null || true
        fi
      '';
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          # A second instance with the unfree CUDA stack enabled, used only by
          # the cpp-cuda output. cudaSupport flips libtorch-bin to the cu130
          # "shared-with-deps" download (a bundled binary, not a from-source
          # build), so this stays a download + patchelf, not a heavy compile.
          pkgsCuda = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };
          torch = torchPackageFor pkgs;
          # Racket 9.3 from the scoped pin; everything else stays on the
          # main pin (see the nixpkgsRacket input comment).
          pkgsRacket = import nixpkgsRacket { inherit system; };
          racketPkg = racketFor pkgs pkgsRacket;
          racket-deps = racketDepsFor pkgs racketPkg;

          cppCommonInputs = [ torch pkgs.gtest pkgs.libsndfile pkgs.stb ];
          cppNativeInputs = [ pkgs.cmake pkgs.clang-tools pkgs.ninja pkgs.pkg-config ];
          cppCmakeFlags = [
            "-DBUILD_TESTING=ON"
            "-DCMAKE_CXX_STANDARD=20"
          ];

          # Build the C++ shim against a given package set's libtorch. `cpp`
          # links the CPU libtorch-bin and runs its gtests in the sandbox;
          # `cpp-cuda` links the CUDA libtorch from pkgsCuda with checks off —
          # its gtest binary needs the host NVIDIA driver (libcuda.so.1), absent
          # in the build sandbox, so GPU verification runs on the host instead
          # (the device tests self-skip the CUDA cases without a real device).
          # The CUDA libtorch's Caffe2 CMake config refuses to configure unless
          # it can find a CUDA toolkit (even though the runtime libs are bundled
          # in the cu130 download), so the cuda variant adds the matching
          # cudaPackages_13 toolkit and points legacy/modern FindCUDA at it.
          mkCpp = { p, doCheck ? true, cuda ? false }:
            let cudaTk = p.cudaPackages_13.cudatoolkit;
            in p.stdenv.mkDerivation {
              pname = "torchrkt-cpp";
              inherit version doCheck;
              src = ./cpp;
              nativeBuildInputs = [ p.cmake p.clang-tools p.ninja p.pkg-config ]
                ++ p.lib.optional cuda p.cudaPackages_13.cuda_nvcc;
              buildInputs = [ (torchPackageFor p) p.gtest p.libsndfile p.stb ]
                ++ p.lib.optional cuda cudaTk;
              cmakeFlags = cppCmakeFlags ++ p.lib.optionals cuda [
                "-DCUDA_TOOLKIT_ROOT_DIR=${cudaTk}"
                "-DCUDAToolkit_ROOT=${cudaTk}"
              ];
              checkPhase = ''
                runHook preCheck
                # Diagnostic: if the libtorch-linked binary can't start on this
                # host (GitHub's virtualized macOS runners abort at startup),
                # surface the dyld/runtime error that gtest discovery swallows.
                ./torchrkt_tests --gtest_list_tests \
                  || echo "torchrkt_tests cannot run here (exit $?)"
                ctest --output-on-failure
                runHook postCheck
              '';
            };
          cpp = mkCpp { p = pkgs; };
          cpp-cuda = mkCpp {
            p = pkgsCuda;
            doCheck = false;
            cuda = true;
          };

          cpp-format = pkgs.stdenv.mkDerivation {
            pname = "torchrkt-cpp-format";
            inherit version;
            src = ./cpp;
            nativeBuildInputs = cppNativeInputs;
            buildInputs = cppCommonInputs;
            cmakeFlags = cppCmakeFlags;
            buildPhase = ''
              runHook preBuild
              cmake --build . --target format-check
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              touch $out
              runHook postInstall
            '';
          };

          cpp-tidy = pkgs.stdenv.mkDerivation {
            pname = "torchrkt-cpp-tidy";
            inherit version;
            src = ./cpp;
            nativeBuildInputs = cppNativeInputs;
            buildInputs = cppCommonInputs;
            cmakeFlags = cppCmakeFlags;
            buildPhase = ''
              runHook preBuild
              cmake --build . --target tidy
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              touch $out
              runHook postInstall
            '';
          };

          cpp-line-count = pkgs.stdenv.mkDerivation {
            pname = "torchrkt-cpp-line-count";
            inherit version;
            src = ./cpp;
            dontConfigure = true;
            dontBuild = true;
            installPhase = ''
              runHook preInstall
              failed=0
              while IFS= read -r file; do
                lines=$(wc -l < "$file")
                if [ "$lines" -gt 500 ]; then
                  echo "ERROR: $file has $lines lines; limit is 500" >&2
                  failed=1
                fi
              # generated/ shards are exempt: their size is the generator's
              # concern, not a hand-maintainability gate.
              done < <(find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) -not -path '*/generated/*')
              if [ "$failed" -ne 0 ]; then
                exit 1
              fi
              touch $out
              runHook postInstall
            '';
          };

          # One builder, two Racket versions: `racket` (the 9.3 default from
          # the scoped pin) and `racket92` (the previous version from the main
          # pin — the supported floor). Both live in `checks`, so every
          # `nix flake check` — locally and in each CI cell — exercises both;
          # racket-deps is shared (the prefetched package sources are
          # version-independent).
          mkRacketPackage = pname: racketPkg: pkgs.stdenv.mkDerivation {
            inherit pname version;
            src = ./.;

            nativeBuildInputs = [ racketPkg pkgs.makeWrapper ];
            buildInputs = [ cpp ];

            buildPhase = ''
              runHook preBuild

              export PLTUSERHOME=$TMPDIR/racket-home
              export TORCHRKT_NATIVE_LIB_PATH=${cpp}
              mkdir -p $PLTUSERHOME

              # Runtime deps (threading-lib + closure) install offline from
              # the prefetched source trees; the sandbox has no network.
              raco pkg install --batch --copy --no-docs --no-setup --scope user \
                ${racket-deps}/*/

              # Stage the native lib so define-runtime-path resolves it during
              # testing.  libtorch itself is reached via the rpath Nix baked
              # into libtorchrkt, so it is NOT copied (it is multi-GB).
              ${stageNativeLibs cpp}
              [ "$_stage_failed" = 0 ] || exit 1

              raco pkg install --batch --deps fail --no-setup --copy --scope user \
                --name torch ./torch

              raco setup --no-docs --pkgs torch

              runHook postBuild
            '';

            doCheck = true;
            checkPhase = ''
              runHook preCheck
              # python-cross-test self-skips when python3 `torch` is absent.
              raco test ./torch/
              # Each examples/racket/NN-name.rkt is a literate scribble/lp2
              # program; its runner + RackUnit checks live in examples/test/.
              raco test examples/test/
              runHook postCheck
            '';

            installPhase = ''
              runHook preInstall

              mkdir -p $out/share $out/bin
              cp -r $PLTUSERHOME $out/share/racket-home

              makeWrapper ${racketPkg}/bin/racket $out/bin/torch \
                --set PLTUSERHOME $out/share/racket-home \
                --add-flags "-l torch"

              runHook postInstall
            '';
          };

          racket = mkRacketPackage "torch-rkt" racketPkg;
          # The floor check is only meaningful while the main pin actually
          # carries 9.2: this assertion trips loudly when a main-pin update
          # moves pkgs.racket, forcing a deliberate new-floor decision
          # (re-point a scoped input at the old rev, or advance the floor —
          # see #41/#50) instead of silently testing 9.3 twice.
          racket92 = assert pkgs.lib.assertMsg
            (pkgs.lib.versions.majorMinor pkgs.racket.version == "9.2")
            ("racket92 floor check: the main pin's racket is now "
             + pkgs.racket.version
             + ", not 9.2 — re-point the supported floor (see #41/#50)");
            mkRacketPackage "torch-rkt-racket92" pkgs.racket;

          copy-native-libs = pkgs.writeShellApplication {
            name = "copy-native-libs";
            # The app must run on a bare host, not only inside `nix develop`:
            # without this it inherits the caller's PATH.
            runtimeInputs = [ pkgs.coreutils ];
            text = ''
              ${stageNativeLibs cpp}
              [ "$_stage_failed" = 0 ] || exit 1
              echo "Native library staged to $PWD/torch/native-libs"
              ls -la "$PWD/torch/native-libs"
            '';
          };

          # The ATen generator (`nix run .#codegen`): python3 with torchgen
          # (from the python torch wheel) + the pinned clang-format the
          # generator formats its C++ output with. Writes into the working
          # tree, so it must run from the repo root — much lighter than the
          # full dev shell when all you need is regeneration.
          codegen = pkgs.writeShellApplication {
            name = "codegen";
            runtimeInputs = [
              (pkgs.python314.withPackages (ps: [ ps.torch ]))
              pkgs.clang-tools
            ];
            text = ''
              if [ ! -f codegen/generate.py ]; then
                echo "codegen: run from the repo root (codegen/ not found)" >&2
                exit 1
              fi
              exec python3 -m codegen "$@"
            '';
          };
        in
        {
          default = racket;
          inherit cpp cpp-format cpp-line-count cpp-tidy racket
            racket92 racket-deps codegen copy-native-libs;
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          # The CUDA libtorch-bin has no darwin download, so even evaluating
          # this output aborts `nix flake check` on aarch64-darwin.
          inherit cpp-cuda;
        });

      apps = forAllSystems (system: {
        codegen = {
          type = "app";
          program = "${self.packages.${system}.codegen}/bin/codegen";
        };
        copy-native-libs = {
          type = "app";
          program = "${self.packages.${system}.copy-native-libs}/bin/copy-native-libs";
        };
      });

      checks = forAllSystems (system: {
        inherit (self.packages.${system})
          cpp cpp-format cpp-line-count cpp-tidy racket racket92;
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          torch = torchPackageFor pkgs;
          pkgsRacket = import nixpkgsRacket { inherit system; };
          racketPkg = racketFor pkgs pkgsRacket;
          racket-deps = racketDepsFor pkgs racketPkg;
          cpp = self.packages.${system}.cpp;
          cpp-cuda = self.packages.${system}.cpp-cuda;
          pkgsCuda = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          };
          # The cuDNN the CUDA libtorch links (see cudnnFor); conv/pool ops
          # dlopen its libcudnn_*.so.9 libraries by soname at runtime.
          cudaCudnn = cudnnFor pkgsCuda;

          # Python with PyTorch, for interactive parity work
          # (`nix develop --command python3`) and the python-cross-test. On
          # x86_64-linux it is PyTorch's own wheels at libtorchVersion, the same
          # build the shim links (nix/torch-wheels.nix, #197). Darwin keeps
          # nixpkgs' torch until the macOS wheels have been tried on a Mac.
          pythonWheels = cudaLibs:
            let
              python = pkgs.python314.override {
                self = python;
                packageOverrides = import ./nix/torch-wheels.nix {
                  inherit (pkgs) lib stdenv fetchurl autoPatchelfHook
                    ffmpeg-headless libheif;
                  inherit cudaLibs;
                };
              };
            in
            assert pkgs.lib.assertMsg
              (python.pkgs.torch.version == libtorchVersion)
              ("nix/torch-wheels.nix pins torch " + python.pkgs.torch.version
               + " but the shim links libtorch " + libtorchVersion
               + "; move the wheels with it");
            python.withPackages
              (ps: [ ps.soundfile ps.torch ps.torchaudio ps.torchvision ]);
          pythonEnv =
            if system == "x86_64-linux" then pythonWheels null
            else pkgs.python314.withPackages
              (ps: [ ps.soundfile ps.torch ps.torchaudio ps.torchvision ]);

          # The `.#cuda` shell's Python: the cu130 wheels, linked against the
          # CUDA libraries the CUDA libtorch bundles and its cuDNN 9.24, so both
          # sides of a CUDA parity test load the same ones.
          cudaRuntime = pkgs.runCommand "libtorch-cuda-runtime" { } ''
            mkdir -p $out/lib
            for f in ${torchPackageFor pkgsCuda}/lib/*.so*; do
              case "''${f##*/}" in
                libtorch*|libc10*|libcaffe2*|libshm*|libgomp*) ;;
                *) ln -s "$f" "$out/lib/" ;;
              esac
            done
          '';
          pythonCudaEnv =
            pythonWheels [ cudaRuntime (pkgs.lib.getLib cudaCudnn) ];

          ocamlTorch = import ./nix/ocaml-torch.nix { inherit pkgs; };
          ocamlInputs = with pkgs.ocamlPackages; [
            ocaml dune_3 findlib utop ocaml-lsp ocamlformat ocamlTorch
          ];

          baseInputs = [
            pkgs.cmake
            pkgs.clang-tools
            pkgs.gtest
            pkgs.libsndfile
            pkgs.ninja
            pkgs.stb
            pkgs.pkg-config
            racketPkg
            torch
            pkgs.stdenv.cc
          ];

          # Parameterised by the shim this shell wants.  Exactly one staging
          # call per entry.
          provisionRacketFor = shim: ''
            export TORCHRKT_NATIVE_LIB_PATH="${shim}"
            export PLTUSERHOME="$PWD/.racket-user"
            _rkt_ver=$(racket --version 2>&1 | grep -oE 'v[0-9]+\.[0-9]+' | tr -d 'v' | tr '.' '-')
            # Bump the ordinal whenever the installed package list below
            # changes: the stamp is what makes provisioning a one-time cost,
            # so an already-provisioned checkout would otherwise skip the new
            # package and only fail later, where it is used. (deps3: cover-lib)
            deps_stamp="$PLTUSERHOME/.deps3-installed-torch-''${_rkt_ver}"
            # In-tree zo caches compiled piecewise across commits can defeat
            # the compilation manager, so bytecode is keyed to HEAD by a
            # stamp-and-clear (a per-rev PLTCOMPILEDROOTS would recompile
            # the copied dep packages on every pull).
            _rev=$(git rev-parse HEAD 2>/dev/null || echo norev)
            _rev_stamp="$PLTUSERHOME/.provisioned-rev"
            _old_rev=$(cat "$_rev_stamp" 2>/dev/null || echo none)
            if [ "$_old_rev" != "$_rev" ]; then
              _clear_ok=1
              if [ "$_old_rev" != "none" ] || [ -f "$deps_stamp" ]; then
                echo "bytecode cache: clearing compiled/ (''${_old_rev:0:12} -> ''${_rev:0:12})"
                for _d in torch examples scripts codegen; do
                  [ -d "$_d" ] || continue
                  find "$_d" -type d -name compiled -prune -exec rm -rf {} + \
                    || _clear_ok=0
                done
              else
                echo "bytecode cache: fresh for ''${_rev:0:12}"
              fi
              if [ "$_clear_ok" = 1 ]; then
                mkdir -p "$PLTUSERHOME"
                echo "$_rev" > "$_rev_stamp"
              else
                echo "WARNING: stale bytecode not fully cleared; will retry on next shell entry" >&2
              fi
            fi
            ${stageNativeLibsIfStale shim}
            if [ ! -f "$deps_stamp" ]; then
              echo "Installing Racket package (link mode, Racket ''${_rkt_ver})..."
              mkdir -p "$PLTUSERHOME"
              # A shell hook has no errexit, so every step is chained: the
              # stamp is what makes provisioning a one-time cost, and stamping
              # a half-provisioned checkout leaves it broken until someone
              # deletes the stamp by hand. The dev tools come unpinned from the
              # live catalog, cover-lib beside the linters; it rides along when
              # #63 pins them, and it cannot join the racket-deps FOD as things
              # stand, since a fixed-output derivation may not reference store
              # paths and the doc packages in that closure embed them.
              if raco pkg install --batch --copy --no-docs --no-setup \
                     --scope user --skip-installed ${racket-deps}/*/ \
                 && raco pkg install --batch --auto --no-setup --link \
                      --scope user --skip-installed --name torch "$PWD/torch" \
                 && raco setup --no-docs --pkgs torch \
                 && { echo "Installing Racket dev tools (Resyntax + racket-review + cover)..."
                      raco pkg install --batch --auto --scope user \
                        --skip-installed resyntax review cover-lib; }; then
                touch "$deps_stamp"
              else
                echo "WARNING: provisioning failed; not stamping, so the next" >&2
                echo "         shell entry retries it." >&2
              fi
              echo "Done. Lint: resyntax analyze --local-git-repository . origin/master"
              echo "      full sweep: resyntax analyze --directory torch  |  raco review <files>"
              echo "      coverage:   racket scripts/coverage.rkt"
            fi
            export PATH="$(racket -e '(require setup/dirs)(display (path->string (find-user-console-bin-dir)))'):$PATH"
          '';

          # The CUDA verification shell: after the normal provisioning, swap the
          # CPU native lib for the CUDA-linked one and expose the host NVIDIA
          # driver. The nix libtorch's autoAddDriverRunpath points at
          # /run/opengl-driver/lib (a NixOS path absent on this Ubuntu host), so
          # we put just libcuda.so.1 / libnvidia-ml.so.1 on LD_LIBRARY_PATH —
          # only the driver libs, so nix's own libs (glibc, libstdc++) are not
          # shadowed by the system copies. libcuda dlopens its PTX JIT,
          # libnvidia-ptxjitcompiler.so.1, whenever a kernel ships as PTX (some
          # of cuFFT's do on sm_86), and nix's glibc never reads the host's
          # ld.so.cache, so the JIT goes in the farm too; without it stft on
          # CUDA fails with CUFFT_INTERNAL_ERROR (#180). Run:
          #   nix develop .#cuda --command raco test torch/tests/device-test.rkt
          # Driver farm only; `provisionRacketFor cpp-cuda` stages the shim and
          # points TORCHRKT_NATIVE_LIB_PATH at it.
          cudaHook = ''
            echo "Staging host NVIDIA driver farm..."
            _drv_farm="$PWD/.cuda-driver"
            rm -rf "$_drv_farm"; mkdir -p "$_drv_farm"
            for _l in libcuda.so.1 libnvidia-ml.so.1 libnvidia-ptxjitcompiler.so.1; do
              # Match the lib name as a fixed string (its dots are ERE
              # metacharacters), then take the path field of that ldconfig line.
              _p=$(/sbin/ldconfig -p 2>/dev/null \
                | grep -F "$_l" | grep -oE '/[^ ]+' | head -1)
              if [ -n "$_p" ]; then
                ln -sf "$_p" "$_drv_farm/$_l"
              else
                echo "WARNING: $_l not found via ldconfig; CUDA calls may fail" >&2
              fi
            done
            # Driver farm first (host libcuda), then the cuDNN lib dir — conv/pool
            # dlopen libcudnn_*.so.9 by soname, and the autoAddDriverRunpath
            # doesn't cover that. (matmul and friends work without it; only the
            # cuDNN-backed ops need it.)
            export LD_LIBRARY_PATH="$_drv_farm:${pkgs.lib.getLib cudaCudnn}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            # The cu130 wheels (pythonCudaEnv) find their CUDA libraries and
            # cuDNN through RUNPATH and need only the host driver; the
            # cross-test pins the python child's LD_LIBRARY_PATH to just this
            # farm when it's set.
            export RKTORCH_CUDA_DRIVER_PATH="$_drv_farm"
            echo "CUDA shell ready. Verify:"
            echo "  raco test torch/tests/device-test.rkt"
          '';
        in
        {
          # Full interactive shell. `nix develop` (or `nix develop --command
          # python3`) has the Python `torch` on PATH, so you can explore
          # PyTorch behaviour beside the Racket bindings and run the parity
          # cross-test for real:
          #   raco test torch/tests/python-cross-test.rkt
          default = pkgs.mkShell {
            buildInputs = baseInputs ++ [ pythonEnv ];
            shellHook = provisionRacketFor cpp;
          };

          ocaml = pkgs.mkShell {
            buildInputs = baseInputs ++ [ pythonEnv ] ++ ocamlInputs;
            shellHook = provisionRacketFor cpp;
          };

          # Lean shell without Python torch, used by the Resyntax CI lint job so
          # it doesn't pull torch's closure just to run the linter.
          ci = pkgs.mkShell {
            buildInputs = baseInputs;
            shellHook = provisionRacketFor cpp;
          };
        }
        # GPU verification shell: provisions Racket as usual, then stages the
        # CUDA-linked native lib and the host driver (see cudaHook). The device
        # tests' CUDA cases run for real here on an NVIDIA host; on a CPU-only
        # box they self-skip. Linux-only — it stages the cu130 CUDA libtorch and
        # the host driver. Omitted on non-Linux (rather than a `throw`, which
        # would abort `nix flake check`'s eval of every devShell on darwin) so
        # `nix develop .#cuda` there reports a plain "no such attribute".
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          cuda = pkgs.mkShell {
            buildInputs = baseInputs ++ [ pythonCudaEnv ];
            shellHook = provisionRacketFor cpp-cuda + cudaHook;
          };
        });
    };
}
