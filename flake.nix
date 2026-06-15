{
  description = "rktorch - Racket bindings to libtorch (PyTorch)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
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
      racketDepsFor = pkgs:
        pkgs.stdenvNoCC.mkDerivation {
          name = "torch-rkt-racket-deps";
          dontUnpack = true;
          nativeBuildInputs = [ pkgs.racket pkgs.cacert pkgs.unzip ];
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

      torchPackageFor = pkgs:
        if torchSource == "python" then pkgs.python314Packages.torch
        # nixpkgs' libtorch-bin on darwin leaves a Homebrew install name for
        # OpenMP inside libtorch_cpu.dylib, so anything linking it aborts at
        # dyld load on machines without Homebrew's libomp (e.g. GitHub's macOS
        # runners) and silently depends on Homebrew everywhere else. Rewrite
        # the reference to the nix-provided libomp and ad-hoc re-sign (the
        # edit invalidates the auto-signature applied earlier in fixup).
        else if pkgs.stdenv.isDarwin then
          pkgs.libtorch-bin.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ])
              ++ [ pkgs.darwin.cctools pkgs.darwin.sigtool ];
            postFixup = (old.postFixup or "") + ''
              for lib in $out/lib/*.dylib; do
                if otool -L "$lib" | grep -q '/opt/homebrew/opt/libomp/lib/libomp.dylib'; then
                  install_name_tool -change \
                    /opt/homebrew/opt/libomp/lib/libomp.dylib \
                    ${pkgs.llvmPackages.openmp}/lib/libomp.dylib "$lib"
                  codesign -f -s - "$lib"
                fi
              done
            '';
          })
        else pkgs.libtorch-bin;
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
          racket-deps = racketDepsFor pkgs;

          cppCommonInputs = [ torch pkgs.gtest ];
          cppNativeInputs = [ pkgs.cmake pkgs.clang-tools pkgs.ninja ];
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
              nativeBuildInputs = [ p.cmake p.clang-tools p.ninja ]
                ++ p.lib.optional cuda p.cudaPackages_13.cuda_nvcc;
              buildInputs = [ (torchPackageFor p) p.gtest ]
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

          racket = pkgs.stdenv.mkDerivation {
            pname = "torch-rkt";
            inherit version;
            src = ./.;

            nativeBuildInputs = [ pkgs.racket pkgs.makeWrapper ];
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
              mkdir -p ./torch/native-libs
              cp ${cpp}/lib/libtorchrkt.* ./torch/native-libs/

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

              makeWrapper ${pkgs.racket}/bin/racket $out/bin/torch \
                --set PLTUSERHOME $out/share/racket-home \
                --add-flags "-l torch"

              runHook postInstall
            '';
          };

          copy-native-libs = pkgs.writeShellApplication {
            name = "copy-native-libs";
            text = ''
              DEST="$(pwd)/torch/native-libs"
              mkdir -p "$DEST"
              cp -v --no-preserve=mode ${cpp}/lib/libtorchrkt.* "$DEST/"
              echo "Native library copied to $DEST"
              ls -la "$DEST"
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
          inherit cpp cpp-cuda cpp-format cpp-line-count cpp-tidy racket
            racket-deps codegen copy-native-libs;
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
          cpp cpp-format cpp-line-count cpp-tidy racket;
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          torch = torchPackageFor pkgs;
          racket-deps = racketDepsFor pkgs;
          cpp = self.packages.${system}.cpp;
          cpp-cuda = self.packages.${system}.cpp-cuda;

          # Python with the PyTorch wheel/lib, for interactive parity work
          # (`nix develop --command python3`) and the python-cross-test.  Cached
          # on both supported systems (a ~50 MiB fetch, not a source build).
          pythonEnv = pkgs.python314.withPackages (ps: [ ps.torch ]);

          baseInputs = [
            pkgs.cmake
            pkgs.clang-tools
            pkgs.gtest
            pkgs.ninja
            pkgs.racket
            torch
            pkgs.stdenv.cc
          ];

          provisionRacket = ''
            export TORCHRKT_NATIVE_LIB_PATH="${cpp}"
            export PLTUSERHOME="$PWD/.racket-user"
            _rkt_ver=$(racket --version 2>&1 | grep -oE 'v[0-9]+\.[0-9]+' | tr -d 'v' | tr '.' '-')
            deps_stamp="$PLTUSERHOME/.deps2-installed-torch-''${_rkt_ver}"
            if [ ! -f "$deps_stamp" ]; then
              echo "Installing Racket package (link mode, Racket ''${_rkt_ver})..."
              mkdir -p "$PLTUSERHOME"
              raco pkg install --batch --copy --no-docs --no-setup --scope user --skip-installed \
                ${racket-deps}/*/
              mkdir -p ./torch/native-libs
              cp ${cpp}/lib/libtorchrkt.* ./torch/native-libs/ 2>/dev/null || true
              raco pkg install --batch --auto --no-setup --link --scope user --skip-installed \
                --name torch "$PWD/torch"
              raco setup --no-docs --pkgs torch
              echo "Installing Racket linters (Resyntax + racket-review)..."
              raco pkg install --batch --auto --scope user --skip-installed \
                resyntax review
              touch "$deps_stamp"
              echo "Done. Lint: resyntax analyze --directory torch  |  raco review <files>"
            fi
            export PATH="$(racket -e '(require setup/dirs)(display (path->string (find-user-console-bin-dir)))'):$PATH"
          '';

          # The CUDA verification shell: after the normal provisioning, swap the
          # CPU native lib for the CUDA-linked one and expose the host NVIDIA
          # driver. The nix libtorch's autoAddDriverRunpath points at
          # /run/opengl-driver/lib (a NixOS path absent on this Ubuntu host), so
          # we put just libcuda.so.1 / libnvidia-ml.so.1 on LD_LIBRARY_PATH —
          # only the driver libs, so nix's own libs (glibc, libstdc++) are not
          # shadowed by the system copies. Run:
          #   nix develop .#cuda --command raco test torch/tests/device-test.rkt
          cudaHook = ''
            echo "Staging CUDA libtorchrkt + host NVIDIA driver..."
            mkdir -p ./torch/native-libs
            cp -f --no-preserve=mode ${cpp-cuda}/lib/libtorchrkt.* \
              ./torch/native-libs/
            _drv_farm="$PWD/.cuda-driver"
            rm -rf "$_drv_farm"; mkdir -p "$_drv_farm"
            for _l in libcuda.so.1 libnvidia-ml.so.1; do
              _p=$(/sbin/ldconfig -p 2>/dev/null \
                | grep -oE "/[^ ]*/$_l" | head -1)
              if [ -n "$_p" ]; then ln -sf "$_p" "$_drv_farm/$_l"; fi
            done
            export LD_LIBRARY_PATH="$_drv_farm''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
            shellHook = provisionRacket;
          };

          # Lean shell without Python torch, used by the Resyntax CI lint job so
          # it doesn't pull torch's closure just to run the linter.
          ci = pkgs.mkShell {
            buildInputs = baseInputs;
            shellHook = provisionRacket;
          };

          # GPU verification shell: provisions Racket as usual, then stages the
          # CUDA-linked native lib and the host driver (see cudaHook). The
          # device tests' CUDA cases run for real here on a machine with an
          # NVIDIA GPU; on a CPU-only box they self-skip.
          cuda = pkgs.mkShell {
            buildInputs = baseInputs;
            shellHook = provisionRacket + cudaHook;
          };
        });
    };
}
