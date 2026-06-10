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
      #   "python" -> pkgs.python3Packages.torch: the SAME libtorch the parity
      #               script imports -> bit-exact randn, at the cost of a heavy
      #               (often uncached on darwin) from-source build.
      torchSource = "bin";

      torchPackageFor = pkgs:
        if torchSource == "python" then pkgs.python3Packages.torch
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
          torch = torchPackageFor pkgs;

          cppCommonInputs = [ torch pkgs.gtest ];
          cppNativeInputs = [ pkgs.cmake pkgs.clang-tools pkgs.ninja ];
          cppCmakeFlags = [
            "-DBUILD_TESTING=ON"
            "-DCMAKE_CXX_STANDARD=20"
          ];

          cpp = pkgs.stdenv.mkDerivation {
            pname = "torchrkt-cpp";
            inherit version;
            src = ./cpp;
            nativeBuildInputs = cppNativeInputs;
            buildInputs = cppCommonInputs;
            cmakeFlags = cppCmakeFlags;
            doCheck = true;
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
              done < <(find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \))
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
        in
        {
          default = racket;
          inherit cpp cpp-format cpp-line-count cpp-tidy racket copy-native-libs;
        });

      apps = forAllSystems (system: {
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
          cpp = self.packages.${system}.cpp;

          # Python with the PyTorch wheel/lib, for interactive parity work
          # (`nix develop --command python3`) and the python-cross-test.  Cached
          # on both supported systems (a ~50 MiB fetch, not a source build).
          pythonEnv = pkgs.python3.withPackages (ps: [ ps.torch ]);

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
            deps_stamp="$PLTUSERHOME/.deps-installed-torch-''${_rkt_ver}"
            if [ ! -f "$deps_stamp" ]; then
              echo "Installing Racket package (link mode, Racket ''${_rkt_ver})..."
              mkdir -p "$PLTUSERHOME"
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
        });
    };
}
