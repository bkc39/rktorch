{
  description = "rktorch - Racket bindings to libtorch (PyTorch)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    # A newer nixpkgs pinned ONLY for the CUDA Python torch in the `.#cuda`
    # devShell's parity pass (see pythonCudaEnv). The main `nixpkgs` pin still
    # ships torch-bin 2.11+cu128, whose prebuilt wheel needs CUDA 12 libs and so
    # can't link the cudaPackages_13 stack the cuda libtorch-bin uses; this pin
    # carries torch-bin 2.12+cu130, matching cudaPackages_13 (so the heavy
    # nccl/ucc/nvshmem build is shared with cpp-cuda's CUDA closure). Scoped to
    # that one env so the cpp/racket builds stay on the main pin untouched.
    nixpkgsCuda.url =
      "github:NixOS/nixpkgs/3e41b24abd260e8f71dbe2f5737d24122f972158";
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

  outputs = { self, nixpkgs, nixpkgsCuda, nixpkgsRacket }:
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
          # The CUDA libtorch the shim links; its lib/ holds the bundled cuDNN
          # (libcudnn_*.so.9) that conv/pool ops dlopen by soname at runtime.
          cudaTorch = torchPackageFor (import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
          });

          # Python with the PyTorch wheel/lib, for interactive parity work
          # (`nix develop --command python3`) and the python-cross-test.  Cached
          # on both supported systems (a ~50 MiB fetch, not a source build).
          pythonEnv = pkgs.python314.withPackages (ps: [ ps.torch ]);

          # CUDA-capable Python torch for the accelerator parity pass of the
          # cross-test (the `.#cuda` shell): the cu130 torch-bin wheel — the
          # torch binary itself is a download, no from-source build — in the same
          # CUDA family (cu130) as the cuda libtorch-bin the shim links, from the
          # nixpkgsCuda pin (see inputs). Its NCCL/UCC/nvshmem deps build from
          # source once, shared with cpp-cuda's cudaPackages_13 closure. nixpkgs
          # marks the wheel broken here only because its cuda-bindings *metadata*
          # package is stale (12.9.7 < the 13.0.3 the wheel wants); the wheel
          # carries its own cu130 runtime, so that one gate is a false positive
          # here and is ignored outright — "warn" would repeat the (already
          # settled) diagnosis on every `.#cuda` shell entry. Re-audit if the
          # nixpkgsCuda pin or the torch-bin version moves.
          pkgsCudaPy = import nixpkgsCuda {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
              problems.handlers.torch.unsupported-cuda-version = "ignore";
            };
          };
          # cudaPackages_13 so the cu130 wheel patchelfs against .so.13 (and
          # shares cpp-cuda's CUDA closure). dontCheckRuntimeDeps skips the
          # pythonRuntimeDepsCheckHook, which otherwise rejects the wheel because
          # this nixpkgs' cuda-bindings *metadata* pkg is 12.9.7 (< the >=13.0.3
          # the 2.12 wheel declares); cuda-bindings (cuda-python) isn't on the
          # path of the conv/linear/adam ops the parity pass exercises.
          # Self-enforcing re-audit tripwire for the "ignore" above: the
          # suppression was justified against torch-bin 2.12 (cu130) on this
          # exact nixpkgsCuda pin. If a pin bump moves the wheel version, fail
          # eval loudly here instead of silently carrying the suppression
          # forward — bump this prefix only after re-checking that the
          # unsupported-cuda-version problem is still a metadata-only false
          # positive for the new wheel.
          auditedTorchBinPrefix = "2.12.";
          pythonCudaEnv =
            assert pkgsCudaPy.lib.assertMsg
              (pkgsCudaPy.lib.hasPrefix auditedTorchBinPrefix
                pkgsCudaPy.python314.pkgs.torch-bin.version)
              ''
                torch-bin moved to ${pkgsCudaPy.python314.pkgs.torch-bin.version}
                (audited: ${auditedTorchBinPrefix}x): re-audit the
                unsupported-cuda-version "ignore" above, then update
                auditedTorchBinPrefix.'';
            pkgsCudaPy.python314.withPackages
            (ps: [ ((ps.torch-bin.override {
              cudaPackages = pkgsCudaPy.cudaPackages_13;
            }).overridePythonAttrs (_: { dontCheckRuntimeDeps = true; })) ]);

          baseInputs = [
            pkgs.cmake
            pkgs.clang-tools
            pkgs.gtest
            pkgs.ninja
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
            deps_stamp="$PLTUSERHOME/.deps2-installed-torch-''${_rkt_ver}"
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
              raco pkg install --batch --copy --no-docs --no-setup --scope user --skip-installed \
                ${racket-deps}/*/
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
          # Driver farm only; `provisionRacketFor cpp-cuda` stages the shim and
          # points TORCHRKT_NATIVE_LIB_PATH at it.
          cudaHook = ''
            echo "Staging host NVIDIA driver farm..."
            _drv_farm="$PWD/.cuda-driver"
            rm -rf "$_drv_farm"; mkdir -p "$_drv_farm"
            for _l in libcuda.so.1 libnvidia-ml.so.1; do
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
            # Driver farm first (host libcuda), then the libtorch lib dir so its
            # bundled cuDNN resolves — conv/pool dlopen libcudnn_*.so.9 by
            # soname, and the autoAddDriverRunpath doesn't cover that. (matmul
            # and friends worked without it; only the cuDNN-backed ops need it.)
            export LD_LIBRARY_PATH="$_drv_farm:${cudaTorch}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            # The Python torch-bin (cu130 wheel, pythonCudaEnv) finds its own
            # bundled libtorch/cuDNN via RUNPATH and needs ONLY the host driver —
            # NOT cudaTorch/lib, whose libtorch 2.9 libs would shadow the wheel's
            # 2.12 ones (libtorch_python.so ABI clash). The cross-test pins the
            # python child's LD_LIBRARY_PATH to just this farm when it's set.
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
