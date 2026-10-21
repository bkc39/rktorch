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
      version = "0.1";

      # The one knob that decides the PyTorch-parity story (see plans/):
      #   "bin"    -> pkgs.libtorch-bin: small prebuilt download, fast cached CI
      #               on both platforms; parity is tolerant (cross-test absorbs
      #               any patch-version drift vs the Python torch).
      #   "python" -> pkgs.python314Packages.torch: the SAME libtorch the parity
      #               script imports -> bit-exact randn, at the cost of a heavy
      #               (often uncached on darwin) from-source build.
      torchSource = "bin";

      # The Racket toolchain from the scoped pin, guarded by the ordering
      # invariant the nixpkgsRacket input comment documents: the Racket
      # binary dlopens libtorchrkt built against the MAIN pin's glibc, and
      # glibc symbol versioning only tolerates older-built-lib into
      # newer-glibc-process — so the scoped pin's glibc must be at least
      # as new. Enforced here (not just prose) so a re-pin in the wrong
      # direction fails at evaluation, symmetric with racketReleases' version
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

      racketVersions = [ "9.1" "9.2" "9.3" ];
      racketSources = {
        "9.1" = "0gl3yvj0bn17rm3iq6wzd43gs3a4f22k2rdmb2ks5138hbhxqqhh";
      };
      racketFromSource = pkgs: v: pkgs.racket.overrideAttrs {
        version = v;
        src = pkgs.fetchurl {
          url = "https://mirror.racket-lang.org/installers/${v}/racket-${v}-src.tgz";
          sha256 = racketSources.${v};
        };
      };

      racketCell = ./nix/matrix/racket-cell.sh;
      cudaCellText = builtins.readFile ./nix/matrix/drain.sh
        + builtins.readFile ./nix/matrix/cuda-cell.sh;
      cudaDriverFarm = ./nix/cuda-driver-farm.sh;

      # Source catalog only: a built package's compiled/ belongs to one Racket.
      racketDepsDir = ./nix/racket-deps;
      racketDepsScript = "${racketDepsDir}/racket-deps.rkt";
      provisionDepsStep = deps: pkgDir: keep: platformDir: ''
        if ! bash ${racketDepsDir}/provision.sh ${deps} ${pkgDir} ${keep} ${platformDir}; then
          echo "ERROR: provisioning failed; not stamping, so the next" >&2
          echo "       shell entry retries it." >&2
          exit 1
        fi
      '';
      racketDepsFor = pkgs: racketPkg:
        pkgs.stdenvNoCC.mkDerivation (finalAttrs: {
          name = "torch-rkt-racket-deps";
          dontUnpack = true;
          nativeBuildInputs = [ racketPkg pkgs.cacert pkgs.unzip ];
          buildPhase = ''
            runHook preBuild
            export HOME=$TMPDIR/home
            export PLTUSERHOME=$TMPDIR/plt
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            mkdir -p "$PLTUSERHOME" "$TMPDIR/torch-info"
            cp ${./torch/info.rkt} "$TMPDIR/torch-info/info.rkt"
            racket ${racketDepsScript} names "$TMPDIR/torch-info" > "$TMPDIR/wanted"
            mapfile -t wanted < "$TMPDIR/wanted"
            raco pkg install --batch --auto --no-setup --scope user \
              --catalog https://pkgs.racket-lang.org "''${wanted[@]}"
            racket ${racketDepsScript} installed > "$TMPDIR/deps"
            mapfile -t deps < "$TMPDIR/deps"
            raco pkg archive "$TMPDIR/archive" "''${deps[@]}"
            mkdir -p "$out"
            for z in "$TMPDIR"/archive/pkgs/*.zip; do
              name="$(basename "$z" .zip)"
              mkdir -p "$out/$name"
              unzip -q "$z" -d "$out/$name"
            done
            racket ${racketDepsScript} closure "$out"
            racket ${racketDepsScript} manifest "$TMPDIR/torch-info" > "$out/.info-deps"
            racket ${racketDepsScript} guard "$out"
            racket ${racketDepsScript} check "$TMPDIR/torch-info" "$out"
            runHook postBuild
          '';
          dontInstall = true;
          outputHashMode = "recursive";
          outputHashAlgo = "sha256";
          outputHash = "sha256-YyR47U1gR5twmnllsNOddO8mhFJ4MCtmsDu8WjY2JRI=";
          passthru.tests = racketDepsTestsFor pkgs racketPkg finalAttrs.finalPackage;
        });

      racketDepsTestsFor = pkgs: racketPkg: racket-deps:
        let
          run = name: script: pkgs.runCommand name { nativeBuildInputs = [ racketPkg pkgs.git ]; } ''
            export HOME=$TMPDIR/home PLTUSERHOME=$TMPDIR/plt
            rd() { racket ${racketDepsScript} "$@"; }
            fails() {
              if "$@" > "$TMPDIR/log" 2>&1; then
                echo "expected a failure: $*" >&2
                exit 1
              fi
            }
            ${script}
            mkdir "$out"
          '';
          fixtures = ./nix/racket-deps/fixtures;
        in
        {
          racket-deps-current = run "torch-rkt-racket-deps-current" ''
            mkdir -p "$TMPDIR/info"
            cp ${./torch/info.rkt} "$TMPDIR/info/info.rkt"
            rd check "$TMPDIR/info" ${racket-deps}
            for platform in x86_64-linux aarch64-macosx; do
              [ "$(rd platform-packages "$TMPDIR/info" ${./pkgs} $platform)" = "${./pkgs}/torch-$platform" ]
            done
            [ "$(rd platform-packages "$TMPDIR/info" ${./pkgs} aarch64-darwin)" = "${./pkgs}/torch-aarch64-macosx" ]
            [ "$(rd platform-packages "$TMPDIR/info" ${./pkgs} x86_64-linux-natipkg)" = "${./pkgs}/torch-x86_64-linux" ]
          '';
          racket-deps-guard-test = run "torch-rkt-racket-deps-guard-test" ''
            t=$TMPDIR/trees
            mkdir -p "$t/source/pkg/doc" "$t/zo/pkg/compiled" "$t/sxref/pkg/doc/pkg" \
              "$t/linked/pkg" "$t/elsewhere/compiled" \
              "$t/sealed/pkg/private"
            touch "$t/source/pkg/main.rkt" "$t/source/pkg/doc/guide.scrbl" \
              "$t/zo/pkg/compiled/x_rkt.zo" "$t/sxref/pkg/doc/pkg/out0.sxref" \
              "$t/elsewhere/compiled/x_rkt.zo" "$t/sealed/pkg/main.rkt"
            ln -s "$t/elsewhere/compiled" "$t/linked/pkg/compiled"
            chmod 000 "$t/sealed/pkg/private"
            [ ! -r "$t/sealed/pkg/private" ]
            rd guard "$t/source"
            fails rd guard "$t/zo"
            fails rd guard "$t/sxref"
            fails rd guard "$t/linked"
            fails rd guard "$t/sealed"
            grep -q 'could not open directory' "$TMPDIR/log"
            chmod 755 "$t/sealed/pkg/private"
          '';
          racket-deps-info-test = run "torch-rkt-racket-deps-info-test" ''
            built=$TMPDIR/built
            mkdir -p "$built"
            rd manifest ${fixtures}/current > "$built/.info-deps"
            rd check ${fixtures}/current "$built"
            fails rd check ${fixtures}/added "$built"
            grep -q 'bump racket-deps outputHash' "$TMPDIR/log"
            fails rd check ${fixtures}/versioned "$built"
            fails rd check ${fixtures}/removed "$built"
            [ "$(rd names ${fixtures}/no-build-deps)" = threading-lib ]
            fails rd names ${fixtures}/distribution-only
            rd check ${fixtures}/base-versioned "$built"
            fails rd check ${fixtures}/base-unmet "$built"
            grep -qF "base requires #:version 99.0; found $(racket -e '(display (version))') (installation)" "$TMPDIR/log"
            for v in 1.2 1.3; do
              cp -r ${fixtures}/fetched-$v "$TMPDIR/fetched-$v"
              chmod -R u+w "$TMPDIR/fetched-$v"
              rd manifest ${fixtures}/versioned > "$TMPDIR/fetched-$v/.info-deps"
            done
            fails rd check ${fixtures}/versioned "$TMPDIR/fetched-1.2"
            grep -q 'threading-lib requires #:version 1.3; found 1.2' "$TMPDIR/log"
            rd check ${fixtures}/versioned "$TMPDIR/fetched-1.3"
            fails rd names ${fixtures}/platform
            grep -q 'extra-lib has #:platform unix; platform-qualified' "$TMPDIR/log"
            rd check ${fixtures}/own-platform "$built"
            for platform in x86_64-linux aarch64-macosx; do
              [ "$(rd platform-packages ${fixtures}/own-platform ${fixtures}/own-platform/pkgs $platform)" \
                = "${fixtures}/own-platform/pkgs/torch-$platform" ]
            done
            [ "$(rd platform-packages ${fixtures}/own-platform ${fixtures}/own-platform/pkgs aarch64-darwin)" \
              = "${fixtures}/own-platform/pkgs/torch-aarch64-macosx" ]
            [ "$(rd platform-packages ${fixtures}/own-platform ${fixtures}/own-platform/pkgs x86_64-linux-natipkg)" \
              = "${fixtures}/own-platform/pkgs/torch-x86_64-linux" ]
            [ -z "$(rd platform-packages ${fixtures}/own-platform ${fixtures}/own-platform/pkgs x86_64-macosx)" ]
            racket ${racketDepsDir}/platform-lists-test.rkt ${racketDepsScript} \
              ${./torch/foreign/raw/native-location.rkt} ${./torch/info.rkt}
            racket ${racketDepsDir}/platform-lists-test.rkt ${racketDepsScript} \
              ${./torch/foreign/raw/native-location.rkt} ${fixtures}/own-platform/info.rkt
            fails racket ${racketDepsDir}/platform-lists-test.rkt ${racketDepsScript} \
              ${./torch/foreign/raw/native-location.rkt} ${fixtures}/own-platform-crossed/info.rkt
            grep -qF 'declares torch-x86_64-linux with #:platform "aarch64-macosx", which platform-packages' "$TMPDIR/log"
            fails racket ${racketDepsDir}/platform-lists-test.rkt ${racketDepsScript} \
              ${./torch/foreign/raw/native-location.rkt} ${fixtures}/own-platform-unmapped/info.rkt
            grep -qF 'declares torch-aarch64-macosx with #:platform "x86_64-macosx", which platform-packages' "$TMPDIR/log"
            fails rd platform-packages ${fixtures}/own-platform-skew ${fixtures}/own-platform/pkgs
            grep -qF 'torch-x86_64-linux has #:version "0.1"; ${fixtures}/own-platform-skew has native-version "0.2"' "$TMPDIR/log"
            fails rd platform-packages ${fixtures}/own-platform-ahead ${fixtures}/own-platform/pkgs
            grep -qF 'torch-x86_64-linux is version "0.1"; ${fixtures}/own-platform-ahead has native-version "0.2"' "$TMPDIR/log"
            [ "$(rd platform-packages ${fixtures}/own-platform-split ${fixtures}/own-platform/pkgs x86_64-linux)" \
              = "${fixtures}/own-platform/pkgs/torch-x86_64-linux" ]
            fails rd platform-packages ${fixtures}/own-platform-unversioned ${fixtures}/own-platform/pkgs
            grep -qF '${fixtures}/own-platform-unversioned has no native-version' "$TMPDIR/log"
            fails rd platform-packages ${fixtures}/own-platform-bad-native ${fixtures}/own-platform/pkgs
            grep -qF 'has native-version "0.1-alpha", which is not a valid Racket version' "$TMPDIR/log"
            mkdir "$TMPDIR/no-skeletons"
            fails rd platform-packages ${fixtures}/own-platform "$TMPDIR/no-skeletons"
            grep -q 'torch-x86_64-linux has no info.rkt with a version' "$TMPDIR/log"
            fails rd platform-packages ${fixtures}/own-platform-symbol ${fixtures}/own-platform/pkgs
            grep -qF 'torch-x86_64-linux has #:platform unix; expected a string such as "x86_64-linux"' "$TMPDIR/log"
            fails rd names ${fixtures}/own-platform-symbol
            grep -qF 'torch-x86_64-linux has #:platform unix; expected a string' "$TMPDIR/log"
            [ "$(rd base-version ${fixtures}/floored)" = 9.0 ]
            fails rd base-version ${fixtures}/current
            fails rd base-version ${fixtures}/versioned
            fails rd check ${fixtures}/base-platform "$built"
            grep -q 'base has #:platform windows; platform-qualified' "$TMPDIR/log"
            fails rd names ${fixtures}/base-platform
            grep -q 'base has #:platform windows; platform-qualified' "$TMPDIR/log"
            fails rd check ${fixtures}/bad-version "$built"
            grep -qF 'base has #:version "99", which is not a valid Racket version' "$TMPDIR/log"
            fails rd names ${fixtures}/bad-version
            grep -qF 'base has #:version "99", which is not a valid Racket version' "$TMPDIR/log"
            fails rd base-version ${fixtures}/base-platform
            grep -q 'base has #:platform windows; platform-qualified' "$TMPDIR/log"
            fails rd base-version ${fixtures}/bad-version
            grep -qF 'base has #:version "99", which is not a valid Racket version' "$TMPDIR/log"
            fails rd names ${fixtures}/source
            grep -qF 'https://example.com/foo.git is a package source; racket-deps fetches catalog packages by name' "$TMPDIR/log"
            rd closure ${fixtures}/closure-clean
            fails rd closure ${fixtures}/closure-platform
            grep -q 'extra-lib has #:platform unix; platform-qualified' "$TMPDIR/log"
            grep -q 'declared in plat-lib/info.rkt' "$TMPDIR/log"
          '';
          racket-deps-provision-test = run "torch-rkt-racket-deps-provision-test" ''
            timeout -k 10 600 bash ${racketDepsDir}/provision-test.sh \
              ${racketDepsDir}/provision.sh ${fixtures}/provision \
              ${pkgs.writeText "provision-deps-step" (provisionDepsStep ''"$1"'' ''"$2"'' ''"$3"'' ''"$4"'')}
          '';
        };

      abiStampTestsFor = pkgs:
        let
          cApi = ./cpp/include/torchrkt/c_api;
          run = name: script: pkgs.runCommand name { nativeBuildInputs = [ pkgs.cmake ]; } ''
            stamp() { cmake -DC_API_DIR="$1" -P ${./cpp/cmake/abi_stamp.cmake}; }
            current() {
              want=$(stamp "$1")
              have=$(tr -d '"\n' < "$2")
              if [ "$want" != "$have" ]; then
                echo "torch/foreign/raw/abi-stamp.rktd holds $have, but the c_api headers hash to $want." >&2
                echo "Regenerate it with: nix run .#abi-stamp (nix run .#codegen also does), and commit it." >&2
                return 1
              fi
            }
            fails() {
              if "$@" > "$TMPDIR/log" 2>&1; then
                echo "expected a failure: $*" >&2
                exit 1
              fi
            }
            ${script}
            mkdir "$out"
          '';
        in
        {
          abi-stamp-current = run "torch-rkt-abi-stamp-current" ''
            current ${cApi} ${./torch/foreign/raw/abi-stamp.rktd}
          '';
          abi-stamp-test = run "torch-rkt-abi-stamp-test" ''
            base=$(stamp ${cApi})
            [[ "$base" =~ ^[0-9a-f]{64}$ ]]
            copy() { cp -r ${cApi} "$TMPDIR/$1"; chmod -R u+w "$TMPDIR/$1"; }
            copy same
            [ "$(stamp "$TMPDIR/same")" = "$base" ]
            copy edited
            echo >> "$TMPDIR/edited/global.h"
            [ "$(stamp "$TMPDIR/edited")" != "$base" ]
            fails current "$TMPDIR/edited" ${./torch/foreign/raw/abi-stamp.rktd}
            grep -q 'nix run .#abi-stamp' "$TMPDIR/log"
            copy renamed
            mv "$TMPDIR/renamed/global.h" "$TMPDIR/renamed/global2.h"
            [ "$(stamp "$TMPDIR/renamed")" != "$base" ]
            copy nested
            echo >> "$TMPDIR/nested/generated/math.h"
            [ "$(stamp "$TMPDIR/nested")" != "$base" ]
            copy added
            touch "$TMPDIR/added/extra.h"
            [ "$(stamp "$TMPDIR/added")" != "$base" ]
          '';
        };

      releaseTestsFor = pkgs: racketPkg:
        let
          src = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./flake.nix
              ./nix/racket-deps/racket-deps.rkt
              ./pkgs
              ./scripts/platform/catalog-entry.rkt
              ./scripts/platform/sources.txt
              ./scripts/release
              ./torch/foreign/raw/abi-stamp.rktd
              ./torch/info.rkt
              ./torch/private/util.rkt
            ];
          };
          run = name: script: pkgs.runCommand name { nativeBuildInputs = [ racketPkg pkgs.git ]; } ''
            export HOME=$TMPDIR/home PLTUSERHOME=$TMPDIR/plt
            cp -r ${src} "$TMPDIR/src"
            chmod -R u+w "$TMPDIR/src"
            cd "$TMPDIR/src"
            ${script}
            mkdir "$out"
          '';
        in
        {
          release-current = run "torch-rkt-release-current" ''
            racket scripts/release/release.rkt check
          '';
          release-test = run "torch-rkt-release-test" ''
            raco test --timeout 300 scripts/release
          '';
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
        else zipTorchFor pkgs;

      zipTorchFor = pkgs:
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
      # Same discipline as torch/vision/mnist.rkt's cache write.
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
          racket-linters =
            pkgs.callPackage ./nix/racket-linters.nix { racket = racketPkg; };

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
          mkCpp = { p, doCheck ? true, cuda ? false, torch ? torchPackageFor p
                  , installTests ? false }:
            let cudaTk = p.cudaPackages_13.cudatoolkit;
            in p.stdenv.mkDerivation ({
              pname = "torchrkt-cpp";
              inherit version doCheck;
              src = ./cpp;
              nativeBuildInputs = [ p.cmake p.clang-tools p.ninja p.pkg-config ]
                ++ p.lib.optional cuda p.cudaPackages_13.cuda_nvcc;
              buildInputs = [ torch p.gtest p.libsndfile p.stb ]
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
            } // p.lib.optionalAttrs installTests {
              postInstall = ''
                rpath=$(patchelf --print-rpath torchrkt_tests | tr : '\n' \
                  | grep -v "^$NIX_BUILD_TOP" | paste -sd:)
                install -Dm755 torchrkt_tests $out/libexec/torchrkt/torchrkt_tests
                patchelf --set-rpath "$out/lib:$rpath" $out/libexec/torchrkt/torchrkt_tests
              '';
            });
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

          mkRacketPackage = { pname, toolchain, shim ? cpp, docs ? false
                            , floorTest ? false }:
            pkgs.stdenv.mkDerivation {
              inherit pname version;
              src = ./.;
              passthru = {
                inherit toolchain;
                tests = pkgs.lib.optionalAttrs floorTest {
                  racket-floor = racketFloorTest toolchain;
                };
              };

              nativeBuildInputs = [
                toolchain
                pkgs.makeWrapper
                racket-deps.tests.racket-deps-current
              ];
              buildInputs = [ shim ];

              buildPhase = ''
                runHook preBuild
                export PLTUSERHOME=$TMPDIR/racket-home
                bash ${racketCell} install ${shim} ${racket-deps} ${if docs then "docs" else "no-docs"}
                runHook postBuild
              '';

              doCheck = true;
              checkPhase = ''
                runHook preCheck
                bash ${racketCell} test
                runHook postCheck
              '';

              installPhase = ''
                runHook preInstall

                mkdir -p $out/share $out/bin
                cp -r $PLTUSERHOME $out/share/racket-home

                makeWrapper ${toolchain}/bin/racket $out/bin/torch \
                  --set PLTUSERHOME $out/share/racket-home \
                  --add-flags "-l torch"

                runHook postInstall
              '';
            };

          racketFloorTest = toolchain:
            let
              oldest = builtins.head racketVersions;
              checked = pkgs.lib.versions.majorMinor toolchain.version;
            in
            pkgs.runCommand "torch-rkt-racket-floor"
              { nativeBuildInputs = [ racketPkg ]; } ''
              export HOME=$TMPDIR/home PLTUSERHOME=$TMPDIR/plt
              same() { [ "$1" = "$2" ] && [ "$2" = "$3" ]; }
              if same 9.0 9.1 9.1 || same 9.1 9.1 9.2; then exit 1; fi
              mkdir -p "$TMPDIR/info"
              cp ${./torch/info.rkt} "$TMPDIR/info/info.rkt"
              floor=$(racket ${racketDepsScript} base-version "$TMPDIR/info")
              if ! same "$floor" ${oldest} ${checked}; then
                echo "torch/info.rkt's base floor is $floor, racketVersions' oldest is ${oldest}, the floor check builds ${checked}" >&2
                exit 1
              fi
              mkdir $out
            '';

          racketReleases = pkgs.lib.genAttrs racketVersions (v:
            let
              toolchain = { "9.2" = pkgs.racket; "9.3" = racketPkg; }.${v}
                or (racketFromSource pkgs v);
            in
            assert pkgs.lib.assertMsg
              (pkgs.lib.versions.majorMinor toolchain.version == v)
              ("racket " + v + ": the toolchain is now " + toolchain.version
               + "; re-point racketReleases or the floor (see #41/#50)");
            toolchain);

          racket = mkRacketPackage {
            pname = "torch-rkt";
            toolchain = racketReleases."9.3";
          };
          racket91 = mkRacketPackage {
            pname = "torch-rkt-racket91";
            toolchain = racketReleases."9.1";
            floorTest = true;
          };

          cachedTorch = pkgs.lib.last (pkgs.lib.sort
            (a: b: pkgs.lib.versionOlder a.version b.version)
            [ pkgs.libtorch-bin pkgs.python314Packages.torch ]);
          cachedTorchSkip =
            if pkgs.lib.versionAtLeast cachedTorch.version libtorchVersion then null
            else cachedTorch.version + " < floor "
              + pkgs.lib.versions.majorMinor libtorchVersion;

          cppCudaTests = mkCpp {
            p = pkgsCuda;
            torch = zipTorchFor pkgsCuda;
            doCheck = false;
            cuda = true;
            installTests = true;
          };
          cudaCell = pkgs.writeShellApplication {
            name = "torch-matrix-cuda-cell";
            runtimeInputs = [ racketReleases."9.3" ] ++ pkgs.stdenv.initialPath;
            inheritPath = false;
            runtimeEnv = {
              CELL_SRC = "${./.}";
              CELL_SHIM = "${cppCudaTests}";
              CELL_DEPS = "${racket-deps}";
              CELL_CUDNN = "${pkgs.lib.getLib (cudnnFor pkgsCuda)}/lib";
              CELL_RACKET_CELL = "${racketCell}";
              CELL_DRIVER_FARM = "${cudaDriverFarm}";
            };
            text = cudaCellText;
          };

          # Steps name .drv files: evaluating the matrix builds nothing.
          matrixRow = { racket, libtorch, skip ? null, steps ? [ ] }:
            let
              step = s: s.label + "="
                + builtins.unsafeDiscardOutputDependency s.drv.drvPath;
              rest = if skip != null then [ "skip" skip ] else [
                "build"
                (pkgs.lib.concatMapStringsSep " " step steps)
              ];
            in
            pkgs.lib.concatStringsSep "\t" ([ racket libtorch "nix" ] ++ rest) + "\n";
          cpuRows = libtorch: shim: skip: map
            (v: matrixRow {
              racket = v;
              inherit libtorch skip;
              steps = [
                { label = "shim"; drv = shim; }
                { label = "toolchain"; drv = racketReleases.${v}; }
                {
                  label = "cell";
                  drv = mkRacketPackage {
                    pname = "torch-rkt-matrix-racket${v}";
                    toolchain = racketReleases.${v};
                    inherit shim;
                    docs = true;
                  };
                }
              ];
            })
            racketVersions;
          matrixCells = pkgs.writeText "torch-matrix-${system}.tsv"
            (pkgs.lib.concatStrings (
              cpuRows "zip ${libtorchVersion}"
                (mkCpp { p = pkgs; torch = zipTorchFor pkgs; })
                null
              ++ cpuRows "nixpkgs ${cachedTorch.version}"
                (mkCpp { p = pkgs; torch = cachedTorch; })
                cachedTorchSkip
              ++ pkgs.lib.optional pkgs.stdenv.isLinux (matrixRow {
                racket = "9.3";
                libtorch = "zip ${libtorchVersion} cu130";
                steps = [
                  { label = "shim"; drv = cppCudaTests; }
                  { label = "toolchain"; drv = racketReleases."9.3"; }
                  { label = "run"; drv = cudaCell; }
                ];
              })));
          matrix = pkgs.writeShellApplication {
            name = "torch-matrix";
            runtimeInputs = [ pkgs.coreutils pkgs.gawk pkgs.gnugrep pkgs.gnused ];
            runtimeEnv = {
              MATRIX_SYSTEM = system;
              MATRIX_CELLS = "${matrixCells}";
            };
            text = builtins.readFile ./nix/matrix/drain.sh
              + builtins.readFile ./nix/matrix/run.sh;
            passthru.cells = matrixCells;
          };

          libtorch-zip = pkgs.fetchzip libtorchZips."${system}-cpu";

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
          # (from nixpkgs' python torch) + the pinned clang-format the
          # generator formats its C++ output with. Writes into the working
          # tree, so it must run from the repo root — much lighter than the
          # full dev shell when all you need is regeneration.
          codegen = pkgs.writeShellApplication {
            name = "codegen";
            runtimeInputs = [
              (pkgs.python314.withPackages (ps: [ ps.torch ]))
              pkgs.clang-tools
              pkgs.cmake
            ];
            text = ''
              if [ ! -f codegen/generate.py ]; then
                echo "codegen: run from the repo root (codegen/ not found)" >&2
                exit 1
              fi
              exec python3 -m codegen "$@"
            '';
          };

          abi-stamp = pkgs.writeShellApplication {
            name = "abi-stamp";
            runtimeInputs = [ pkgs.cmake ];
            text = ''
              if [ ! -f cpp/cmake/abi_stamp.cmake ]; then
                echo "abi-stamp: run from the repo root (cpp/cmake/abi_stamp.cmake not found)" >&2
                exit 1
              fi
              cmake -DOUT=torch/foreign/raw/abi-stamp.rktd -P cpp/cmake/abi_stamp.cmake
              echo "abi-stamp: torch/foreign/raw/abi-stamp.rktd is $(cat torch/foreign/raw/abi-stamp.rktd)"
            '';
          };
        in
        {
          default = racket;
          inherit abi-stamp cpp cpp-format cpp-line-count cpp-tidy racket racket91
            racket-deps racket-linters codegen copy-native-libs matrix
            libtorch-zip;
        }
        // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
          # The CUDA libtorch-bin has no darwin download, so even evaluating
          # this output aborts `nix flake check` on aarch64-darwin.
          inherit cpp-cuda;
        });

      apps = forAllSystems (system: {
        abi-stamp = {
          type = "app";
          program = "${self.packages.${system}.abi-stamp}/bin/abi-stamp";
        };
        codegen = {
          type = "app";
          program = "${self.packages.${system}.codegen}/bin/codegen";
        };
        copy-native-libs = {
          type = "app";
          program = "${self.packages.${system}.copy-native-libs}/bin/copy-native-libs";
        };
        matrix = {
          type = "app";
          program = "${self.packages.${system}.matrix}/bin/torch-matrix";
        };
      });

      inherit libtorchZips;

      checks = forAllSystems (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          inherit (self.packages.${system})
            cpp cpp-format cpp-line-count cpp-tidy racket racket91 matrix;
          matrix-cuda-cell = pkgs.writeShellApplication {
            name = "torch-matrix-cuda-cell";
            text = cudaCellText;
          };
          matrix-runner-test = pkgs.runCommand "torch-matrix-runner-test" { } ''
            DRIVER_FARM=${cudaDriverFarm} timeout -k 10 600 bash ${./nix/matrix/run-test.sh} ${./nix/matrix}
            mkdir $out
          '';
        } // self.packages.${system}.racket-deps.tests
          // self.packages.${system}.racket91.tests
          // abiStampTestsFor pkgs
          // releaseTestsFor pkgs (racketFor pkgs (import nixpkgsRacket { inherit system; }))
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            platform-audit-test = pkgs.runCommand "torch-rkt-platform-audit-test"
              { nativeBuildInputs = [ pkgs.binutils ]; } ''
              audit() {
                if bash ${./scripts/platform/linux/audit-bundle.sh} "$@" > "$TMPDIR/log" 2>&1; then
                  echo "the audit passed $*" >&2
                  cat "$TMPDIR/log" >&2
                  exit 1
                fi
              }
              audit ${self.packages.${system}.cpp}/lib
              grep -q 'FAIL libtorchrkt.so: mentions /nix/store/' "$TMPDIR/log"
              grep -q 'FAIL libtorchrkt.so: runpath entry /nix/store/' "$TMPDIR/log"
              grep -q 'FAIL libtorchrkt.so: needs GLIBC_2\.[0-9]* > 2.28' "$TMPDIR/log"
              grep -qF 'FAIL libtorchrkt.so: NEEDs libc10.so, neither bundled nor a system library' "$TMPDIR/log"
              mkdir "$TMPDIR/sibling"
              cp ${self.packages.${system}.cpp}/lib/libtorchrkt.so "$TMPDIR/sibling/"
              cp ${self.packages.${system}.cpp}/lib/libtorchrkt.so "$TMPDIR/sibling/libc10.so"
              audit "$TMPDIR/sibling"
              grep -qF 'FAIL libtorchrkt.so: NEEDs bundled libc10.so, but its runpath lacks $ORIGIN' "$TMPDIR/log"
              PATH=${pkgs.writeShellScriptBin "objdump" "true"}/bin:$PATH \
                audit ${self.packages.${system}.cpp}/lib
              grep -q 'objdump -T reads GLIBC_<none>, readelf -V GLIBC_2\.' "$TMPDIR/log"
              mkdir "$out"
            '';
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
          racket-linters = self.packages.${system}.racket-linters;
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
          # nixpkgs' torch until the macOS wheels have been tried on a Mac, and
          # torchSource = "python" keeps it everywhere: the shim then links
          # nixpkgs' torch, so the twins must run that same build.
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
          pythonFromNixpkgs = p: p.python314.withPackages
            (ps: [ ps.soundfile ps.torch ps.torchaudio ps.torchvision ]);
          pythonEnv =
            if system == "x86_64-linux" && torchSource == "bin"
            then pythonWheels null
            else pythonFromNixpkgs pkgs;

          # The `.#cuda` shell's Python: with torchSource = "bin", the cu130
          # wheels, linked against the CUDA libraries the CUDA libtorch bundles
          # and its cuDNN 9.24, so both sides of a CUDA parity test load the
          # same ones.
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
            if torchSource == "bin"
            then pythonWheels [ cudaRuntime (pkgs.lib.getLib cudaCudnn) ]
            else pythonFromNixpkgs pkgsCuda;

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
            # In-tree zo caches compiled piecewise across commits can defeat
            # the compilation manager, so bytecode is keyed to HEAD by a
            # stamp-and-clear (a per-rev PLTCOMPILEDROOTS would recompile
            # the copied dep packages on every pull).
            _rev=$(git rev-parse HEAD 2>/dev/null || echo norev)
            _rev_stamp="$PLTUSERHOME/.provisioned-rev"
            _old_rev=$(cat "$_rev_stamp" 2>/dev/null || echo none)
            if [ "$_old_rev" != "$_rev" ]; then
              _clear_ok=1
              if [ "$_old_rev" != "none" ] || compgen -G "$PLTUSERHOME/.deps[0-9]*" > /dev/null; then
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
            ${provisionDepsStep racket-deps ''"$PWD/torch"'' racket-linters ''"$PWD/pkgs"''}
            # Stamped by the dev tools' store path, so a bump reaches checkouts
            # provisioned before it: install adds the missing packages, then
            # update moves all of them, from an older pin or the live catalog
            # alike, onto these trees.
            linters_stamp="$PLTUSERHOME/.linters-${builtins.baseNameOf racket-linters}"
            if [ ! -f "$linters_stamp" ]; then
              echo "Installing the pinned Racket dev tools (Resyntax + racket-review + cover)..."
              if raco pkg install --batch --copy --no-docs --no-setup --deps fail \
                     --scope user --skip-installed ${racket-linters}/*/ \
                 && raco pkg update --batch --copy --no-docs --no-trash \
                      --deps fail --scope user ${racket-linters}/*/; then
                touch "$linters_stamp"
                echo "Done. Lint: scripts/resyntax analyze --local-git-repository ."
                echo "      full sweep: scripts/resyntax analyze --directory torch  |  raco review <files>"
                echo "      coverage:   racket scripts/coverage.rkt"
              else
                echo "WARNING: installing the dev tools failed; not stamping, so" >&2
                echo "         the next shell entry retries it." >&2
              fi
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
            bash ${cudaDriverFarm} "$_drv_farm"
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
