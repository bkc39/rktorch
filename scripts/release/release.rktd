(platform-zips
 "torch-x86_64-linux/pkg/torch-x86_64-linux.zip"
 "torch-aarch64-macosx/out/pkg/torch-aarch64-macosx.zip")

(lgpl-sources
 "lame-3.100.tar.gz"
 "libsndfile-1.2.2.tar.xz"
 "mpg123-1.33.4.tar.bz2")

(natives-inputs
 ".github/workflows/platform-linux-build.yml"
 ".github/workflows/platform-linux.yml"
 ".github/workflows/platform-macos.yml"
 "cpp"
 "pkgs"
 "scripts/platform")

(natives-inputs-exclude
 "cpp/.clang-format"
 "cpp/.clang-tidy"
 "cpp/cmake/tidy-parallel.sh"
 "cpp/tests/"
 "scripts/platform/catalog-entry.rkt"
 "scripts/platform/catalog.sh"
 "scripts/platform/linux/audit-bundle.sh"
 "scripts/platform/linux/in-host-smoke.sh"
 "scripts/platform/linux/smoke.sh"
 "scripts/platform/macos/audit-bundle.sh"
 "scripts/platform/macos/audit-test.sh"
 "scripts/platform/macos/install-racket.sh"
 "scripts/platform/size-budget.sh"
 "scripts/platform/smoke-install.sh")
