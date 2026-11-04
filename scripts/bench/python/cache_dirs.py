"""The dataset caches the Racket side fills: RKTORCH_*_DIR, or NAME under
rktorch/ in Racket's (find-system-path 'cache-dir).

    python3 cache_dirs.py VARIABLE NAME [PLATFORM]

prints the directory, PLATFORM standing in for sys.platform."""

import os
import sys


def racket_cache_dir(environ, platform):
    """(find-system-path 'cache-dir) on Linux and macOS, as the Racket
    reference describes it."""
    user_home = environ.get("PLTUSERHOME", "")
    pinned = os.path.isabs(user_home)
    home = user_home if pinned else (environ.get("HOME") or os.path.expanduser("~"))
    if platform == "darwin":
        return os.path.join(home, "Library", "Caches", "Racket")
    xdg = environ.get("XDG_CACHE_HOME", "")
    base = (os.path.join(xdg, "racket") if os.path.isabs(xdg) and not pinned
            else os.path.join(home, ".cache", "racket"))
    legacy = os.path.join(home, ".racket")
    return legacy if not os.path.isdir(base) and os.path.isdir(legacy) else base


def cache_dir(variable, name, environ=None, platform=None):
    environ = os.environ if environ is None else environ
    override = environ.get(variable)
    if override:
        return override
    return os.path.join(racket_cache_dir(environ, platform or sys.platform),
                        "rktorch", name)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(f"usage: {sys.argv[0]} VARIABLE NAME [PLATFORM]")
    print(cache_dir(*sys.argv[1:3], platform=sys.argv[3] if len(sys.argv) == 4 else None))
