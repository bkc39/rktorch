#!/usr/bin/env python3
"""Drive `racket -i` on a REAL pty and send a REAL Ctrl-D (0x04).

Honours REPRO_DEVICE (cpu|cuda|mps) and REPRO_LOGDIR.

A piped stdin is not the same condition: on a tty, `racket -i` loads xrepl's
terminal line editor, and Ctrl-D is an EOT byte the reader interprets, not an
EOF on a closed pipe.  Usage:

    pty-ctrl-d.py <mode> [iterations]

modes: idle     Ctrl-D at a quiet prompt
       busy     Ctrl-D while a long computation is still running
       printing Ctrl-D while a huge tensor is streaming to the terminal
       intr     Ctrl-C then Ctrl-D
       spam     ten Ctrl-Ds in a row
"""
import os, pty, re, select, signal, sys, time

DEV = os.environ.get("REPRO_DEVICE", "cpu").lower()
_PICK = ('(define D (case "%s" [("cuda") (cuda-device)] [("mps") (mps-device)] '
         '[else (cpu-device)]))\n' % DEV).encode()
_HOLD = b"(define held (with-default-device D (for/list ([_ (in-range 400)]) (randn 128 128))))\n"
_GUARD = ('(unless (case "%s" [("cuda") (cuda-available?)] [("mps") (mps-available?)] '
          '[("cpu") #t] [else (printf "REPRO-DEVICE-~a\\n" "UNKNOWN") (exit 3)]) '
          '(printf "REPRO-DEVICE-~a\\n" "UNAVAILABLE") (exit 3))\n'
          % DEV).encode()
_HEAD = [b"(require torch)\n", _GUARD, _PICK, _HOLD]

SETUP = {
    "idle":     _HEAD + [b"(length held)\n"],
    "busy":     _HEAD + [b"(with-default-device D (for ([_ (in-range 4000)]) (void (matmul (randn 256 256) (randn 256 256)))))\n"],
    "printing": _HEAD + [b"(with-default-device D (randn 1200 1200))\n"],
    "intr":     _HEAD + [b"(with-default-device D (for ([_ (in-range 4000)]) (void (matmul (randn 256 256) (randn 256 256)))))\n"],
    "spam":     _HEAD + [b"(length held)\n"],
}
DELAY = {"idle": 12.0, "busy": 3.0, "printing": 0.35, "intr": 3.0, "spam": 12.0}

def run(mode, idx, budget=90.0):
    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.execvpe("racket", ["racket", "-i"],
                       {**os.environ, "RKTORCH_MEM_TRACE": "1"})
        except OSError:
            os._exit(127)
    out = bytearray()
    t0 = time.time()
    def pump(until):
        while time.time() < until:
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue
            try:
                b = os.read(fd, 65536)
            except OSError:
                return False
            if not b:
                return False
            out.extend(b)
        return True
    pump(time.time() + 6)
    for line in SETUP[mode]:
        os.write(fd, line)
        pump(time.time() + 0.4)
    pump(time.time() + DELAY[mode])
    if mode == "intr":
        os.write(fd, b"\x03")
        pump(time.time() + 1.5)
    os.write(fd, b"\r")
    pump(time.time() + 0.6)
    os.write(fd, b"\x04")
    if mode == "spam":
        for _ in range(9):
            os.write(fd, b"\x04")
    exited = None
    deadline = t0 + budget
    while time.time() < deadline and exited is None:
        pump(time.time() + 0.5)
        try:
            wpid, status = os.waitpid(pid, os.WNOHANG)
            if wpid == pid:
                exited = status
        except ChildProcessError:
            exited = 0
    hung = exited is None
    termignored = False
    if hung:
        os.kill(pid, signal.SIGTERM)
        time.sleep(4)
        try:
            wpid, status = os.waitpid(pid, os.WNOHANG)
            if wpid != pid:
                termignored = True
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
        except (ChildProcessError, ProcessLookupError):
            pass
    os.close(fd)
    text = bytes(out)
    if b"REPRO-DEVICE-UNAVAILABLE" in text or b"REPRO-DEVICE-UNKNOWN" in text:
        print(f"ABORT pty/{mode}: device '{DEV}' unusable on this host", flush=True)
        raise SystemExit(3)
    imr = text.count(b"invalid memory reference")
    flat = text.replace(b"\n", b"").replace(b"\r", b"").replace(b";", b"").replace(b" ", b"")
    setup_error = (flat.count(b"btforcontext]") - flat.count(b"userbreak")) > 0
    m = re.findall(rb"\(failures \. (\d+)\)", text)
    fin_failures = int(m[-1]) if m else 0
    casc = text.count(b"error display handler") + text.count(b"error escape handler")
    abnormal = exited is not None and exited != 0
    inconclusive = hung and not termignored and not imr and not casc
    bad = imr or casc or abnormal or setup_error or fin_failures or (hung and termignored)
    if bad or inconclusive:
        d = os.environ.get("REPRO_LOGDIR", "/tmp")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"rktorch-pty-{mode}-{DEV}-{idx}.log")
        open(p, "wb").write(text)
        print(f"{'FAIL' if bad else 'INCONCLUSIVE'} mode={mode} iter={idx} hung={hung} "
              f"term_ignored={termignored} abnormal_exit={abnormal} "
              f"setup_error={setup_error} finalizer_failures={fin_failures} imr={imr} "
              f"cascade={casc} bytes={len(text)} exit={exited} log={p}", flush=True)
    return bool(bad), bool(inconclusive)

if __name__ == "__main__":
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    results = [run(mode, i) for i in range(1, n + 1)]
    bad = sum(b for b, _ in results)
    incon = sum(i for _, i in results)
    suffix = f" ({incon} inconclusive: Ctrl-D not delivered)" if incon else ""
    print(f"RESULT pty/{mode}-{DEV}: {bad}/{n} failed{suffix}", flush=True)
    if bad:
        raise SystemExit(1)
    raise SystemExit(2 if incon == n else 0)
