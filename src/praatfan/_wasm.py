"""The praatfan_wasm backend's engine: praatfan_rust compiled to wasm32, run in
an embedded V8 (``mini-racer``).

The bundle in ``wasm/`` is the same Rust code as the ``praatfan_rust`` wheel
and the ``praatfan-open-pipe`` binary, built ``--target no-modules`` (see
``wasm/PROVENANCE.txt`` for the commit, and ``rust/wasm-v8/build.sh``). It is
driven through one entry point, ``pipe_request_samples(samples, sample_rate,
request_json)``: the JSON pipe protocol of ``praatfan-open-pipe`` with the
audio handed over as decoded samples. So the WASM backend covers exactly what
the pipe covers (pitch, formant, intensity, harmonicity, spectral moments,
band energy); the adapter in ``selector`` serves everything else from the
pure-Python implementation.

Nothing numeric happens here. Samples never travel as JSON numbers: they cross
as base64 of their little-endian float64 bytes, in chunks, into one
preallocated ``Float64Array`` (a long recording never becomes one giant JS
string; V8 caps a string at ~512M characters).

V8 flags: mini-racer initialises V8 with ``--single-threaded`` unless told
otherwise, which turns off V8's background compilation threads (where wasm
tier-up to the optimising compiler normally happens). V8 is initialised here
with no flags -- its defaults. (Measured 2026-10-15: a 60 s formant analysis
took 2.02 s with V8's defaults and 2.05 s with ``--single-threaded``, so for
this engine it is a precaution, not a speed-up.) The engine itself runs
single-threaded inside a call (wasm32 has no rayon); parallelism is across
processes, each with its own context.

One context per process, created lazily and cached by pid. Calls into a
context are serialised by a lock. A Rust panic is a wasm trap, after which the
instance's memory may be inconsistent: the context is marked broken and
replaced on the next call.

Fork. V8 does not survive ``fork()``: a child forked from a process that has
started V8 hangs as soon as it touches V8 again (``multiprocessing``'s default
start method on Linux before Python 3.14 is fork). A process forked from such a
parent therefore never starts V8 itself; it hands its calls to a persistent
helper process (``HelperEngine``), a fresh interpreter started with
fork+exec, which owns a clean V8. A child forked from a parent that never
started V8 runs in-process as usual.
"""
import atexit
import base64
import json
import os
import subprocess
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(HERE, "wasm")
GLUE = "praatfan.js"
WASM = "praatfan_bg.wasm"
SHIM = "shim.js"
BUNDLE_FILES = (SHIM, GLUE, WASM)

#: V8's own defaults (see the module docstring for why not mini-racer's).
V8_FLAGS = ()

#: Sample bytes per base64 chunk handed to JS: a multiple of 3 and of 8, so
#: every chunk is whole base64 quanta and whole float64 values.
CHUNK_BYTES = 3 * 8 * (1 << 20)          # 24 MiB of samples, 32 MiB of base64

# The host side of the bridge: `__pf_alloc` makes the samples buffer,
# `__pf_put` decodes one base64 chunk straight into it, `__pf_call` serves one
# request and drops the buffer.
_HOST_JS = r"""
var __pf_samples = null;
function __pf_init(b64) {
  wasm_bindgen.initSync({module: praatfan_b64bytes(b64)});
  return wasm_bindgen.version();
}
function __pf_alloc(n) { __pf_samples = new Float64Array(n); return n; }
function __pf_put(offset, s) {
  return praatfan_b64decode(s, new Uint8Array(__pf_samples.buffer), offset);
}
function __pf_call(sr, req) {
  var x = __pf_samples;
  __pf_samples = null;
  if (globalThis.__praatfan_log) globalThis.__praatfan_log.length = 0;
  return wasm_bindgen.pipe_request_samples(x, sr, req);
}
function __pf_drain_log() {
  var l = globalThis.__praatfan_log || [];
  var s = l.join('\n');
  l.length = 0;
  return s;
}
"""


class WasmUnavailable(RuntimeError):
    """The WASM backend cannot run here; the message says why."""


class WasmEngineError(RuntimeError):
    """The engine answered with an error, or trapped."""


def unavailable_reason():
    """None when the WASM backend can run, else one sentence saying why not."""
    try:
        import py_mini_racer  # noqa: F401
    except ImportError:
        return ("the mini-racer package is not installed "
                "(pip install \"praatfan[wasm]\" or pip install mini-racer)")
    missing = [f for f in BUNDLE_FILES if not os.path.isfile(os.path.join(BUNDLE_DIR, f))]
    if missing:
        return (f"the engine bundle is incomplete ({', '.join(missing)} missing from "
                f"{BUNDLE_DIR}); reinstall praatfan")
    return None


class Engine:
    """One V8 context holding one instantiated bundle."""

    def __init__(self):
        why = unavailable_reason()
        if why:
            raise WasmUnavailable(why)
        global _V8_STARTED
        import py_mini_racer as pmr
        _V8_STARTED = True
        try:
            pmr.init_mini_racer(flags=V8_FLAGS, ignore_duplicate_init=True)
        except Exception:                         # noqa: BLE001 - initialised by someone
            pass                                  # else: their flags stand
        self._lock = threading.Lock()
        self._ctx = pmr.MiniRacer()

        def read(name):
            with open(os.path.join(BUNDLE_DIR, name), encoding="utf-8") as fh:
                return fh.read()
        self._ctx.eval(read(SHIM))
        self._ctx.eval(read(GLUE))
        self._ctx.eval(_HOST_JS)
        with open(os.path.join(BUNDLE_DIR, WASM), "rb") as fh:
            wasm_b64 = base64.b64encode(fh.read()).decode("ascii")
        self.version = self._ctx.eval("__pf_init")(wasm_b64)
        self._alloc = self._ctx.eval("__pf_alloc")
        self._put = self._ctx.eval("__pf_put")
        self._call = self._ctx.eval("__pf_call")
        self._drain = self._ctx.eval("__pf_drain_log")
        self.broken = False

    def request(self, samples, sample_rate, request):
        """One pipe request (a dict, without ``wav_path``) against mono
        ``samples`` at ``sample_rate`` Hz -> the parsed response's ``results``.

        Raises ``WasmEngineError`` with the engine's message when the response
        is ``ok: false``; a trap also marks this context broken.
        """
        with self._lock:
            if self.broken:
                raise WasmEngineError("this engine context is closed")
            text = _call_raw(self, samples, sample_rate, request)
        resp = json.loads(text)
        if not resp.get("ok"):
            raise WasmEngineError(resp.get("error") or "the wasm engine returned no result")
        return resp["results"]

    def close(self):
        """Free the V8 context."""
        ctx, self._ctx = self._ctx, None
        self.broken = True
        if ctx is not None:
            ctx.close()


class HelperEngine:
    """The engine in a helper process (see "Fork" in the module docstring).

    Same ``request`` contract as ``Engine``. The wire: one header line of JSON
    (``sample_rate``, ``n`` samples, the request) followed by the samples' raw
    little-endian float64 bytes; one line of response JSON back.
    """

    def __init__(self):
        why = unavailable_reason()
        if why:
            raise WasmUnavailable(why)
        env = dict(os.environ)
        root = os.path.dirname(HERE)              # this praatfan, wherever it lives
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [sys.executable, "-c", "from praatfan._wasm import _serve; _serve()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self.version = self._exchange(np.zeros(0), 1.0, None)
        self.broken = False

    def _exchange(self, samples, sample_rate, request):
        raw = np.ascontiguousarray(samples, dtype="<f8").tobytes()
        head = json.dumps({"sample_rate": float(sample_rate), "n": len(raw) // 8,
                           "request": request}, allow_nan=False)
        try:
            self._proc.stdin.write(head.encode() + b"\n" + raw)
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except OSError as exc:
            line, err = b"", exc
        else:
            err = None
        if not line:
            self.broken = True
            raise WasmEngineError(f"the wasm helper process died ({err or 'no answer'})")
        out = json.loads(line)
        if "fatal" in out:
            self.broken = True
            raise WasmEngineError(out["fatal"])
        return out["text"]

    def request(self, samples, sample_rate, request):
        with self._lock:
            if self.broken:
                raise WasmEngineError("this engine context is closed")
            text = self._exchange(samples, sample_rate, request)
        resp = json.loads(text)
        if not resp.get("ok"):
            raise WasmEngineError(resp.get("error") or "the wasm engine returned no result")
        return resp["results"]

    def close(self):
        self.broken = True
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
            except Exception:                     # noqa: BLE001 - shutting down anyway
                proc.kill()


def _serve():
    """The helper process's loop (``HelperEngine``): one request per header."""
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    eng = None
    while True:
        line = inp.readline()
        if not line:
            return
        head = json.loads(line)
        raw = inp.read(8 * head["n"])
        try:
            if eng is None or eng.broken:
                eng = Engine()
            if head["request"] is None:
                reply = {"text": eng.version}
            else:
                samples = np.frombuffer(raw, dtype="<f8")
                with eng._lock:
                    text = _call_raw(eng, samples, head["sample_rate"], head["request"])
                reply = {"text": text}
        except Exception as exc:                  # noqa: BLE001 - reported to the parent
            reply = {"fatal": str(exc)}
        out.write(json.dumps(reply).encode() + b"\n")
        out.flush()


def _call_raw(eng, samples, sample_rate, request):
    """``Engine.request`` without parsing the response (the helper forwards it)."""
    raw = np.ascontiguousarray(samples, dtype="<f8").tobytes()
    try:
        eng._alloc(len(raw) // 8)
        off = 0
        for i in range(0, len(raw), CHUNK_BYTES):
            off = eng._put(off, base64.b64encode(raw[i:i + CHUNK_BYTES]).decode("ascii"))
        return eng._call(float(sample_rate), json.dumps(request, allow_nan=False))
    except Exception as exc:                      # noqa: BLE001 - JS errors vary by type
        eng.broken = True
        try:
            log = eng._drain().strip()
        except Exception:                         # noqa: BLE001
            log = ""
        raise WasmEngineError(f"the wasm engine failed: {exc}"
                              + (f" ({log})" if log else "")) from None


#: This process has started V8 (``Engine``).
_V8_STARTED = False
#: This process was forked from one that had started V8, so must not touch it.
_FORKED_FROM_V8 = False


#: Contexts inherited across fork. Never used, closed or collected: their
#: owner threads do not exist in the child, so closing one (``__del__``
#: included) waits forever.
_INHERITED = []


def _after_fork_in_child():
    global _FORKED_FROM_V8, _V8_STARTED, _ENGINES_LOCK
    _FORKED_FROM_V8 = _FORKED_FROM_V8 or _V8_STARTED
    _V8_STARTED = False
    _ENGINES_LOCK = threading.Lock()              # the parent's may have been held
    _INHERITED.extend(_ENGINES.values())
    _ENGINES.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


_ENGINES = {}
_ENGINES_LOCK = threading.Lock()


def _close_all():
    """Close every context at interpreter exit, while threads still run.

    Left to the garbage collector, a ``MiniRacer`` is closed from ``__del__``
    during finalisation, after Python has stopped scheduling daemon threads,
    and closing joins mini-racer's event-loop thread: the process can hang at
    exit. An ``atexit`` hook runs before that point."""
    with _ENGINES_LOCK:
        engines = list(_ENGINES.values())
        _ENGINES.clear()
    for e in engines:
        try:
            e.close()
        except Exception:                         # noqa: BLE001 - exiting anyway
            pass


atexit.register(_close_all)


def engine():
    """This process's engine context, created on first use (and after a trap)."""
    pid = os.getpid()
    with _ENGINES_LOCK:
        e = _ENGINES.get(pid)
        if e is None or e.broken:
            if e is not None:
                try:
                    e.close()
                except Exception:                 # noqa: BLE001 - replacing it anyway
                    pass
            e = HelperEngine() if _FORKED_FROM_V8 else Engine()
            _ENGINES[pid] = e
        return e


def run(samples, sample_rate, analyses):
    """Run pipe ``analyses`` (a list of dicts) on mono samples -> their results."""
    return engine().request(samples, sample_rate, {"analyses": analyses})
