"""Live mode for the slice viewer: serve a built viewer folder on 127.0.0.1 and rebuild the page whenever the
generator changes, so that an edit to ``build_slice_viewer.PAGE`` shows up in the browser within ~2 s.

    python -m cmb.review.live_server OUT_DIR [--port 8765] [--no-watch] [--open]

- Serves OUT_DIR on http://127.0.0.1:PORT/ -- loopback only, never a LAN address: the folders hold patient images.
  OUT_DIR is either one viewer folder (quality_control.html, data.json, img/) or a parent folder whose direct
  subfolders are viewers (e.g. charite/swi, charite/t2star); then every viewer is served and rebuilt.
- The page polls its own Last-Modified header (see PAGE in build_slice_viewer.py) and reloads itself; case and
  slice survive the reload (URL hash + sessionStorage).
- With --watch (default) the server checks the mtime of build_slice_viewer.py every second and, when it changed,
  reruns the build with the arguments stored in OUT_DIR/build_args.json plus ``--refresh-html`` (page only,
  data.json and images untouched). Build errors are printed and the old page stays.
"""
from __future__ import annotations
import argparse, http.server, json, os, subprocess, sys, threading, time, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = os.path.join(HERE, "build_slice_viewer.py")


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)


def rebuild(out: str) -> None:
    args_file = os.path.join(out, "build_args.json")
    if not os.path.exists(args_file):
        print(f"[live] no {args_file}: build the page once with build_slice_viewer first", flush=True); return
    cmd = [sys.executable, "-m", "cmb.review.build_slice_viewer"] + json.load(open(args_file)) + ["--refresh-html"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(HERE)), capture_output=True, text=True)
    if r.returncode == 0:
        print(f"[live] {time.strftime('%H:%M:%S')} page rebuilt in {time.time() - t0:.1f} s", flush=True)
    else:
        print(f"[live] BUILD FAILED (old page kept):\n{r.stderr[-2000:]}", flush=True)


def viewers(out: str) -> list:
    """The viewer folders under ``out``: ``out`` itself or its direct subfolders that hold a quality_control.html."""
    if os.path.exists(os.path.join(out, "quality_control.html")):
        return [out]
    return sorted(d for d in (os.path.join(out, n) for n in os.listdir(out)) if os.path.exists(os.path.join(d, "quality_control.html")))


def watch(out: str) -> None:
    last = os.path.getmtime(GENERATOR)
    while True:
        time.sleep(1.0)
        try:
            m = os.path.getmtime(GENERATOR)
        except OSError:
            continue
        if m != last:
            last = m; time.sleep(0.3)  # let the editor finish writing
            for v in viewers(out):
                rebuild(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-watch", action="store_true", help="only serve, do not rebuild on generator changes")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    found = viewers(out)
    if not found:
        sys.exit(f"no quality_control.html in {out} or its subfolders")
    if not a.no_watch:
        threading.Thread(target=watch, args=(out,), daemon=True).start()
    urls = [f"http://127.0.0.1:{a.port}/" + ("" if v == out else os.path.basename(v) + "/") + "quality_control.html" for v in found]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), lambda *x: Handler(*x, directory=out))
    print(f"[live] serving {out}   (watching {os.path.basename(GENERATOR)}: {'yes' if not a.no_watch else 'no'}; Ctrl-C stops)", flush=True)
    for u in urls:
        print(f"[live] {u}", flush=True)
    if a.open:
        for u in urls:
            webbrowser.open(u)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
