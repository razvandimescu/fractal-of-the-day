#!/usr/bin/env python3
"""Serve the docs/ site locally and open it in a browser.

Usage: python serve.py [port]   (default 8000)
"""
import functools
import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# ports browsers refuse with ERR_UNSAFE_PORT (Chromium kRestrictedPorts)
UNSAFE = {1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 69, 77, 79,
          87, 95, 101, 102, 103, 104, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137,
          139, 143, 161, 179, 389, 427, 465, 512, 513, 514, 515, 526, 530, 531, 532, 540,
          548, 554, 556, 563, 587, 601, 636, 989, 990, 993, 995, 1719, 1720, 1723, 2049,
          3659, 4045, 5060, 5061, 6000, 6566, 6665, 6666, 6667, 6668, 6669, 6697, 10080}
if port in UNSAFE:
    sys.exit(f"port {port} is blocked by browsers (ERR_UNSAFE_PORT) — try e.g. 8000, 8080, 5173")

if not (DOCS / "index.html").exists():
    sys.exit("docs/index.html not found")
if not (DOCS / "data" / "today.json").exists():
    print("note: no rendered assets yet — run `python build_site.py` first\n")

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
socketserver.TCPServer.allow_reuse_address = True
url = f"http://localhost:{port}/"

with socketserver.TCPServer(("", port), handler) as httpd:
    print(f"serving {DOCS} at {url}  (ctrl-c to stop)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
