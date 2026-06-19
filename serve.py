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
