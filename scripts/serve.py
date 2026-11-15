#!/usr/bin/env python3
"""极简静态服务器,用于本地打开金字塔查看器(file:// 下 fetch 会被拦)。

用法:python scripts/serve.py out_demo/pyramid [port]
然后浏览器打开 http://localhost:8000/
"""
from __future__ import annotations

import http.server
import os
import shutil
import socketserver
import sys

VIEWER = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "viewer", "index.html")


def main() -> None:
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1
                           else "out_demo/pyramid")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    # 把查看器首页复制到金字塔根,与 manifest.json 同目录
    dst = os.path.join(root, "index.html")
    if os.path.exists(VIEWER):
        shutil.copyfile(VIEWER, dst)
    os.chdir(root)

    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map[".json"] = "application/json"
    handler.extensions_map[".jpg"] = "image/jpeg"
    with socketserver.ThreadingTCPServer(("", port), handler) as httpd:
        print(f"查看器: http://localhost:{port}/  (根目录 {root})")
        print("Ctrl+C 退出")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")


if __name__ == "__main__":
    main()
