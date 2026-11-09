"""Local web server for the bubble UI, with semantic search.

Serves the static site in web/ and one API endpoint:
    GET /api/search?q=<text>  ->  {"query": ..., "results": [{code, title, group, cluster, score}, ...]}
An empty "results" list means the query was too far from every job ("No results").

Run from the project root:
    .venv\\Scripts\\python src\\serve.py [port]
"""
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from load_onet import ROOT
from search import JobSearch

WEB = ROOT / "web"
MAX_QUERY_CHARS = 200


class Handler(SimpleHTTPRequestHandler):
    engine: JobSearch = None  # set in main()

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/search":
            return self.search(parse_qs(url.query).get("q", [""])[0])
        return super().do_GET()

    def search(self, query: str):
        query = query.strip()[:MAX_QUERY_CHARS]
        body = json.dumps({"query": query, "results": self.engine.search(query)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # Always revalidate static files so an updated page never mixes with a cached old one
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):  # keep the console quiet except for searches
            super().log_message(fmt, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("Loading search model...", flush=True)
    Handler.engine = JobSearch()
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(WEB)))
    print(f"Bubbles is running at http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
