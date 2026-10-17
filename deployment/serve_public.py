"""Portable release server behind a loopback reverse proxy. No training imports."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal
import re
import threading
from urllib.parse import urlsplit
from scripts import serve_sft_web as web
try:
    from .native_engine import ModelEngine
except ImportError:
    from native_engine import ModelEngine


def handler(controller):
    base = web.make_handler(controller)
    class PublicHandler(base):
        def setup(self):
            super().setup(); self.connection.settimeout(20)

        def allowed_origin(self):
            if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
                raise web.APIError(403, 'Invalid backend host')

        def body(self):
            value = super().body()
            if urlsplit(self.path).path == '/api/stop':
                rid = value.get('request_id') if isinstance(value, dict) else None
                if not isinstance(rid, str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,80}', rid):
                    raise web.APIError(400, 'A valid request_id is required')
            return value

    return PublicHandler


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(12)
        super().__init__(*args, **kwargs)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close(); return
        try: super().process_request(request, address)
        except BaseException: self.slots.release(); raise

    def process_request_thread(self, request, address):
        try: super().process_request_thread(request, address)
        finally: self.slots.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=web.ROOT / 'models/Clover-125M')
    p.add_argument('--origin', required=True, help='Public URL for the startup message')
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--context-tokens', type=int, choices=(256, 512, 1024), default=1024)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.threads <= 64: p.error('--threads must be between 1 and 64')
    origin = urlsplit(args.origin)
    if origin.scheme not in ('http', 'https') or origin.path or not origin.netloc:
        p.error('Use an exact origin, without path or trailing slash')
    web.ASSETS = web.ROOT / 'web/clover'
    web.LIMITS['max_new_tokens'] = (1, 256)
    web.DEFAULTS['context_tokens'] = args.context_tokens
    web.LIMITS['context_tokens'] = (256, args.context_tokens)
    controller = web.Controller(ModelEngine(args.root, context=args.context_tokens, threads=args.threads))
    server = BoundedServer(('127.0.0.1', args.port), handler(controller))
    def stop(signum, frame):
        controller.stop(); raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    print(f'{controller.engine.label} ready: {args.origin}; backend loopback:{server.server_port}', flush=True)
    try: server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt: pass
    finally: controller.stop(); server.server_close()


if __name__ == '__main__': main()
