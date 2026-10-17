"""Lightweight HTTP/SSE controller for the native Clover web runtime."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import re
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from urllib.parse import urlsplit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.sft_format import validate_messages
ASSETS = ROOT / 'web/clover'
DEFAULTS = dict(do_sample=True, temperature=0.35, top_p=0.85, top_k=30, repetition_penalty=1.1, no_repeat_ngram_size=4, max_new_tokens=128, seed=42, context_tokens=1024)
LIMITS = dict(temperature=(0.05, 1.5), top_p=(0.05, 1.0), top_k=(0, 200), repetition_penalty=(0.8, 1.5), no_repeat_ngram_size=(0, 8), max_new_tokens=(1, 512), seed=(0, 2147483647), context_tokens=(256, 1024))
INTS = {'top_k', 'no_repeat_ngram_size', 'max_new_tokens', 'seed', 'context_tokens'}

class APIError(ValueError):

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status

def parameters(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise APIError(400, '未知的生成参数。')
    result = {**DEFAULTS, **value}
    if type(result['do_sample']) is not bool:
        raise APIError(400, '采样开关必须为布尔值。')
    for key, (low, high) in LIMITS.items():
        number = result[key]
        valid = type(number) is int if key in INTS else type(number) in (int, float)
        if not valid or not math.isfinite(number) or (not low <= number <= high):
            raise APIError(400, f'{key} 应在 {low} 到 {high} 之间。')
    return result

def fit_history(messages, encode, context, max_new):
    """Drop oldest COMPLETE pairs, retaining system/latest input intact."""
    try:
        validate_messages(messages, complete=False)
    except (ValueError, TypeError) as error:
        raise APIError(400, str(error)) from error
    kept = [dict(m) for m in messages]
    dropped = 0
    while True:
        try:
            ids = encode(kept)
        except ValueError as error:
            raise APIError(400, str(error)) from error
        if len(ids) + max_new <= context:
            return (kept, ids, dropped)
        start = int(kept[0]['role'] == 'system')
        if len(kept) - start <= 1:
            raise APIError(400, '当前消息与预留回答超过上下文。请缩短消息、系统提示或输出长度。')
        del kept[start:start + 2]
        dropped += 1

class Job:

    def __init__(self, rid):
        self.id = rid
        self.cancel = threading.Event()
        self.events = queue.Queue()
        self.worker = None

class Controller:

    def __init__(self, engine=None):
        self.engine = engine
        self.guard = threading.Lock()
        self.state_guard = threading.Lock()
        self.current = None

    def info(self):
        return dict(ready=self.engine is not None, preview=self.engine is None, model=self.engine.label if self.engine else 'Clover · 界面预览', device=str(self.engine.device) if self.engine else None, context_limit=self.engine.context if self.engine else 1024, busy=self.current is not None, defaults=DEFAULTS, limits=LIMITS)

    def begin(self, payload):
        if self.engine is None:
            raise APIError(503, '当前为界面预览，未加载模型。训练结束后用正常启动命令运行。')
        if not isinstance(payload, dict):
            raise APIError(400, '请求必须为 JSON 对象。')
        rid = payload.get('request_id', '')
        if not isinstance(rid, str) or not re.fullmatch('[A-Za-z0-9_-]{8,80}', rid):
            raise APIError(400, '无效的请求编号。')
        params = parameters(payload.get('parameters', {}))
        messages = payload.get('messages')
        if not self.guard.acquire(blocking=False):
            raise APIError(409, '模型正在生成，请等待或停止当前回答。')
        try:
            plan = self.engine.prepare(messages, params)
            job = Job(rid)
            with self.state_guard:
                self.current = job
            job.events.put(dict(type='meta', request_id=rid, prompt_tokens=len(plan[1]), context_tokens=min(self.engine.context, params['context_tokens']), dropped_turns=plan[2], max_new_tokens=params['max_new_tokens'], seed=params['seed']))
            job.worker = threading.Thread(target=self._run, args=(job, plan, params), daemon=True)
            job.worker.start()
            return job
        except BaseException:
            with self.state_guard:
                self.current = None
            self.guard.release()
            raise

    def _run(self, job, plan, params):
        started = time.perf_counter()
        try:
            result = self.engine.generate(plan, params, job)
            elapsed = time.perf_counter() - started
            job.events.put(dict(type='done', **result, elapsed_seconds=elapsed, tokens_per_second=result['output_tokens'] / max(elapsed, 0.001)))
        except Exception as error:
            traceback.print_exc()
            job.events.put(dict(type='error', message=str(error) or type(error).__name__))
        finally:
            with self.state_guard:
                self.current = None
            self.guard.release()
            job.events.put(None)

    def stop(self, rid=None):
        with self.state_guard:
            job = self.current
            if job is not None and (rid is None or rid == job.id):
                job.cancel.set()
                return True
            return False

def make_handler(controller):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):
            print(f'[{self.log_date_time_string()}] {fmt % args}', file=sys.stderr, flush=True)

        def allowed_origin(self):
            port = self.server.server_port
            if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                raise APIError(403, '仅允许本机访问。')
            origin = self.headers.get('Origin')
            if origin:
                try:
                    parsed = urlsplit(origin)
                    valid = parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1') and (parsed.port == port)
                except ValueError:
                    valid = False
                if not valid:
                    raise APIError(403, '不允许跨站请求。')

        def common_headers(self):
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")

        def reply(self, status, value, kind='application/json; charset=utf-8'):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.common_headers()
            self.close_connection = True
            self.send_header('Connection', 'close')
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                raise APIError(415, '需要 JSON 请求。')
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                raise APIError(400, '无效请求长度。')
            if not 0 < length <= 65536:
                raise APIError(413, '请求过大或为空。')
            try:
                return json.loads(self.rfile.read(length).decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                raise APIError(400, '无效 JSON。')

        def do_GET(self):
            try:
                self.allowed_origin()
                path = urlsplit(self.path).path
                if path == '/api/state':
                    return self.reply(200, controller.info())
                files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.css': ('app.css', 'text/css; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/clover.svg': ('clover.svg', 'image/svg+xml')}
                if path not in files:
                    raise APIError(404, 'Not found')
                filename, kind = files[path]
                self.reply(200, (ASSETS / filename).read_bytes(), kind)
            except APIError as error:
                self.reply(error.status, {'error': str(error)})

        def do_POST(self):
            job = None
            try:
                self.allowed_origin()
                payload = self.body()
                path = urlsplit(self.path).path
                if path == '/api/stop':
                    if not isinstance(payload, dict):
                        raise APIError(400, '无效请求。')
                    return self.reply(200, {'stopped': controller.stop(payload.get('request_id', ''))})
                if path != '/api/chat':
                    raise APIError(404, 'Not found')
                job = controller.begin(payload)
                self.send_response(200)
                self.common_headers()
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                last = time.monotonic()
                while True:
                    try:
                        event = job.events.get(timeout=0.5)
                    except queue.Empty:
                        if time.monotonic() - last > 2:
                            self.wfile.write(b': keepalive\n\n')
                            self.wfile.flush()
                            last = time.monotonic()
                        continue
                    if event is None:
                        break
                    self.wfile.write(('data: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode('utf-8'))
                    self.wfile.flush()
                    last = time.monotonic()
            except APIError as error:
                self.reply(error.status, {'error': str(error)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                if job:
                    job.cancel.set()
            except Exception:
                if job:
                    job.cancel.set()
                traceback.print_exc()
                if not job:
                    self.reply(500, {'error': '服务器错误，请查看终端日志。'})
    return Handler
