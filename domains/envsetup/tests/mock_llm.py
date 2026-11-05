"""
A minimal OpenAI-compatible chat-completions server that replays a scripted list of
tool calls.  Used to exercise the agent runner end to end without a real model.

    from domains.envsetup.tests.mock_llm import MockLLMServer
    with MockLLMServer([("bash", {"command": "ls", "reason": "look"}), ("finish", {...})]) as srv:
        runner = EnvSetupAgentRunner(base_url=srv.base_url, model="openai/mock", ...)
"""
from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Tuple


class MockLLMServer:
    def __init__(self, script: List[Tuple[str, dict]], host: str = "127.0.0.1", port: int = 0):
        self.script = list(script)
        self.calls: List[dict] = []
        self._i = 0
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                body = json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                with server._lock:
                    server.calls.append(req)
                    if server._i < len(server.script):
                        name, args = server.script[server._i]
                    else:
                        name, args = "finish", {"report": "script exhausted", "success": False}
                    server._i += 1
                msg = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                       "function": {"name": name, "arguments": json.dumps(args)}}]}
                resp = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": 0,
                        "model": req.get("model", "mock"),
                        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
                body = json.dumps(resp).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self.base_url = f"http://{host}:{self._httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
