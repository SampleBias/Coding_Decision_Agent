#!/usr/bin/env python3
"""HTTP sidecar for the Laya-CDA test suite.

    python3 suite/sidecar/server.py
    python3 suite/sidecar/server.py --backend real --checkpoint S4MPL3BI4S/Coding_Decision_Agent

The TUI never loads weights. It POSTs a case to /v1/compare, which grades with
Laya-CDA and, when OPENROUTER_API_KEY is set, with Jev on the OpenRouter
Decisions API. Both are asked the same questions. Mock mode is the default
for Laya and does not download a model. Real mode loads the checkpoint through
the coding_decision_agent SDK, which uses the Hugging Face cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "suite"
for path in (ROOT, ROOT / "sdk", SUITE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sidecar import jev
from sidecar.mock_grader import MODEL_HEURISTIC, MODEL_SCRIPTED, answers_for, from_scripted, heuristic_labels
from sidecar.schema_wire import assert_same_schema, wire_families
from sidecar.state import compact_state

DEFAULT_CATALOG = SUITE / "cases" / "catalog.json"
FAMILIES = ("code_review", "tool_call", "agent_trace", "routing")


class Context:
    def __init__(self, catalog: dict, backend: str, real=None):
        self.catalog = catalog
        self.backend = backend
        self.real = real
        self.cases = {row["case_id"]: row for row in catalog["cases"]}
        self.live_families = wire_families()
        assert_same_schema(catalog["families"], self.live_families)
        if backend == "real":
            self.model_id = real.checkpoint
            self.device = real.device_name
        else:
            self.model_id = MODEL_SCRIPTED
            self.device = "none"

    def health(self) -> dict:
        return {
            "ok": True,
            "service": "laya-cda-sidecar",
            "backend": self.backend,
            "model_id": self.model_id,
            "device": self.device,
            "n_cases": len(self.cases),
            "families": [fam["id"] for fam in self.live_families],
            "jev_configured": jev.configured(),
            "jev_model": jev.model_name() if jev.configured() else "",
        }

    def compare(self, body: dict) -> dict:
        """Grade one case with Laya-CDA and Jev. A missing Jev key is not a failed grade."""
        laya = self.grade(body)
        jev_grade, jev_error = self._jev(body, laya.get("state") or {})
        return {"ok": True, "laya": laya, "jev": jev_grade, "jev_error": jev_error}

    def grade_jev(self, body: dict) -> dict:
        family, _case_id, fields = self._resolve(body)
        state = compact_state(family, fields)
        grade, error = self._jev(body, state)
        if error or grade is None:
            raise jev.JevUnavailable(error or "Jev did not return a grade")
        return grade

    def _jev(self, body: dict, state: dict) -> tuple[Optional[dict], Optional[str]]:
        family, case_id, _fields = self._resolve(body)
        questions = [q for fam in self.live_families if fam["id"] == family for q in fam["questions"]]
        try:
            grade = jev.decide(family, state, questions)
        except jev.JevUnavailable as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        grade["case_id"] = case_id
        return grade, None

    def _resolve(self, body: dict) -> tuple[str, Optional[str], dict]:
        family = body.get("family")
        case_id = body.get("case_id")
        fields = body.get("fields")
        if case_id and case_id in self.cases and not isinstance(fields, dict):
            row = self.cases[case_id]
            fields = row["fields"]
            family = family or row["family"]
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if not isinstance(fields, dict):
            raise ValueError("fields must be an object")
        return family, case_id, fields

    def grade(self, body: dict) -> dict:
        family, case_id, fields = self._resolve(body)
        questions = [q for fam in self.live_families if fam["id"] == family for q in fam["questions"]]
        t0 = time.perf_counter()
        note = ""
        missing: list[str] = []
        if self.backend == "real":
            packed = self.real.grade(family, fields, {q["id"]: q for q in questions})
            answers = packed["answers"]
            missing = packed["missing"]
            source = "model"
            model_id = self.real.checkpoint
            device = self.real.device_name
            note = "checkpoint"
        else:
            row = self.cases.get(case_id) if case_id else None
            if row and row.get("family") == family and row.get("mock"):
                answers = from_scripted(questions, row["mock"])
                source = "scripted"
                model_id = MODEL_SCRIPTED
                note = "scripted mock, not the checkpoint"
            else:
                labels, masses = heuristic_labels(family, fields)
                answers = answers_for(questions, labels, masses)
                source = "heuristic"
                model_id = MODEL_HEURISTIC
                note = "keyword heuristic, not the checkpoint"
            device = "none"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": True,
            "backend": self.backend,
            "source": source,
            "model_id": model_id,
            "device": device,
            "family": family,
            "case_id": case_id,
            "latency_ms": latency_ms,
            "state": compact_state(family, fields),
            "answers": answers,
            "missing": missing,
            "note": note,
        }


def load_catalog(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"catalog not found: {path}\nBuild it with: python3 suite/cases/build_catalog.py")
    with path.open() as handle:
        return json.load(handle)


def make_context(catalog_path: Path, backend: str, checkpoint: str, device: Optional[str]) -> Context:
    catalog = load_catalog(catalog_path)
    real = None
    if backend == "real":
        from sidecar.real_grader import RealGrader
        real = RealGrader(checkpoint, device=device)
    return Context(catalog, backend, real)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s  %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stderr.flush()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        ctx: Context = self.server.ctx  # type: ignore[attr-defined]
        if path in ("/", "/v1/health"):
            self._send(200, ctx.health() if path == "/v1/health" else {
                "service": "laya-cda-sidecar",
                "health": "/v1/health",
                "grade": "/v1/grade",
                "compare": "/v1/compare",
            })
            return
        if path == "/v1/schema":
            self._send(200, {"families": ctx.live_families})
            return
        if path == "/v1/cases":
            self._send(200, {"cases": ctx.catalog["cases"], "pairs": ctx.catalog["pairs"], "replays": ctx.catalog["replays"]})
            return
        self._send(404, {"ok": False, "error": f"unknown path {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/grade", "/v1/compare", "/v1/jev"):
            self._send(404, {"ok": False, "error": f"unknown path {path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, {"ok": False, "error": "bad Content-Length"})
            return
        if length > 2_000_000:
            self._send(413, {"ok": False, "error": "body too large"})
            return
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            self._send(400, {"ok": False, "error": f"invalid json: {exc}"})
            return
        ctx: Context = self.server.ctx  # type: ignore[attr-defined]
        try:
            if path == "/v1/compare":
                payload = ctx.compare(body)
            elif path == "/v1/jev":
                payload = ctx.grade_jev(body)
            else:
                payload = ctx.grade(body)
            self._send(200, payload)
        except jev.JevUnavailable as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except (KeyError, ValueError, TypeError) as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # the model can fail in ways we should surface, not hide
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True


def serve(ctx: Context, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.ctx = ctx  # type: ignore[attr-defined]
    httpd.daemon_threads = True
    return httpd


def serve_background(ctx: Context, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    httpd = serve(ctx, host, port)
    thread = threading.Thread(target=httpd.serve_forever, name="cda-sidecar", daemon=True)
    thread.start()
    return httpd


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--backend", choices=("mock", "real"), default="mock")
    parser.add_argument("--checkpoint", default="S4MPL3BI4S/Coding_Decision_Agent",
                        help="Hugging Face repo id or a local checkpoint directory (real backend)")
    parser.add_argument("--device", default=None, help="torch device passed to laya.load, for example cpu or cuda")
    args = parser.parse_args(argv)
    ctx = make_context(args.cases, args.backend, args.checkpoint, args.device)
    httpd = serve(ctx, args.host, args.port)
    host, port = httpd.server_address[:2]
    print(
        f"Laya-CDA sidecar  backend={ctx.backend}  model={ctx.model_id}  "
        f"device={ctx.device}  jev={'on ' + jev.model_name() if jev.configured() else 'off'}  "
        f"http://{host}:{port}  cases={len(ctx.cases)}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
