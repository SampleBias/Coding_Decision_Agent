import json

from data import compaction as C

DIFF = """diff --git a/pkg/paginate.py b/pkg/paginate.py
index 1111111..2222222 100644
--- a/pkg/paginate.py
+++ b/pkg/paginate.py
@@ -1,12 +1,12 @@
 import math
 
 
 def paginate(items, page, size):
     start = page * size
-    end = start + size + 1
+    end = start + size
     return items[start:end]
 
 
 def pages(n, size):
     return math.ceil(n / size)
diff --git a/tests/test_paginate.py b/tests/test_paginate.py
new file mode 100644
--- /dev/null
+++ b/tests/test_paginate.py
@@ -0,0 +1,4 @@
+from pkg.paginate import paginate
+
+def test_last_page():
+    assert paginate(list(range(10)), 1, 5) == [5, 6, 7, 8, 9]
"""


def test_compact_diff_keeps_changed_lines_and_headers():
    out = C.compact_diff(DIFF, budget=2000)
    assert "### pkg/paginate.py" in out
    assert "### tests/test_paginate.py" in out
    assert "-    end = start + size + 1" in out
    assert "+    end = start + size" in out
    assert "unchanged lines" in out
    assert "def pages" not in out  # far from any change -> elided


def test_compact_diff_respects_budget():
    big = DIFF * 40
    out = C.compact_diff(big, budget=1500)
    assert len(out) <= 1500


def test_diff_stats():
    s = C.diff_stats(DIFF)
    assert s["files"] == 2 and s["added"] == 5 and s["removed"] == 1


def test_compact_tests_prefers_failures():
    log = "\n".join(["collecting ...", "ok line"] * 50 + ["FAILED tests/test_x.py::test_a - AssertionError",
                                                        "1 failed, 99 passed in 0.5s"])
    out = C.compact_tests(log, budget=300)
    assert "FAILED" in out and "1 failed" in out and len(out) <= 300


def test_compact_tools_openai_and_plain():
    tools = [
        {"type": "function", "function": {"name": "get_weather", "description": "Get weather. Long extra text.",
                                          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                                                         "required": ["city"]}}},
        {"name": "search", "description": "Search the web", "parameters": {"q": {"type": "string"}, "k": {"type": "integer"}}},
    ]
    out = C.compact_tools(tools, budget=500)
    assert "get_weather(city*: string (City name)) - Get weather" in out
    assert "search(q: string, k: integer) - Search the web" in out


def test_normalize_call_variants():
    assert C.normalize_call({"name": "f", "arguments": {"a": 1}}) == {"name": "f", "arguments": {"a": 1}}
    assert C.normalize_call({"function": {"name": "f", "arguments": "{\"a\": 1}"}})["arguments"] == {"a": 1}
    assert C.normalize_call({"f": {"a": 1}}) == {"name": "f", "arguments": {"a": 1}}
    assert C.normalize_call('{"tool": "f", "args": {"a": 1}}')["name"] == "f"
    assert C.normalize_call("garbage")["name"] == "<unparsed>"


def test_compact_calls_and_steps():
    calls = C.compact_calls([{"name": "f", "arguments": {"x": "y" * 500}}], budget=300)
    assert calls.startswith("1. f(") and len(calls) <= 300
    steps = [{"role": "ai", "content": f"step {i}"} for i in range(20)]
    out = C.compact_steps(steps, max_recent=3, budget=500)
    assert "17 earlier steps omitted" in out and "20. AI: step 19" in out


def test_family_state_builders_fit_budget():
    st = C.code_review_state("task " * 500, DIFF * 30, "python", "FAILED x\n" * 200)
    assert st["family"] == "code_review"
    assert C.state_chars(st) <= C.cfg()["state_char_budget"] + 500
    st2 = C.tool_call_state("req", [{"name": "t"}] * 40, [{"name": "t", "arguments": {}}])
    assert "and 21 more tools" in st2["tools"]
    st3 = C.agent_trace_state("t", {"model": "m"}, ["c1", "c2"], {"steps": 3}, [{"role": "ai", "content": "x"}])
    assert st3["constraints"] == "c1; c2"
    st4 = C.routing_state("please refactor", {"repo": "r"})
    assert json.dumps(st4)
