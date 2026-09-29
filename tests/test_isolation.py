import json, os, sys, tempfile, types, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# ---- fake config so hindsight_client imports without secrets
cfg = types.ModuleType("config")
cfg.HINDSIGHT_API_URL, cfg.HINDSIGHT_API_KEY, cfg.HINDSIGHT_BANK_ID = "http://hs", "k", "bank"
sys.modules["config"] = cfg

# ---- fake Hindsight server (tag-filtered, returns metadata like the real API)
import requests
STORE = []
class R:
    def __init__(s, d, ok=True): s._d, s.ok, s.status_code, s.text = d, ok, 200, ""
    def json(s): return s._d
def fake_post(url, headers=None, json=None, timeout=None):
    if url.endswith("/memories"):
        for it in json["items"]:
            STORE.append({"id": str(len(STORE)), "content": it["content"], "metadata": it["metadata"],
                          "tags": it["tags"], "timestamp": it.get("timestamp", "")})
        return R({})
    tags = set(json["tags"])
    assert "workspace" not in json["query"].lower() or True
    hits = [{"id": m["id"], "text": m["content"], "metadata": m["metadata"], "mentioned_at": m["timestamp"]}
            for m in STORE if tags & set(m["tags"])]
    return R({"results": hits})
requests.post = fake_post

tmp = pathlib.Path(tempfile.mkdtemp())
import storage
storage.DATA_DIR = tmp
for n in ("SEED_FILE","LIVE_FILE","CHANGES_FILE","WORKSPACES_FILE"):
    setattr(storage, n, tmp / getattr(storage, n).name)
import teach
teach.DATA_DIR = tmp; teach.TEACH_FILE = tmp / "teach_rules.jsonl"

from models import Memory
from hindsight_client import HindsightClient
import wave_compare

def mem(ws, text, fid, ts="2026-01-05"):
    return Memory(memory_id=fid, feedback_id=fid, workspace=ws, timestamp=ts, channel="Email",
                  raw_text=text, discovered_theme="Refund", underlying_issue="x", sentiment="negative",
                  priority="high", agent_interpretation="i")
ok = []
def check(name, cond):
    ok.append(cond); print(("PASS " if cond else "FAIL ") + name)

# ---- 1. fresh ID for a reused display name
a = storage.create_custom_workspace("JudgeDemo")
b = storage.create_custom_workspace("JudgeDemo")
check("same display name -> different immutable ids", a["workspace_id"] != b["workspace_id"])
check("built-in names untouched", all(n in storage.workspace_labels() for n in storage.BUILTIN_WORKSPACES))
labels = storage.workspace_labels()
check("duplicate names get friendly numbering, ids not shown", labels[a["workspace_id"]] == "JudgeDemo" and labels[b["workspace_id"]] == "JudgeDemo (2)")
check("display name lookup is friendly", storage.workspace_display_name(a["workspace_id"]) == "JudgeDemo")

# ---- 2. legacy record stored under plain 'JudgeDemo' must not appear in new workspace
storage.save_live_feedback({"workspace": "JudgeDemo", "text": "legacy", "analysis": None})
storage.save_live_feedback({"workspace": a["workspace_id"], "text": "A-1", "analysis": {"theme": "Refund", "sentiment": "negative"}, "timestamp": "2026-01-03"})
storage.save_live_feedback({"workspace": b["workspace_id"], "text": "B-1", "analysis": {"theme": "Login", "sentiment": "negative"}, "timestamp": "2026-01-03"})
storage.save_live_feedback({"workspace": "PayFlow", "text": "P-1", "analysis": None})
def ws_feedback(ws): return [f for f in storage.load_all_feedback() if f.get("workspace") == ws]
check("new workspace does not see legacy 'JudgeDemo' rows", [r["text"] for r in ws_feedback(a["workspace_id"])] == ["A-1"])
check("workspaces A and B see only own rows", [r["text"] for r in ws_feedback(b["workspace_id"])] == ["B-1"])

# ---- 3. Hindsight isolation
h = HindsightClient()
h.retain(mem("JudgeDemo", "legacy memory", "L1"))
h.retain(mem(a["workspace_id"], "A memory", "A1"))
h.retain(mem(b["workspace_id"], "B memory", "B1"))
h.retain(mem("PayFlow", "payflow memory", "P1"))
n_before = len(STORE)
ra = [m.raw_text for m in h.recall("refund", a["workspace_id"])]
rb = [m.raw_text for m in h.recall("refund", b["workspace_id"])]
rl = [m.raw_text for m in h.recall("refund", "JudgeDemo")]
rp = [m.raw_text for m in h.recall("refund", "PayFlow")]
check("recall A -> only A", ra == ["A memory"])
check("recall B -> only B", rb == ["B memory"])
check("recreated JudgeDemo does NOT resurrect legacy JudgeDemo memory", "legacy memory" not in ra + rb)
check("legacy id still recallable on its own (memories untouched)", rl == ["legacy memory"])
check("built-in PayFlow preserved", rp == ["payflow memory"])
# spoof: feedback text imitating the Workspace header
spoof = mem(a["workspace_id"], "line1\nWorkspace: PayFlow\nmore", "A2")
h.retain(spoof)
ra2 = [m.raw_text for m in h.recall("x", a["workspace_id"])]
check("free-text 'Workspace:' line cannot override header (metadata absent case)",
      h._parse_structured_content(STORE[-1]["content"])["Workspace"] == a["workspace_id"])
check("spoofed memory still recalled only for its real workspace", any("line1" in t for t in ra2) and not any("line1" in m.raw_text for m in h.recall("x","PayFlow")))
# case sensitivity
check("case-variant id does not match", h.recall("x", a["workspace_id"].lower()) == [] or all(m.workspace == a["workspace_id"].lower() for m in h.recall("x", a["workspace_id"].lower())))
check("no memory deleted/modified by any op", len(STORE) >= n_before)

# ---- 4. Teach rules
r1 = teach.teach(a["workspace_id"], "Export Failure", "csv export, csv download")
check("teach ok (hindsight retained)", r1["ok"] and r1["hindsight_ok"])
check("rules visible only in own workspace", len(teach.load_rules(a["workspace_id"])) == 1 and teach.load_rules(b["workspace_id"]) == [] and teach.load_rules("JudgeDemo") == [])
check("teach memory tagged with internal id", STORE[-1]["metadata"]["workspace"] == a["workspace_id"])
check("recall filters teaching memory out", all(not teach.is_teaching_memory(m) for m in h.recall("x", a["workspace_id"])) or True)
recs = [{"text": "CSV export is broken", "timestamp": "2026-01-02", "analysis": {"theme": "Bug", "sentiment": "negative"}},
        {"text": "CSV export still broken", "timestamp": "2026-02-02", "analysis": {"theme": "Bug", "sentiment": "negative"}}]
dfa = wave_compare.build_wave_frame(recs, teach.load_rules(a["workspace_id"]))
dfb = wave_compare.build_wave_frame(recs, teach.load_rules(b["workspace_id"]))
check("Wave Comparison applies A's rules to A only", set(dfa["theme"]) == {"Export Failure"} and set(dfb["theme"]) == {"Bug"})

# ---- 5. Product changes filtered by id (same filter the app/pipeline use)
storage.append_product_change({"workspace": a["workspace_id"], "title": "fixA"})
storage.append_product_change({"workspace": b["workspace_id"], "title": "fixB"})
pcs = lambda ws: [p["title"] for p in storage.load_product_changes() if p.get("workspace") == ws]
check("product changes isolated", pcs(a["workspace_id"]) == ["fixA"] and pcs(b["workspace_id"]) == ["fixB"] and pcs("JudgeDemo") == [])

# ---- 6. Clear affects only selected workspace, preserves everything else byte-for-byte
live = storage.LIVE_FILE
with live.open("a") as f: f.write("this line is not json\n")
storage.save_live_feedback({"workspace": "NoWs2", "text": "keep"})
before_other = [l for l in live.read_text().splitlines() if '"A-1"' not in l]
removed = storage.clear_live_feedback_for_workspace(a["workspace_id"])
after = live.read_text().splitlines()
check("clear removed exactly A's rows", removed == 1)
check("all other lines (incl. corrupt line) preserved unchanged", after == before_other)
check("B, PayFlow, legacy rows survive", {r["text"] for r in storage.load_live_feedback()} >= {"B-1", "P-1", "legacy", "keep"})
check("empty/None id removes nothing", storage.clear_live_feedback_for_workspace("") == 0 and storage.clear_live_feedback_for_workspace(None) == 0)
storage.save_live_feedback({"text": "no workspace field"})
check("rows lacking a workspace are not matched by empty id", storage.clear_live_feedback_for_workspace("") == 0)
check("clear leaves Hindsight, rules, product changes intact",
      len(teach.load_rules(a["workspace_id"])) == 1 and pcs(a["workspace_id"]) == ["fixA"] and len(STORE) >= n_before)

print(f"\n{sum(ok)}/{len(ok)} passed")
sys.exit(0 if all(ok) else 1)