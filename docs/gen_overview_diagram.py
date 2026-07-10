#!/usr/bin/env python3
"""
High-level architecture sequence diagram — research-harness.
House style: white bg, muted neutrals, single orange accent (#D97843),
technical drawing precision, generous negative space.
Outputs: docs/assets/architecture_overview.svg + .png

Changes vs v1:
- CLI removed (noise; User talks directly to Orchestrator)
- Run state is updated as actions are observed, not used as a predefined plan
- Evolutionary loop shown with explicit loop frame
"""

import math

# ─── Canvas & palette ────────────────────────────────────────────────────────
W       = 1380
BG      = "#FFFFFF"
INK     = "#1C1C1E"
NEUTRAL = "#6B7280"
RULE    = "#D1D5DB"
ACCENT  = "#D97843"   # used for SESSION LOOP frame only

# Lifeline x-centres  (7 participants, no CLI)
LX = {
    "user":    80,
    "orch":    270,
    "router":  470,
    "llm":     660,
    "agents":  850,
    "store":   1050,
    "world":   1260,
}

LABEL = {
    "user":   "User",
    "orch":   "Orchestrator",
    "router": "TaskRouter /\nEvolutionLoop",
    "llm":    "LLM Client",
    "agents": "Agents /\nEvaluators",
    "store":  "Artifact\nStore",
    "world":  "World / Obs /\nDiagnostics",
}

BOX_W, BOX_H = 120, 46
TITLE_H  = 62
HEADER_Y = TITLE_H + 4
LL_START = HEADER_Y + BOX_H + 2
MSG_START = LL_START + 52
DY = 52
LL_END_EXTRA = 50

# ─── SVG helpers ─────────────────────────────────────────────────────────────

def lx(k): return LX[k]

def esc(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def txt(x, y, s, size=11, fill=INK, anchor="middle", bold=False,
        tracking=None, opacity=1.0):
    style = ("font-family:-apple-system,'SF Pro Text','Inter',"
             "'Helvetica Neue',sans-serif;"
             f"font-size:{size}px;")
    if bold:    style += "font-weight:600;"
    if tracking: style += f"letter-spacing:{tracking}em;"
    op = f' opacity="{opacity}"' if opacity < 1 else ""
    lines = s.split('\n')
    if len(lines) == 1:
        return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                f'style="{style}" fill="{fill}"{op}>{esc(s)}</text>')
    lh = size + 4
    base = y - (len(lines)-1)*lh/2
    o = [f'<text text-anchor="{anchor}" style="{style}" fill="{fill}"{op}>']
    for i, ln in enumerate(lines):
        o.append(f'  <tspan x="{x:.1f}" y="{base+i*lh:.1f}">{esc(ln)}</tspan>')
    o.append('</text>')
    return '\n'.join(o)

def rect(x, y, w, h, fill, stroke="none", sw=1, rx=4, opacity=1.0, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"{d}/>')

def arrow(x1, y, x2, color=INK, dashed=False, label="", fsize=10.5):
    AL, AW = 8, 4
    going_r = x2 > x1
    dash = 'stroke-dasharray="6,3"' if dashed else ''
    tip = x2
    w1x = (x2-AL) if going_r else (x2+AL);  w1y = y-AW
    w2x = w1x;                                w2y = y+AW
    parts = [
        f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{tip:.1f}" y2="{y:.1f}" '
        f'stroke="{color}" stroke-width="1.4" {dash}/>',
        f'<polygon points="{tip:.1f},{y:.1f} {w1x:.1f},{w1y:.1f} '
        f'{w2x:.1f},{w2y:.1f}" fill="{color}"/>',
    ]
    if label:
        mx = (x1 + x2) / 2
        parts.append(txt(mx, y - 7, label, size=fsize, fill=color))
    return '\n'.join(parts)

def self_loop(x, y, color, label="", fsize=10.5, w=44):
    p = (f"M {x:.1f} {y-8:.1f} C {x+w:.1f} {y-8:.1f} "
         f"{x+w:.1f} {y+12:.1f} {x:.1f} {y+12:.1f}")
    pts = f"{x:.1f},{y+12:.1f} {x+7:.1f},{y+5:.1f} {x+7:.1f},{y+19:.1f}"
    parts = [
        f'<path d="{p}" stroke="{color}" stroke-width="1.4" fill="none"/>',
        f'<polygon points="{pts}" fill="{color}"/>',
    ]
    if label:
        parts.append(txt(x + w + 6, y + 4, label, size=fsize,
                         fill=color, anchor="start"))
    return '\n'.join(parts)

def fill_bg(x1, y1, x2, y2, fill, opacity=0.07):
    return rect(x1, y1, x2-x1, y2-y1, fill, "none", 0, rx=6, opacity=opacity)

def frame_box(x1, y1, x2, y2, stroke, label="", label_fill="white",
              fill_opacity=0.12, dash="8,4"):
    w, h = x2-x1, y2-y1
    parts = [rect(x1, y1, w, h, stroke, stroke, 1.8, rx=6,
                  opacity=fill_opacity, dash=dash)]
    if label:
        lw = len(label) * 6.8 + 16
        parts.append(rect(x1, y1-1, lw, 17, stroke, "none", 0, rx=3))
        parts.append(txt(x1 + lw/2, y1 + 12, label, size=9,
                         fill=label_fill, bold=True, tracking=0.08))
    return '\n'.join(parts)

def sec_label(x, y, s, col=NEUTRAL):
    return txt(x, y, s, size=8.5, fill=col, anchor="start",
               bold=True, tracking=0.1)

# ─── Layout ──────────────────────────────────────────────────────────────────
items   = []
markers = {}
y_cur = [MSG_START]

def M(frm, to, label, ret=False, col=INK):
    items.append(("msg", y_cur[0], frm, to, label, ret, col))
    y_cur[0] += DY

def S(key, label, col=NEUTRAL):
    items.append(("self", y_cur[0], key, label, col))
    y_cur[0] += DY + 8

def gap(n=1.0):
    y_cur[0] += int(DY * n * 0.45)

def mark(name):
    markers[name] = y_cur[0]

# ── STARTUP ───────────────────────────────────────────────────────────────────
mark("startup_y1")
M("user",  "orch",  "goal + flags")
M("orch",  "world", "load prior context + dedup keys")
M("world", "orch",  "run history · provenance", ret=True, col=NEUTRAL)
M("orch",  "llm",   "plan + source strategy")
M("llm",   "orch",  "ResearchPlan + SourceStrategy", ret=True, col=NEUTRAL)
mark("startup_y2")

gap()

# ── TASK ROUTING ──────────────────────────────────────────────────────────────
mark("routing_y1")
M("orch",   "router", "classify task")
M("router", "orch",   "research / optimize / challenge", ret=True, col=NEUTRAL)
mark("routing_y2")

gap()

# ── SESSION LOOP ──────────────────────────────────────────────────────────────
mark("session_y1")
gap(0.6)

# ── Research path ─────────────────────────────────────────────────────────────
mark("res_y1")
M("orch",   "agents", "run pipeline  [Literature → Hypothesis → Critic → Synthesis]")
M("agents", "store",  "persist sources · claims · hypotheses · traces")
M("agents", "orch",   "AgentTraces · scores", ret=True, col=NEUTRAL)
mark("res_y2")

gap(0.8)

# ── Optimize / Challenge path ─────────────────────────────────────────────────
mark("opt_y1")
M("orch",   "router", "EvolutionaryOuterLoop.run()")

mark("evo_y1")   # ← evolutionary loop frame starts here
M("router", "llm",   "propose variants  [code / query]")
M("llm",    "router", "Variant[]", ret=True, col=NEUTRAL)
M("router", "store",  "persist variants · evaluations · rounds")
S("router", "PlateauDetector → rotate retriever / boost temp", col=NEUTRAL)
mark("evo_y2")   # ← evolutionary loop frame ends here

M("router", "orch",  "best result · termination signal", ret=True, col=NEUTRAL)
mark("opt_y2")

gap(0.4)
mark("session_y2")

gap()

# ── FINALIZATION ──────────────────────────────────────────────────────────────
mark("final_y1")
M("orch",  "store", "finalize: costs · diagnostics · report · run state")
M("store", "world", "mirror → world_model.sqlite · observability")
mark("final_y2")

gap()

# ── PHASES 6-7 ────────────────────────────────────────────────────────────────
mark("p67_y1")
M("orch",  "world", "P6 · benchmark vs history · stage harness changes")
M("world", "orch",  "P7 · accept winners · promote prompt_versions", ret=True, col=NEUTRAL)
mark("p67_y2")

gap()
M("orch", "user", "deliver artifacts · benchmarks · diagrams", ret=True, col=NEUTRAL)
mark("end_y")

total_h = y_cur[0] + 70
LL_END  = total_h - LL_END_EXTRA

# ─── Render ───────────────────────────────────────────────────────────────────
PAD = 14

def sy(n): return markers[n] - 20
def ey(n): return markers[n] + 10

out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" '
    f'viewBox="0 0 {W} {total_h}">',
    rect(0, 0, W, total_h, BG),
    rect(0, 0, W, TITLE_H + BOX_H + 8, "#F9FAFB", "none", 0, rx=0),
    f'<line x1="0" y1="{TITLE_H+BOX_H+8:.0f}" x2="{W}" '
    f'y2="{TITLE_H+BOX_H+8:.0f}" stroke="{RULE}" stroke-width="1"/>',
    txt(W/2, 28, "Research Harness — Session Architecture",
        size=20, fill=INK, bold=True),
    txt(W/2, 48, "Startup · Task Routing · Session Loop · Finalization · Phases 6–7",
        size=11, fill=NEUTRAL),
]

# ── section fills ─────────────────────────────────────────────────────────────
out.append(fill_bg(PAD, sy("startup_y1"), W-PAD, ey("routing_y2"), "#6B7280", opacity=0.05))
out.append(fill_bg(PAD, sy("session_y1"), W-PAD, ey("session_y2"), ACCENT,    opacity=0.07))
out.append(fill_bg(PAD+12, sy("res_y1"),  W-PAD-12, ey("res_y2"),  "#6B7280", opacity=0.06))
out.append(fill_bg(PAD+12, sy("opt_y1"),  W-PAD-12, ey("opt_y2"),  "#6B7280", opacity=0.06))
out.append(fill_bg(PAD, sy("final_y1"),   W-PAD, ey("final_y2"),   "#6B7280", opacity=0.05))
out.append(fill_bg(PAD, sy("p67_y1"),     W-PAD, ey("end_y")+20,  "#6B7280", opacity=0.05))

# SESSION LOOP — accent frame
out.append(frame_box(PAD, sy("session_y1"), W-PAD, ey("session_y2"),
                     ACCENT, label="SESSION LOOP  (one run)"))

# EVOLUTIONARY LOOP — nested INK frame
out.append(frame_box(PAD+20, sy("evo_y1")-4, W-PAD-20, ey("evo_y2")+4,
                     INK, label="EVOLUTIONARY LOOP  [0 … N iterations]",
                     label_fill="white", fill_opacity=0.04, dash="5,4"))

# ── section labels ─────────────────────────────────────────────────────────────
LPAD = PAD + 6
out.append(sec_label(LPAD, sy("startup_y1")+14, "STARTUP",                   INK))
out.append(sec_label(LPAD, sy("routing_y1")+14, "TASK ROUTING",              INK))
out.append(sec_label(LPAD+12, sy("res_y1")+14,  "RESEARCH PATH",             NEUTRAL))
out.append(sec_label(LPAD+12, sy("opt_y1")+14,  "OPTIMIZE / CHALLENGE PATH", NEUTRAL))
out.append(sec_label(LPAD, sy("final_y1")+14,   "FINALIZATION",              INK))
out.append(sec_label(LPAD, sy("p67_y1")+14,     "PHASES 6–7  EVOLUTION GATE", INK))

# ── lifeline headers ──────────────────────────────────────────────────────────
for key in LX:
    x  = lx(key)
    cx = x - BOX_W // 2
    lbl = LABEL[key]
    out.append(rect(cx, HEADER_Y, BOX_W, BOX_H, BG, INK, sw=1.2, rx=5))
    lines = lbl.split('\n')
    lh    = 13
    base  = HEADER_Y + (BOX_H - len(lines)*lh)/2 + lh - 1
    for i, ln in enumerate(lines):
        out.append(txt(x, base+i*lh, ln, size=10.5, fill=INK, bold=True))
    out.append(f'<line x1="{x:.1f}" y1="{HEADER_Y+BOX_H:.0f}" '
               f'x2="{x:.1f}" y2="{LL_END:.0f}" '
               f'stroke="{RULE}" stroke-width="1" stroke-dasharray="5,5"/>')

# ── messages ──────────────────────────────────────────────────────────────────
for item in items:
    kind = item[0]
    if kind == "msg":
        _, yp, frm, to, label, is_ret, col = item
        out.append(arrow(lx(frm), yp, lx(to), col, is_ret, label))
    elif kind == "self":
        _, yp, key, label, col = item
        out.append(self_loop(lx(key), yp, col, label))

# ── legend ────────────────────────────────────────────────────────────────────
LGX, LGY = PAD, total_h - 38
out.append(rect(LGX, LGY, 520, 28, BG, RULE, sw=0.8, rx=4))
out.append(txt(LGX+10, LGY+18, "Legend:", size=9, fill=NEUTRAL,
               bold=True, anchor="start"))
for ox, dashed, lbl, col in [
        (80,  False, "solid = call / request",      INK),
        (250, True,  "dashed = return / response",  NEUTRAL)]:
    d = ' stroke-dasharray="5,3"' if dashed else ""
    out.append(f'<line x1="{LGX+ox}" y1="{LGY+16}" x2="{LGX+ox+22}" '
               f'y2="{LGY+16}" stroke="{col}" stroke-width="1.4"{d}/>')
    out.append(f'<polygon points="{LGX+ox+22},{LGY+16} {LGX+ox+15},{LGY+12} '
               f'{LGX+ox+15},{LGY+20}" fill="{col}"/>')
    out.append(txt(LGX+ox+28, LGY+20, lbl, size=9, fill=col, anchor="start"))

out.append(rect(LGX+460, LGY+9, 10, 10, ACCENT, "none", 0, rx=2))
out.append(txt(LGX+475, LGY+19, "session loop", size=9, fill=NEUTRAL, anchor="start"))

out.append(txt(W-PAD, total_h-10, "research-harness · architecture overview",
               size=9, fill=RULE, anchor="end"))
out.append('</svg>')

svg = '\n'.join(out)
out_path = ("/Users/michaelzoubkoff/Documents/research-harness/"
            "docs/assets/architecture_overview.svg")
with open(out_path, "w") as f:
    f.write(svg)
print(f"Written {len(svg):,} bytes → {out_path}")
