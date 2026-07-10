#!/usr/bin/env python3
"""
Generate a clean, comprehensive sequence diagram for the research-harness
architecture. Outputs SVG to docs/assets/session_flow.svg
"""

import math
import textwrap

# ─── canvas & participants ────────────────────────────────────────────────────
W = 1840

PARTS = [
    ("user",   "User",              90,   "#c8e6c9", "#2e7d32", "#1b5e20"),
    ("cli",    "CLI",               265,  "#bbdefb", "#1565c0", "#0d47a1"),
    ("orch",   "Orchestrator",      470,  "#b2ebf2", "#00838f", "#006064"),
    ("router", "TaskRouter /\nEvolutionLoop", 660, "#e1bee7", "#6a1b9a", "#4a148c"),
    ("llm",    "LLM Client",        855,  "#ffe0b2", "#e65100", "#bf360c"),
    ("agents", "Agents /\nEvaluators", 1045, "#c5cae9", "#283593", "#1a237e"),
    ("search", "Search\nBackends",  1220, "#f8bbd0", "#880e4f", "#560027"),
    ("store",  "Artifact\nStore",   1390, "#dcedc8", "#558b2f", "#33691e"),
    ("world",  "World / Obs /\nDiagnostics", 1570, "#b2dfdb", "#00695c", "#004d40"),
]

LX = {k: x for k, x, *_ in [(p[0], p[2]) for p in PARTS]}
LC = {p[0]: (p[3], p[4], p[5]) for p in PARTS}

BOX_W, BOX_H = 128, 52
TITLE_H = 55          # space for title at top
HEADER_Y = TITLE_H + 8
LL_START = HEADER_Y + BOX_H + 2
MSG_START = LL_START + 42
DY = 40               # vertical spacing between messages

# ─── helpers ─────────────────────────────────────────────────────────────────

def lx(k): return LX[k]


def xml_esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect_el(x, y, w, h, fill, stroke, sw=1.5, rx=5, opacity=1.0, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{d}/>')


def text_el(x, y, s, size=11, fill="#1a202c", anchor="middle",
            bold=False, italic=False, spacing=None):
    style = f"font-family:'Inter','Helvetica Neue',sans-serif;font-size:{size}px;"
    if bold:   style += "font-weight:700;"
    if italic: style += "font-style:italic;"
    if spacing: style += f"letter-spacing:{spacing}em;"
    lines = s.split('\n')
    if len(lines) == 1:
        return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                f'style="{style}" fill="{fill}">{xml_esc(s)}</text>')
    lh = size + 4
    out = [f'<text text-anchor="{anchor}" style="{style}" fill="{fill}">']
    base = y - (len(lines) - 1) * lh / 2
    for i, ln in enumerate(lines):
        out.append(f'  <tspan x="{x:.1f}" y="{base + i*lh:.1f}">{xml_esc(ln)}</tspan>')
    out.append('</text>')
    return '\n'.join(out)


def arrow_el(x1, y, x2, color, is_return=False, label="", fsize=10):
    AL, AW = 9, 4.5
    going_right = x2 > x1
    dash = 'stroke-dasharray="7,3"' if is_return else ''
    parts = []
    # shaft
    parts.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
                 f'stroke="{color}" stroke-width="1.5" {dash}/>')
    # arrowhead
    if going_right:
        pts = f"{x2:.1f},{y:.1f} {x2-AL:.1f},{y-AW:.1f} {x2-AL:.1f},{y+AW:.1f}"
    else:
        pts = f"{x2:.1f},{y:.1f} {x2+AL:.1f},{y-AW:.1f} {x2+AL:.1f},{y+AW:.1f}"
    parts.append(f'<polygon points="{pts}" fill="{color}"/>')
    # label centred above
    if label:
        mx = (x1 + x2) / 2
        bg_w = len(label) * fsize * 0.58 + 10
        bg_h = fsize + 5
        parts.append(rect_el(mx - bg_w/2, y - bg_h - 3, bg_w, bg_h,
                              "#fafbfc", "none", 0))
        parts.append(text_el(mx, y - 6, label, size=fsize, fill=color))
    return '\n'.join(parts)


def self_loop(x, y, color, label="", fsize=10, loop_w=46):
    r = 14
    # small bump to the right
    p = (f"M {x:.1f} {y-8:.1f} "
         f"C {x+loop_w:.1f} {y-8:.1f} {x+loop_w:.1f} {y+12:.1f} {x:.1f} {y+12:.1f}")
    AL = 7
    pts = f"{x:.1f},{y+12:.1f} {x+AL:.1f},{y+6:.1f} {x+AL:.1f},{y+18:.1f}"
    parts = [
        f'<path d="{p}" stroke="{color}" stroke-width="1.4" fill="none"/>',
        f'<polygon points="{pts}" fill="{color}"/>',
    ]
    if label:
        parts.append(text_el(x + loop_w + 6, y + 4, label, size=fsize,
                             fill=color, anchor="start"))
    return '\n'.join(parts)


def section_bg(x1, y1, x2, y2, fill, opacity=0.16):
    w, h = x2 - x1, y2 - y1
    return rect_el(x1, y1, w, h, fill, "none", 0, rx=8, opacity=opacity)


def section_frame(x1, y1, x2, y2, fill, stroke, label="", label_fill="white"):
    w, h = x2 - x1, y2 - y1
    parts = [rect_el(x1, y1, w, h, fill, stroke, 1.8, rx=8,
                     opacity=0.14, dash="9,4")]
    if label:
        lw = len(label) * 7.2 + 14
        parts.append(rect_el(x1, y1, lw, 18, stroke, "none", 0, rx=3))
        parts.append(text_el(x1 + lw/2, y1 + 13, label, size=9,
                             fill=label_fill, bold=True))
    return '\n'.join(parts)


# ─── build message list ───────────────────────────────────────────────────────
# Each entry: ("msg", y, from, to, label, is_return, color)
#             ("self", y, key, label, color)
#             ("note", y, key, text)   ← right-justified note box

items = []
sections = {}   # name -> (y1, y2) filled in as we go
y = MSG_START


def M(frm, to, label, ret=False, col=None):
    global y
    c = col or LC[frm][2]
    items.append(("msg", y, frm, to, label, ret, c))
    y += DY


def S(key, label, col=None):
    global y
    c = col or LC[key][2]
    items.append(("self", y, key, label, c))
    y += DY + 6   # self-loops are taller


def gap(n=1):
    global y
    y += DY * n * 0.4


def section_mark(name):
    sections[name] = y


# ── Startup ──────────────────────────────────────────────────────────────────
section_mark("setup_y1")
M("user",  "cli",   "goal · flags · interactive setup")
M("cli",   "orch",  "argv / output_dir / session_id")
M("orch",  "world", "query prior traces + dedup keys")
M("world", "orch",  "provenance history + run_observability", ret=True)
M("orch",  "llm",   "interpret_goal()")
M("llm",   "orch",  "TaskType + ResearchPlan + stopping_signals", ret=True)
M("orch",  "llm",   "create_source_strategy()")
M("llm",   "orch",  "SourceStrategy[] (retriever · queries · limits)", ret=True)
M("orch",  "store", "init RunRecord · ArtifactStore · SessionStore")
section_mark("setup_y2")

# ── Task Routing ──────────────────────────────────────────────────────────────
gap()
section_mark("routing_y1")
M("orch",   "router", "decide(goal, task_type, mode)", col="#6a1b9a")
M("router", "llm",    "classify task via LLM (if live)", col="#6a1b9a")
M("llm",    "router", "TaskIngestionDecision", ret=True, col="#6a1b9a")
M("router", "orch",   "product_agent: research / optimize / challenge", ret=True, col="#6a1b9a")
section_mark("routing_y2")

# ── Session loop ──────────────────────────────────────────────────────────────
gap()
section_mark("session_y1")
gap(0.5)

# ── Research Path ─────────────────────────────────────────────────────────────
section_mark("research_y1")
M("orch",   "agents", "[parallel] LiteratureAgent.execute(queries)", col="#283593")
M("agents", "search", "search(query, limit)", col="#880e4f")
M("search", "agents", "Document[] + relevance_scores", ret=True, col="#880e4f")
M("agents", "store",  "add_source() · add_claim()  (URL/title dedup)", col="#558b2f")
M("agents", "orch",   "AgentTrace (prompt · tokens · cost · status)", ret=True, col="#283593")
gap(0.5)
M("orch",   "agents", "HypothesisAgent.execute(claims)", col="#283593")
M("agents", "llm",    "generate_hypotheses(relevant_claims)", col="#e65100")
M("llm",    "agents", "Hypothesis[] (novelty · testability · next_exp)", ret=True, col="#e65100")
M("agents", "store",  "add_hypothesis() · add_open_question()", col="#558b2f")
gap(0.5)
M("orch",   "agents", "CriticAgent.execute(claims, hypotheses)", col="#283593")
M("agents", "store",  "add_contradiction() · flag_low_confidence_claims()", col="#558b2f")
M("agents", "orch",   "AgentTrace (contradictions found)", ret=True, col="#283593")
gap(0.5)
M("orch",   "agents", "SynthesisAgent.execute(all artifacts)", col="#283593")
M("agents", "llm",    "write_report(sources · claims · hyps · contrs)", col="#e65100")
M("llm",    "agents", "final_report.md content", ret=True, col="#e65100")
M("agents", "store",  "write_report() · fabricated_source_check()", col="#558b2f")
section_mark("research_y2")

gap()

# ── Optimize / Challenge Path ──────────────────────────────────────────────────
section_mark("optimize_y1")
M("orch",   "router", "EvolutionaryOuterLoop.run(max_outer_iterations)", col="#6a1b9a")
section_mark("loop_y1")
M("router", "llm",    "propose_variants(kind, parent_ids, temperature)", col="#e65100")
M("llm",    "router", "Variant[] (code / query payload)", ret=True, col="#e65100")
M("router", "store",  "add_variant(outer_iter · kind · payload · parents)", col="#558b2f")
M("router", "agents", "[parallel] evaluate(variant) via inner loop", col="#283593")
M("agents", "store",  "add_variant_evaluation(score · metrics · judge_scores)", col="#558b2f")
M("router", "store",  "add_evolution_round(best_score · termination_signal)", col="#558b2f")
S("router", "PlateauDetector: rotate_retriever / boost_temperature / mutate", col="#6a1b9a")
section_mark("loop_y2")
M("router", "orch",   "termination: score_threshold / plateau / budget_hit", ret=True, col="#6a1b9a")
M("router", "store",  "write optimization_result.json · optimal_code.py", col="#558b2f")
section_mark("optimize_y2")

gap(0.3)
section_mark("session_y2")

# ── Finalization ───────────────────────────────────────────────────────────────
gap()
section_mark("finalize_y1")
M("orch",  "store", "write_run_state() · cost.json · cost_events.json")
M("orch",  "store", "diagnose_snapshot() → failure_taxonomy · score_patterns")
M("store", "world", "mirror rows → world_model.sqlite · run_observability", col="#00695c")
M("world", "orch",  "updated dedup_keys · provenance_edges", ret=True, col="#00695c")
section_mark("finalize_y2")

gap()

# ── Phases 6-7: Evolution Gate ────────────────────────────────────────────────
section_mark("evolution_y1")
M("orch",  "world", "P6: benchmark vs history · stage harness_changes", col="#00695c")
M("world", "orch",  "P7: accept winners · persist prompt_versions · defaults", ret=True, col="#00695c")
M("orch",  "store", "persist prompt_versions for next session", col="#558b2f")
gap()
M("world", "orch",  "next invocation: reads promoted world state (CLI overlay)", ret=True, col="#004d40")
section_mark("evolution_y2")

gap()
M("cli",   "user",  "deliver artifacts · benchmarks · diagrams", ret=True, col="#1565c0")
section_mark("deliver_y")

total_h = y + 60

# ─── render ──────────────────────────────────────────────────────────────────

out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" '
    f'viewBox="0 0 {W} {total_h}">',
    # global background
    f'<rect width="{W}" height="{total_h}" fill="#f7f8fa"/>',

    # title
    text_el(W/2, 26, "Research Harness — Complete Session Flow",
            size=21, fill="#1a202c", bold=True),
    text_el(W/2, 44, "Startup · Task Routing · Session Loop (Research + Optimize/Challenge) · "
            "Finalization · Phases 6–7 Evolution Gate",
            size=11, fill="#718096"),
]

PAD_X = 16

def sy(name): return sections[name] - 22
def ey(name): return sections[name] + 8

# ── section backgrounds (painted before arrows) ───────────────────────────────
out.append(section_bg(PAD_X, sy("setup_y1"),    W - PAD_X, ey("setup_y2"),    "#1e88e5"))
out.append(section_bg(PAD_X, sy("routing_y1"),  W - PAD_X, ey("routing_y2"),  "#9c27b0"))
out.append(section_frame(PAD_X, sy("session_y1"), W - PAD_X, ey("session_y2"),
           "#9e9e9e", "#616161", label="SESSION LOOP  (one run)"))
out.append(section_bg(PAD_X+14, sy("research_y1"),  W - PAD_X-14, ey("research_y2"),  "#43a047", opacity=0.11))
out.append(section_bg(PAD_X+14, sy("optimize_y1"),  W - PAD_X-14, ey("optimize_y2"),  "#fb8c00", opacity=0.11))
out.append(section_frame(PAD_X+22, sy("loop_y1"), W - PAD_X-22, ey("loop_y2"),
           "#fb8c00", "#f57c00", label="LOOP  [0 .. max_outer_iterations]"))
out.append(section_bg(PAD_X, sy("finalize_y1"), W - PAD_X, ey("finalize_y2"), "#7e57c2"))
out.append(section_bg(PAD_X, sy("evolution_y1"), W - PAD_X, ey("deliver_y"),  "#00acc1"))

# ── section labels ─────────────────────────────────────────────────────────────
def sec_label(txt, y, col):
    out.append(text_el(PAD_X + 6, y + 14, txt, size=9, fill=col,
                       bold=True, anchor="start", spacing=0.07))

sec_label("STARTUP & INITIALIZATION",         sy("setup_y1"),    "#0d47a1")
sec_label("TASK ROUTING",                     sy("routing_y1"),  "#4a148c")
sec_label("RESEARCH PATH",                    sy("research_y1"), "#1b5e20")
sec_label("OPTIMIZE / CHALLENGE PATH",        sy("optimize_y1"), "#bf360c")
sec_label("FINALIZATION",                     sy("finalize_y1"), "#4527a0")
sec_label("PHASES 6-7  EVOLUTION GATE + PROMOTION", sy("evolution_y1"), "#006064")

# ── lifeline headers ───────────────────────────────────────────────────────────
for key, label, lx_val, fill, stroke, text_col in PARTS:
    cx = lx_val - BOX_W // 2
    out.append(rect_el(cx, HEADER_Y, BOX_W, BOX_H, fill, stroke, sw=2))
    lines = label.split('\n')
    lh = 14
    base = HEADER_Y + (BOX_H - len(lines)*lh) / 2 + lh - 1
    for i, ln in enumerate(lines):
        out.append(text_el(lx_val, base + i*lh, ln, size=11,
                           fill=text_col, bold=True))
    # dashed lifeline
    out.append(f'<line x1="{lx_val}" y1="{HEADER_Y+BOX_H}" x2="{lx_val}" '
               f'y2="{total_h - 50}" stroke="{stroke}" stroke-width="1.2" '
               f'stroke-dasharray="6,5" opacity="0.45"/>')

# ── messages ───────────────────────────────────────────────────────────────────
for item in items:
    kind = item[0]
    if kind == "msg":
        _, yp, frm, to, label, is_ret, col = item
        out.append(arrow_el(lx(frm), yp, lx(to), col, is_ret, label))
    elif kind == "self":
        _, yp, key, label, col = item
        out.append(self_loop(lx(key), yp, col, label))

# ── legend ─────────────────────────────────────────────────────────────────────
LX_BOX = 16
LY_BOX = total_h - 46
out.append(rect_el(LX_BOX, LY_BOX, 640, 38, "white", "#cbd5e0", sw=1, rx=6))
out.append(text_el(LX_BOX + 10, LY_BOX + 14, "Legend:", size=10, fill="#2d3748",
                   bold=True, anchor="start"))

def leg_arrow(offset_x, dashed=False, label=""):
    col = "#374151"
    dash = ' stroke-dasharray="5,3"' if dashed else ""
    o = [
        f'<line x1="{LX_BOX+offset_x}" y1="{LY_BOX+22}" '
        f'x2="{LX_BOX+offset_x+30}" y2="{LY_BOX+22}" '
        f'stroke="{col}" stroke-width="1.5"{dash}/>',
        f'<polygon points="{LX_BOX+offset_x+30},{LY_BOX+22} '
        f'{LX_BOX+offset_x+22},{LY_BOX+18} {LX_BOX+offset_x+22},{LY_BOX+26}" fill="{col}"/>',
    ]
    if label:
        o.append(text_el(LX_BOX + offset_x + 36, LY_BOX + 26, label, size=9,
                         fill="#4a5568", anchor="start"))
    return "\n".join(o)

out.append(leg_arrow(90,  dashed=False, label="solid = request / call"))
out.append(leg_arrow(340, dashed=True,  label="dashed = return / response"))

# footer
out.append(text_el(W - 16, total_h - 12, "research-harness · generated 2026-05-09",
                   size=9, fill="#a0aec0", anchor="end"))

out.append('</svg>')

svg_out = '\n'.join(out)
out_path = "/Users/michaelzoubkoff/Documents/research-harness/docs/assets/session_flow.svg"
with open(out_path, "w") as f:
    f.write(svg_out)
print(f"Written {len(svg_out):,} bytes → {out_path}")
