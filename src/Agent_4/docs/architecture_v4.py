"""Agent_4 architecture diagram — v4.

Reflects the CURRENT state (2026-05-16, late) after the spec-proposer
overhaul: unified SPEC_PROPOSER_SYSTEM prompt, table-based prior-trials
evidence, propose-and-justify `why` field (cite F1 + key=old→new), hard
Phase A → Phase B wall-clock gate at 55%, multi-vectorizer SPARSE_TAIL,
pad_to_max_length + stratify try/except autofixes, quote-tolerance
repair matcher, clear final-submission logging.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(22, 14.5))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(11, 14.05, "Agent_4 Architecture",
        ha="center", va="center", fontsize=24, fontweight="bold")
ax.text(11, 13.55,
        "Autonomous LLM-driven research agent for the Kaggle Disaster Tweets task",
        ha="center", va="center", fontsize=13, style="italic", color="#444")

# --- Palette ---------------------------------------------------------------
COL_LLM = "#FFE6B3";   COL_LLM_EDGE = "#E69500"
COL_ORCH = "#BFE0FF";  COL_ORCH_EDGE = "#1E6FB8"
COL_STORE = "#D5F2D5"; COL_STORE_EDGE = "#338033"
COL_OUT = "#FFD1D1";   COL_OUT_EDGE = "#C44747"
COL_GUARD = "#EAD7FF"; COL_GUARD_EDGE = "#7A2EC9"

def box(x, y, w, h, label, sub="", color=COL_ORCH, edge=COL_ORCH_EDGE,
        fontsize=11, sub_fontsize=8.5, bold=True):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.15",
                          facecolor=color, edgecolor=edge, linewidth=1.4)
    ax.add_patch(rect)
    if sub:
        ax.text(x + w/2, y + h*0.66, label, ha="center", va="center",
                fontsize=fontsize, fontweight=("bold" if bold else "normal"))
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=sub_fontsize, color="#333", style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=("bold" if bold else "normal"))

def arrow(x0, y0, x1, y1, color="#444", lw=1.5, style="-|>",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle=style, mutation_scale=16,
                                 linewidth=lw, color=color,
                                 connectionstyle=connectionstyle))

def label(x, y, text, color="#555", fs=9, bg=True, italic=True):
    bbox = dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.92) if bg else None
    style = "italic" if italic else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color, style=style, bbox=bbox)

# ============================================================================
# Row A — 5 LLM roles
# ============================================================================
ax.text(11, 12.9, "The five LLM roles (Ollama / qwen2.5-coder:14b)",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#666")

llm_y = 11.6
llm_h = 1.0
positions = [
    (0.6,  3.7, "Sweep Planner",   "phase-aware (A=EXPLORE / B=MAX F1)  |  temp=0.4"),
    (4.7,  3.7, "Spec Proposer",   "table prompt | why = cite F1 + key=old→new | temp=0.5"),
    (8.8,  3.7, "Code Generator",  "full training script   |   temp=0.2"),
    (12.9, 3.7, "Repair LLM",      "JSON edit-plan (quote-tolerant)   |   temp=0.2"),
    (17.0, 4.4, "Analyst",         "CONCLUSION / WORKED / FAILED / NEXT   |   temp=0.2"),
]
for x, w, label_text, sub in positions:
    box(x, llm_y, w, llm_h, label_text, sub, color=COL_LLM, edge=COL_LLM_EDGE,
        fontsize=12, sub_fontsize=8.3)

ax.text(11, llm_y - 0.4,
        "5 LLM round-trips per trial. One unified SPEC_PROPOSER_SYSTEM (replaces old SPEC + SEARCH split).",
        ha="center", va="center", fontsize=9.5, color="#666", style="italic")

# ============================================================================
# Row B — Orchestrator pipeline (7 steps)
# ============================================================================
ax.text(11, 10.35, "The orchestrator — deterministic guard rails around each trial",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#666")

pipe_y = 8.8
pipe_w = 2.7
pipe_h = 1.15
gap = 0.35
x0 = 0.5

steps = [
    ("1. Planner\ndecision",     "case-insensitive resolve\n+ cheapest-eligible fallback",  COL_ORCH,  COL_ORCH_EDGE),
    ("2. Spec\nproposal",        "table of prior trials\n+ why field (cite + arrow)",       COL_ORCH,  COL_ORCH_EDGE),
    ("3. Constraint\nengine",    "changed_keys filter\n+ 2-key floor",                      COL_GUARD, COL_GUARD_EDGE),
    ("4. Cross-launch\nveto",    "spec-signature dup—detector\n+ orchestrator-mutate",  COL_GUARD, COL_GUARD_EDGE),
    ("5. Code-gen\n+ autofixes", "validate_spec + autofix:\npad_to_max_length, stratify…", COL_ORCH,  COL_ORCH_EDGE),
    ("6. Sandbox\ndry + full run","sandbox.py + repair\n(quote-tolerant matcher)",          COL_ORCH,  COL_ORCH_EDGE),
    ("7. Analyst",               "structured 4-element\nverdict",                            COL_ORCH,  COL_ORCH_EDGE),
]
xs = []
x = x0
for label_text, sub, c, ec in steps:
    box(x, pipe_y, pipe_w, pipe_h, label_text, sub,
        color=c, edge=ec, fontsize=10.5, sub_fontsize=7.8)
    xs.append(x)
    x += pipe_w + gap

# Sequential arrows
for i in range(len(steps) - 1):
    arrow(xs[i] + pipe_w, pipe_y + pipe_h/2,
          xs[i+1], pipe_y + pipe_h/2,
          color="#1E6FB8", lw=1.7)

# Repair self-loop on step 6
sandbox_x = xs[5]
arrow(sandbox_x + pipe_w * 0.65, pipe_y + pipe_h,
      sandbox_x + pipe_w * 0.35, pipe_y + pipe_h,
      color="#C44747", lw=1.4,
      connectionstyle="arc3,rad=0.9")
label(sandbox_x + pipe_w/2, pipe_y + pipe_h + 0.45,
      "repair loop\n(≤ 4 attempts)",
      color="#C44747", fs=8.5)

# ============================================================================
# Row C — loop-back arrow + Phase A/B gate
# ============================================================================
loop_y_low = 7.05
arrow(xs[-1] + pipe_w/2, pipe_y,
      xs[-1] + pipe_w/2, loop_y_low + 0.2,
      color="#338033", lw=1.6)
arrow(xs[-1] + pipe_w/2, loop_y_low + 0.2,
      xs[0] + pipe_w/2, loop_y_low + 0.2,
      color="#338033", lw=1.6, style="-")
arrow(xs[0] + pipe_w/2, loop_y_low + 0.2,
      xs[0] + pipe_w/2, pipe_y,
      color="#338033", lw=1.6)
label(11, loop_y_low + 0.05,
      "loop until 45-min sweep budget is exhausted   ┃   Phase A (EXPLORE) → Phase B (MAX F1) at 55%·24:48 mark",
      color="#338033", fs=10)

# ============================================================================
# Row D — Persistence (left) + Final submission (right)
# ============================================================================
ax.text(5.5, 6.15, "Persistence + cross-launch memory",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#666")
ax.text(16.5, 6.15, "After the sweep — final submission",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#666")

box(0.5,  4.55, 5.0, 1.25, "Short-term memory",
    "logs/agent4_short_term_memory.json\n20-trial rolling window  |  family-filtered\nspec proposer reads as compact table",
    color=COL_STORE, edge=COL_STORE_EDGE, fontsize=11, sub_fontsize=8.2)
box(5.85, 4.55, 5.0, 1.25, "Per-trial artifacts",
    "runs/agent_4/current/<session>/run_NNN/\nspec.json, train.py, hypothesis.txt,\nmetrics.json, run.log, repair_attempt_*.json",
    color=COL_STORE, edge=COL_STORE_EDGE, fontsize=11, sub_fontsize=8.0)

box(11.5, 4.55, 5.0, 1.25, "Hardcoded submission tail",
    "submit_tails.py  |  multi-vectorizer hstack-aware\nno LLM, no repair, idempotent try/except\nfamily-dispatched: sparse / deep / transformer",
    color=COL_GUARD, edge=COL_GUARD_EDGE, fontsize=11, sub_fontsize=8.0)
box(16.85, 4.55, 4.65, 1.25, "Kaggle CSV",
    "submissions/best_overall_submission.csv\nclear success/fail log line at run end",
    color=COL_OUT, edge=COL_OUT_EDGE, fontsize=11, sub_fontsize=8.5)

# Arrows down to persistence
arrow(xs[1] + pipe_w/2, loop_y_low + 0.2,
      3.0, 5.80,
      color="#338033", lw=1.4)
arrow(xs[3] + pipe_w/2, loop_y_low + 0.2,
      8.35, 5.80,
      color="#338033", lw=1.4)
label(5.8, 6.45, "every trial → save",
      color="#338033", fs=9)

# Memory feeds back into planner
arrow(0.5, 5.1,
      0.05, pipe_y + pipe_h/2,
      color="#338033", lw=1.4,
      connectionstyle="arc3,rad=-0.6")
arrow(0.05, pipe_y + pipe_h/2,
      xs[0], pipe_y + pipe_h * 0.75,
      color="#338033", lw=1.4)
label(0.95, 7.15, "memory feeds\nthe next launch",
      color="#338033", fs=9)

# Final submission arrow chain
arrow(xs[-1] + pipe_w, pipe_y + pipe_h * 0.5,
      11.5 + 2.5, 5.80,
      color="#7A2EC9", lw=1.4,
      connectionstyle="arc3,rad=-0.25")
label(17.6, 7.3, "after sweep:\nrun best on 5k rows",
      color="#7A2EC9", fs=9)
arrow(11.5 + 5.0, 5.1, 16.85, 5.1, color="#C44747", lw=1.8)

# ============================================================================
# Row E — Seven model families
# ============================================================================
ax.text(11, 3.45, "Seven model families the planner can choose from",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#666")

fam_y = 2.15
fam_w = 2.85
fam_h = 1.0
fam_gap = 0.20
families = [
    ("BoW",          "TF-IDF + LogReg"),
    ("BoW_advanced", "word + char n-grams"),
    ("LSTM",         "bidirectional"),
    ("CNN",          "1D Conv"),
    ("EmbeddingDL",  "learned / GloVe + GRU"),
    ("RoBERTa",      "roberta-base"),
    ("BERTweet",     "vinai/bertweet-base"),
]
total_w = len(families) * fam_w + (len(families) - 1) * fam_gap
fx = (22 - total_w) / 2
for label_text, sub in families:
    box(fx, fam_y, fam_w, fam_h, label_text, sub,
        color="#F5F5F5", edge="#666",
        fontsize=11, sub_fontsize=8.5)
    fx += fam_w + fam_gap

arrow(xs[0] + pipe_w/2, pipe_y,
      11, fam_y + fam_h,
      color="#1E6FB8", lw=1.3,
      connectionstyle="arc3,rad=-0.20", style="-|>")

# ============================================================================
# Legend
# ============================================================================
legend_y = 0.4
legend_items = [
    ("LLM call",             COL_LLM,   COL_LLM_EDGE),
    ("Orchestrator step",    COL_ORCH,  COL_ORCH_EDGE),
    ("Guard rail / safety",  COL_GUARD, COL_GUARD_EDGE),
    ("Persistence / memory", COL_STORE, COL_STORE_EDGE),
    ("Output",               COL_OUT,   COL_OUT_EDGE),
]
lx = 1.5
for label_text, c, ec in legend_items:
    ax.add_patch(Rectangle((lx, legend_y), 0.5, 0.45,
                           facecolor=c, edgecolor=ec, linewidth=1.4))
    ax.text(lx + 0.65, legend_y + 0.22, label_text,
            va="center", fontsize=10.5)
    lx += 3.8

# Bottom-right annotation: today's changes
ax.text(11, 0.1,
        "v4 (2026-05-16): unified spec-proposer prompt · table-based prior trials · "
        "why=cite-F1 + key=old→new · Phase A→B gate at 55% · multi-vectorizer tail · "
        "pad_to_max_length + stratify autofixes",
        ha="center", va="center", fontsize=8.5, color="#888", style="italic")

import os
out_path = os.path.join(os.path.dirname(__file__), "architecture_v4.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
