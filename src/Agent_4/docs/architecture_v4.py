"""Build a clean PNG diagram of the Agent_4 architecture.

Layout strategy:
  Row A (top)    — 5 LLM roles in a single horizontal band
  Row B          — 7-step orchestrator pipeline (left→right)
  Row C          — feedback loop arrow (loop-back to planner)
  Row D          — Persistence (left) + Final submission (right)
  Row E (bottom) — 7 model families the planner can choose from

All coordinates are hand-placed so no two boxes overlap and arrows never cross
boxes. Canvas is 22×14 to leave breathing room.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(11, 13.55, "Agent_4 Architecture",
        ha="center", va="center", fontsize=24, fontweight="bold")
ax.text(11, 13.05,
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
        ax.text(x + w/2, y + h*0.65, label, ha="center", va="center",
                fontsize=fontsize, fontweight=("bold" if bold else "normal"))
        ax.text(x + w/2, y + h*0.27, sub, ha="center", va="center",
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
ax.text(11, 12.4, "The five LLM roles (Ollama / qwen2.5-coder:14b)",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#666")

llm_y = 11.1
llm_h = 1.0
positions = [
    (0.6,  3.7, "Sweep Planner",   "picks family   |   temp=0.2"),
    (4.7,  3.7, "Spec Proposer",   "hypothesis + changed_keys + values   |   temp=0.5"),
    (8.8,  3.7, "Code Generator",  "full training script   |   temp=0.2"),
    (12.9, 3.7, "Repair LLM",      "JSON edit-plan   |   temp=0.2"),
    (17.0, 4.4, "Analyst",         "CONCLUSION / WORKED / FAILED / NEXT   |   temp=0.2"),
]
for x, w, label_text, sub in positions:
    box(x, llm_y, w, llm_h, label_text, sub, color=COL_LLM, edge=COL_LLM_EDGE,
        fontsize=12, sub_fontsize=8.5)

ax.text(11, llm_y - 0.4,
        "one trial = 5 LLM round-trips, left-to-right through the pipeline below",
        ha="center", va="center", fontsize=9.5, color="#666", style="italic")

# ============================================================================
# Row B — Orchestrator pipeline (7 steps)
# ============================================================================
ax.text(11, 9.85, "The orchestrator — deterministic guard rails around each trial",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#666")

pipe_y = 8.3
pipe_w = 2.7
pipe_h = 1.15
gap = 0.35
x0 = 0.5

steps = [
    ("1. Planner\ndecision",   "sweep_planner.py",                COL_ORCH,  COL_ORCH_EDGE),
    ("2. Spec\nproposal",      "generate_spec.py / search.py",    COL_ORCH,  COL_ORCH_EDGE),
    ("3. Constraint\nengine",  "changed_keys filter\n+ 2-key floor", COL_GUARD, COL_GUARD_EDGE),
    ("4. Cross-launch\nveto",  "agent.py:execute_family",         COL_GUARD, COL_GUARD_EDGE),
    ("5. Code-gen\n+ validate","agent.py + validate_spec.py",     COL_ORCH,  COL_ORCH_EDGE),
    ("6. Sandbox\ndry + full run","sandbox.py\n60s dry + 1000s full", COL_ORCH, COL_ORCH_EDGE),
    ("7. Analyst",             "structured 4-element\nverdict",   COL_ORCH,  COL_ORCH_EDGE),
]
xs = []
x = x0
for label_text, sub, c, ec in steps:
    box(x, pipe_y, pipe_w, pipe_h, label_text, sub,
        color=c, edge=ec, fontsize=10.5, sub_fontsize=8.0)
    xs.append(x)
    x += pipe_w + gap

# Sequential arrows between pipeline stages
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
# Row C — loop-back arrow (long curve under the pipeline, well clear)
# ============================================================================
loop_y_low = 6.55
# Down from analyst
arrow(xs[-1] + pipe_w/2, pipe_y,
      xs[-1] + pipe_w/2, loop_y_low + 0.2,
      color="#338033", lw=1.6)
# Horizontal along loop_y_low
arrow(xs[-1] + pipe_w/2, loop_y_low + 0.2,
      xs[0] + pipe_w/2, loop_y_low + 0.2,
      color="#338033", lw=1.6, style="-")
# Up into planner
arrow(xs[0] + pipe_w/2, loop_y_low + 0.2,
      xs[0] + pipe_w/2, pipe_y,
      color="#338033", lw=1.6)
label(11, loop_y_low + 0.05,
      "loop until 45-min sweep budget is exhausted",
      color="#338033", fs=10)

# ============================================================================
# Row D — Persistence (left) + Final submission (right)
# ============================================================================
ax.text(5.5, 5.65, "Persistence + cross-launch memory",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#666")
ax.text(16.5, 5.65, "After the sweep — final submission",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#666")

# Persistence boxes (bottom-left half)
box(0.5,  4.05, 5.0, 1.25, "Short-term memory",
    "logs/agent4_short_term_memory.json\n20-trial rolling window  |  committed to git\nfresh clones inherit the history",
    color=COL_STORE, edge=COL_STORE_EDGE, fontsize=11, sub_fontsize=8.5)
box(5.85, 4.05, 5.0, 1.25, "Per-trial artifacts",
    "runs/agent_4/current/<session>/run_NNN/\nspec.json, train.py, hypothesis.txt,\nmetrics.json, run.log, repair_attempt_*.json",
    color=COL_STORE, edge=COL_STORE_EDGE, fontsize=11, sub_fontsize=8.0)

# Final-submission boxes (bottom-right half)
box(11.5, 4.05, 5.0, 1.25, "Hardcoded submission tail",
    "submit_tails.py\nno LLM, no repair, idempotent try/except wrap\nappended to best_train.py",
    color=COL_GUARD, edge=COL_GUARD_EDGE, fontsize=11, sub_fontsize=8.0)
box(16.85, 4.05, 4.65, 1.25, "Kaggle CSV",
    "submissions/best_overall_submission.csv\n+ optional auto-upload via Kaggle CLI",
    color=COL_OUT, edge=COL_OUT_EDGE, fontsize=11, sub_fontsize=8.5)

# Arrows from loop bar down to persistence (vertical, not crossing boxes)
arrow(xs[1] + pipe_w/2, loop_y_low + 0.2,
      3.0, 5.30,
      color="#338033", lw=1.4)
arrow(xs[3] + pipe_w/2, loop_y_low + 0.2,
      8.35, 5.30,
      color="#338033", lw=1.4)
label(5.8, 5.95, "every trial → save",
      color="#338033", fs=9)

# Arrow from short-term memory feeding back into planner (clean curve on the far left)
arrow(0.5, 4.6,
      0.05, pipe_y + pipe_h/2,
      color="#338033", lw=1.4,
      connectionstyle="arc3,rad=-0.6")
arrow(0.05, pipe_y + pipe_h/2,
      xs[0], pipe_y + pipe_h * 0.75,
      color="#338033", lw=1.4)
label(0.95, 6.65, "memory feeds\nthe next launch",
      color="#338033", fs=9)

# Final submission arrow chain (after sweep ends)
arrow(xs[-1] + pipe_w, pipe_y + pipe_h * 0.5,
      11.5 + 2.5, 5.30,
      color="#7A2EC9", lw=1.4,
      connectionstyle="arc3,rad=-0.25")
label(17.6, 6.8, "after sweep:\nrun best on 5k rows",
      color="#7A2EC9", fs=9)
arrow(11.5 + 5.0, 4.6, 16.85, 4.6, color="#C44747", lw=1.8)

# ============================================================================
# Row E — Seven model families
# ============================================================================
ax.text(11, 2.95, "Seven model families the planner can choose from",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#666")

fam_y = 1.65
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

# Arrow from planner to families row (clean diagonal)
arrow(xs[0] + pipe_w/2, pipe_y,
      11, fam_y + fam_h,
      color="#1E6FB8", lw=1.3,
      connectionstyle="arc3,rad=-0.20", style="-|>")

# ============================================================================
# Legend (bottom)
# ============================================================================
legend_y = 0.25
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

out_path = "/tmp/agent4_architecture.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
