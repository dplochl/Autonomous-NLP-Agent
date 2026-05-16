"""Hardcoded submission tails — appended to the frozen training script at
final-submission time. Bypasses the LLM-generated FINAL_SUBMISSION /
WRITE_SUBMISSION code entirely.

How the flow looks:
  1. Sweep / opt: LLM-generated training script runs unchanged. It trains
     the model, tunes the threshold, prints METRICS, nothing more.
  2. Final submission: orchestrator takes the frozen best_train.py,
     APPENDS one of the hardcoded tails below, and runs the combined
     script with AGENT_WRITE_SUBMISSION=0 + AGENT_FINAL_SUBMISSION=0
     (so the LLM's own submission code does NOT fire).
  3. The hardcoded tail runs after the training, builds its own test
     dataset from disk, and writes the Kaggle CSV.

The tail relies only on the following variables being in scope after the
training portion of the script runs:

  All families:
    test_df         — DataFrame with at least 'id' and 'text' columns.
                      Loaded at the top of every LLM-generated script.
    best_threshold  — float, the val-tuned cutoff.

  Sparse (BoW, BoW_advanced):
    vectorizer  — fitted sklearn vectorizer with .transform(list[str])
    classifier  — fitted estimator with .predict_proba returning [N, 2]

  Transformer (RoBERTa, BERTweet):
    trainer    — HF Trainer with the trained model attribute
    tokenizer  — HF AutoTokenizer instance used for training

  Deep (CNN, LSTM, EmbeddingDL):
    model      — torch.nn.Module producing [N, 2] logits from token tensors
    Either tokenize_for_inference, encode_text, or encode — a callable
    taking list[str] and returning a LongTensor suitable as model input.

Templates enforce these names.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Sparse families (BoW, BoW_advanced) — sklearn pipeline
# ---------------------------------------------------------------------------

SPARSE_TAIL = '''
# === AGENT_4 HARDCODED FINAL SUBMISSION TAIL (sparse) ===
import os as _os_sub
import pandas as _pd_sub

# Find the fitted classifier (tolerant to LLM naming variations).
_clf = None
for _n in ("classifier", "model", "clf", "lr", "lr_model", "log_reg", "logreg", "estimator"):
    _o = globals().get(_n)
    if _o is not None and hasattr(_o, "predict_proba"):
        _clf = _o
        break
if _clf is None:
    raise RuntimeError("[AGENT_SUBMIT] could not find a fitted classifier with .predict_proba")

# Find ALL fitted vectorizers. Single-vectorizer BoW uses just one;
# BoW_advanced trains on hstack([word_vectorizer, char_vectorizer]) so the
# classifier expects the COMBINED feature count. We collect every vectorizer
# in scope, then pick the single one (or hstack subset) whose feature count
# matches the classifier's n_features_in_.
_vecs = []
for _n in ("vectorizer", "tfidf_vectorizer", "tfidf", "vec",
          "word_vectorizer", "tfidf_word",
          "char_vectorizer", "tfidf_char"):
    _o = globals().get(_n)
    if (_o is not None and hasattr(_o, "transform")
            and hasattr(_o, "vocabulary_") and _o not in [v for _, v in _vecs]):
        _vecs.append((_n, _o))
if not _vecs:
    raise RuntimeError("[AGENT_SUBMIT] could not find a fitted vectorizer with .transform")

# Build the same text field the training code used.
_text_col = None
for _candidate in ("text_for_model", "x", "text_input"):
    if _candidate in test_df.columns:
        _text_col = _candidate
        break
if _text_col is None:
    if "keyword" in test_df.columns:
        test_df["_agent_submit_text"] = test_df.apply(
            lambda r: f"{r['keyword']} [SEP] {r['text']}" if str(r.get('keyword', '')) else str(r.get('text', '')),
            axis=1,
        )
        _text_col = "_agent_submit_text"
    else:
        _text_col = "text"

_test_X = test_df[_text_col].astype(str).tolist()
_transforms = [(_n, _v.transform(_test_X)) for _n, _v in _vecs]
_target_n = getattr(_clf, "n_features_in_", None)

if _target_n is None:
    # No way to check — use the first vectorizer.
    _X_test = _transforms[0][1]
    _used = [_transforms[0][0]]
else:
    # Single-vectorizer match wins first (simpler, common case).
    _match = next(((n, t) for n, t in _transforms if t.shape[1] == _target_n), None)
    if _match is not None:
        _X_test = _match[1]
        _used = [_match[0]]
    elif sum(t.shape[1] for _, t in _transforms) == _target_n:
        # All vectorizers concatenated match — hstack them in the order found.
        from scipy.sparse import hstack as _hstack_sub
        _X_test = _hstack_sub([t for _, t in _transforms]).tocsr()
        _used = [n for n, _ in _transforms]
    else:
        raise RuntimeError(
            f"[AGENT_SUBMIT] vectorizer feature count mismatch: "
            f"classifier expects {_target_n} features, but the vectorizers in scope "
            f"produce { {n: t.shape[1] for n, t in _transforms} } "
            f"(neither a single match nor a clean hstack)."
        )
print(f"[AGENT_SUBMIT] vectorizer(s) used: {_used} -> {_X_test.shape[1]} features")
_test_probs = _clf.predict_proba(_X_test)[:, 1]
_test_preds = (_test_probs >= float(best_threshold)).astype(int)
_sub_df = _pd_sub.DataFrame({"id": test_df["id"].astype(int), "target": _test_preds})

_sub_path = _os_sub.environ.get(
    "DISASTER_AGENT_SUBMISSION_PATH",
    _os_sub.environ.get("AGENT_SUBMISSION_PATH", "submission.csv"),
)
_os_sub.makedirs(_os_sub.path.dirname(_sub_path) or ".", exist_ok=True)
_sub_df.to_csv(_sub_path, index=False)
_pos = int((_sub_df["target"] == 1).sum())
_neg = int((_sub_df["target"] == 0).sum())
print(f"[AGENT_SUBMIT] wrote {len(_sub_df)} rows to {_sub_path} (threshold={float(best_threshold):.4f}, pos={_pos}, neg={_neg})")
'''


# ---------------------------------------------------------------------------
# Transformer families (RoBERTa, BERTweet) — HuggingFace Trainer
# ---------------------------------------------------------------------------

TRANSFORMER_TAIL = '''
# === AGENT_4 HARDCODED FINAL SUBMISSION TAIL (transformer) ===
import os as _os_sub
import numpy as _np_sub
import pandas as _pd_sub
import torch as _torch_sub

# Build text the same way training did.
_text_col = None
for _candidate in ("text_for_model", "x", "text_input"):
    if _candidate in test_df.columns:
        _text_col = _candidate
        break
if _text_col is None:
    if "keyword" in test_df.columns:
        test_df["_agent_submit_text"] = test_df.apply(
            lambda r: f"{r['keyword']} [SEP] {r['text']}" if str(r.get('keyword', '')) else str(r.get('text', '')),
            axis=1,
        )
        _text_col = "_agent_submit_text"
    else:
        _text_col = "text"

_max_len = int(globals().get("max_len", 128))
_texts = test_df[_text_col].astype(str).tolist()

# Build a minimal Dataset that yields tokenizer output dicts.
class _AgentSubmitDataset(_torch_sub.utils.data.Dataset):
    def __init__(self, texts, tok, mlen):
        self.enc = tok(texts, padding="max_length", truncation=True, max_length=mlen, return_tensors="pt")
    def __len__(self):
        return self.enc["input_ids"].size(0)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}

# Find the HF Trainer + tokenizer (tolerant to LLM naming variations).
_tr = None
for _n in ("trainer", "hf_trainer", "trainer_obj"):
    _o = globals().get(_n)
    if _o is not None and hasattr(_o, "predict"):
        _tr = _o
        break
if _tr is None:
    raise RuntimeError("[AGENT_SUBMIT] could not find an HF Trainer with .predict")
_tok = None
for _n in ("tokenizer", "auto_tokenizer", "tok"):
    _o = globals().get(_n)
    if _o is not None and callable(_o):
        _tok = _o
        break
if _tok is None:
    raise RuntimeError("[AGENT_SUBMIT] could not find a tokenizer (callable)")

_ds = _AgentSubmitDataset(_texts, _tok, _max_len)
_logits = _tr.predict(_ds).predictions
_e = _np_sub.exp(_logits - _logits.max(axis=1, keepdims=True))
_probs = (_e / _e.sum(axis=1, keepdims=True))[:, 1]
_preds = (_probs >= float(best_threshold)).astype(int)
_sub_df = _pd_sub.DataFrame({"id": test_df["id"].astype(int), "target": _preds})

_sub_path = _os_sub.environ.get(
    "DISASTER_AGENT_SUBMISSION_PATH",
    _os_sub.environ.get("AGENT_SUBMISSION_PATH", "submission.csv"),
)
_os_sub.makedirs(_os_sub.path.dirname(_sub_path) or ".", exist_ok=True)
_sub_df.to_csv(_sub_path, index=False)
_pos = int((_sub_df["target"] == 1).sum())
_neg = int((_sub_df["target"] == 0).sum())
print(f"[AGENT_SUBMIT] wrote {len(_sub_df)} rows to {_sub_path} (threshold={float(best_threshold):.4f}, pos={_pos}, neg={_neg})")
'''


# ---------------------------------------------------------------------------
# Deep families (CNN, LSTM, EmbeddingDL) — torch.nn.Module
# ---------------------------------------------------------------------------

DEEP_TAIL = '''
# === AGENT_4 HARDCODED FINAL SUBMISSION TAIL (deep) ===
import os as _os_sub
import torch as _torch_sub
import numpy as _np_sub
import pandas as _pd_sub

# Find the encoding helper the script defined.
_encode = None
for _name in ("tokenize_for_inference", "encode_text", "encode_texts", "encode"):
    if _name in dir() and callable(globals().get(_name)):
        _encode = globals()[_name]
        break
if _encode is None:
    raise RuntimeError(
        "AGENT_SUBMIT: deep tail could not find a text-encoding helper. "
        "Training code must define one of: tokenize_for_inference, encode_text, encode_texts, encode."
    )

_text_col = None
for _candidate in ("text_for_model", "x", "text_input"):
    if _candidate in test_df.columns:
        _text_col = _candidate
        break
if _text_col is None:
    if "keyword" in test_df.columns:
        test_df["_agent_submit_text"] = test_df.apply(
            lambda r: f"{r['keyword']} [SEP] {r['text']}" if str(r.get('keyword', '')) else str(r.get('text', '')),
            axis=1,
        )
        _text_col = "_agent_submit_text"
    else:
        _text_col = "text"

_texts = test_df[_text_col].astype(str).tolist()
model.eval()
with _torch_sub.no_grad():
    _enc = _encode(_texts)
    _out = model(_enc)
    if hasattr(_out, "logits"):
        _out = _out.logits
    _probs = _torch_sub.softmax(_out, dim=-1)[:, 1].cpu().numpy()
_preds = (_probs >= float(best_threshold)).astype(int)
_sub_df = _pd_sub.DataFrame({"id": test_df["id"].astype(int), "target": _preds})

_sub_path = _os_sub.environ.get(
    "DISASTER_AGENT_SUBMISSION_PATH",
    _os_sub.environ.get("AGENT_SUBMISSION_PATH", "submission.csv"),
)
_os_sub.makedirs(_os_sub.path.dirname(_sub_path) or ".", exist_ok=True)
_sub_df.to_csv(_sub_path, index=False)
_pos = int((_sub_df["target"] == 1).sum())
_neg = int((_sub_df["target"] == 0).sum())
print(f"[AGENT_SUBMIT] wrote {len(_sub_df)} rows to {_sub_path} (threshold={float(best_threshold):.4f}, pos={_pos}, neg={_neg})")
'''


# Family -> category mapping
FAMILY_CATEGORY = {
    "BoW": "sparse",
    "BoW_advanced": "sparse",
    "CNN": "deep",
    "LSTM": "deep",
    "EmbeddingDL": "deep",
    "RoBERTa": "transformer",
    "BERTweet": "transformer",
}

TAILS = {
    "sparse": SPARSE_TAIL,
    "deep": DEEP_TAIL,
    "transformer": TRANSFORMER_TAIL,
}


def tail_for_family(family_label: str) -> str:
    category = FAMILY_CATEGORY.get(family_label)
    if category is None:
        raise ValueError(f"Unknown family for submission tail: {family_label!r}")
    return TAILS[category]


# Marker used to find the tail boundary so we can strip+reattach a fresh tail
# after each repair iteration. The repair LLM has been observed to silently
# rewrite our tail; strip-and-reattach guarantees the orchestrator-owned tail
# stays pristine regardless of what the LLM does.
TAIL_MARKER = "# === AGENT_4 HARDCODED FINAL SUBMISSION TAIL"


def strip_submission_tail(code: str) -> str:
    """Remove any previously-appended submission tail from `code`."""
    idx = code.find(TAIL_MARKER)
    if idx < 0:
        return code
    return code[:idx].rstrip() + "\n"


# Marker that surrounds the try/except wrap. Lets us detect and strip a
# previously-applied wrap so repeated calls to append_submission_tail are
# idempotent (the orchestrator calls it twice — once in
# prepare_final_submission_payload and once in execute_final_submission).
WRAP_BEGIN = "# === AGENT_4 TRY-WRAP BEGIN ===  (do not edit; see submit_tails.py)"
WRAP_END   = "# === AGENT_4 TRY-WRAP END ==="


def _strip_try_wrap(code: str) -> str:
    """Remove a previously-applied try/except wrap from `code`. If the wrap
    isn't present, returns the input unchanged."""
    begin = code.find(WRAP_BEGIN)
    end = code.find(WRAP_END)
    if begin < 0 or end < 0 or end <= begin:
        return code
    # The original (un-indented) LLM code lives between WRAP_BEGIN and the
    # start of the `try:` block. We reverse the wrap by extracting only the
    # indented body of the try block and dedenting it. Lines that started
    # empty / whitespace-only stay as-is.
    inner_lines: list[str] = []
    capturing = False
    for line in code[begin:end].splitlines():
        if line.lstrip().startswith("try:") and not capturing:
            capturing = True
            continue
        if not capturing:
            continue
        if line.lstrip().startswith("except Exception as _agent4_wrap_exc"):
            break
        # Strip the leading 4 spaces we added.
        inner_lines.append(line[4:] if line.startswith("    ") else line)
    before = code[:begin].rstrip()
    after = code[end + len(WRAP_END):].lstrip("\n")
    middle = "\n".join(inner_lines).rstrip()
    return (before + ("\n" if before else "") + middle + ("\n" + after if after else "\n"))


def _wrap_in_try_except(code: str) -> str:
    """Wrap `code` in a try/except so the orchestrator-owned tail that follows
    runs even when the LLM's own test-prediction or submission-writing code
    raises (e.g. shape-mismatch, wrong test split, broken pd.DataFrame call).

    Variables assigned inside the try block (trainer, tokenizer, best_threshold,
    test_df, ...) remain accessible at module scope after the except handler,
    so the hardcoded tail downstream can still build a valid submission CSV
    using whatever state was set up before the failure.

    Idempotent: re-applying the wrap to already-wrapped code re-uses the
    original (unindented) inner block.
    """
    # If we're re-wrapping previously-wrapped code, first strip the old wrap.
    code = _strip_try_wrap(code)
    indented = "\n".join("    " + line if line.strip() else line for line in code.splitlines())
    return (
        WRAP_BEGIN + "\n"
        + "try:\n"
        + indented
        + "\nexcept Exception as _agent4_wrap_exc:\n"
        + "    import traceback as _agent4_tb\n"
        + "    print('[AGENT_4 WRAP] LLM section raised:', repr(_agent4_wrap_exc))\n"
        + "    print('[AGENT_4 WRAP] traceback follows; hardcoded tail below will compensate.')\n"
        + "    _agent4_tb.print_exc()\n"
        + WRAP_END + "\n"
    )


def append_submission_tail(code: str, family_label: str) -> str:
    """Append the hardcoded submission tail. Strips any prior tail first
    (defensive — in case the repair loop introduced one), then wraps the
    rest of the script in a try/except so the tail is guaranteed to run
    even when the LLM's own submission code raises.
    """
    tail = tail_for_family(family_label)
    base = strip_submission_tail(code)
    wrapped = _wrap_in_try_except(base)
    return wrapped.rstrip() + "\n\n" + tail.strip() + "\n"
