"""S1 - BFCL scoring stub: AST-match function calls, and score tool selection on BFCL "multiple".

Shows
  1. A simplified version of BFCL's AST checker (bfcl_eval/eval_checker/ast_eval/ast_checker.py): a
     predicted call matches if the function name matches, no unknown or missing required arguments,
     every argument's value is among the acceptable values ("" = may be omitted), with BFCL's string
     normalisation and int -> float promotion. Scored here on predictions derived from the answers
     (perfect, then deliberately perturbed) to show what the checker catches; pass
     ``--predictions preds.jsonl`` ({"id", "calls": [{"name", "arguments"}]}) to score a real model.
  2. The selection question CATS answers, on BFCL "multiple" (2-4 candidate functions per request):
     how often BM25 over name + description + parameters ranks the gold function first.

Needs
  BFCL v4 data in data/external/bfcl/raw/ (``python scripts/fetch_external.py bfcl``; raw.githubusercontent
  .com only). Without it, three inline examples in BFCL's format are used. Hugging Face (the dataset
  card gorilla-llm/Berkeley-Function-Calling-Leaderboard) and the GitHub API are not needed.
  No model and no key.

  python samples/s01_bfcl_scoring.py [--n 200] [--predictions preds.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from cats.index import BM25

RAW = Path(__file__).resolve().parents[1] / "data" / "external" / "bfcl" / "raw"
INLINE_Q = [
    {"id": "demo_0", "question": [[{"role": "user", "content": "Area of a triangle with base 10 and height 5?"}]],
     "function": [{"name": "calculate_triangle_area", "description": "Area of a triangle.", "parameters": {
         "type": "dict", "properties": {"base": {"type": "integer"}, "height": {"type": "integer"},
                                        "unit": {"type": "string"}}, "required": ["base", "height"]}}]},
    {"id": "demo_1", "question": [[{"role": "user", "content": "Weather in Boston in fahrenheit"}]],
     "function": [{"name": "get_weather", "description": "Current weather.", "parameters": {
         "type": "dict", "properties": {"city": {"type": "string"}, "unit": {"type": "string",
                                                                             "enum": ["c", "f"]}},
         "required": ["city"]}}]},
    {"id": "demo_2", "question": [[{"role": "user", "content": "5 factorial"}]],
     "function": [{"name": "math.factorial", "description": "Factorial.", "parameters": {
         "type": "dict", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}]},
]
INLINE_A = [
    {"id": "demo_0", "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]},
    {"id": "demo_1", "ground_truth": [{"get_weather": {"city": ["Boston", "Boston, MA"], "unit": ["f"]}}]},
    {"id": "demo_2", "ground_truth": [{"math.factorial": {"number": [5]}}]},
]
PY_TYPES = {"integer": (int,), "float": (float, int), "number": (float, int), "string": (str,), "boolean": (bool,),
            "array": (list, tuple), "tuple": (list, tuple), "dict": (dict,), "object": (dict,), "any": (object,)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(v):
    """BFCL's standardize_string for strings (case, spaces and common punctuation ignored); recursive."""
    if isinstance(v, str):
        return re.sub(r"[ ,./\-_*^]", "", v).lower().replace("'", '"')
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def ast_match(func: dict, call: dict, truth: dict) -> tuple[bool, str]:
    """One predicted call vs. one ground-truth entry {name: {param: [acceptable values]}}."""
    (gname, gparams), = truth.items()
    if call.get("name", "").replace(".", "_") != gname.replace(".", "_"):
        return False, "wrong_function"
    props = func["parameters"].get("properties", {})
    args = call.get("arguments", {})
    for p in func["parameters"].get("required", []):
        if p not in args:
            return False, "missing_required"
    for p, v in args.items():
        if p not in props:
            return False, "unexpected_param"
        want = PY_TYPES.get(props[p].get("type", "any"), (object,))
        if isinstance(v, bool) and bool not in want:
            return False, "type_error"
        if not isinstance(v, want):
            return False, "type_error"
        ok = gparams.get(p, [])
        if ok and norm(v) not in [norm(x) for x in ok] and not (isinstance(v, int) and float(v) in ok):
            return False, "wrong_value"
    for p, ok in gparams.items():
        if p not in args and "" not in ok:
            return False, "missing_param"
    return True, "ok"


def perfect(truth: dict) -> dict:
    (name, params), = truth.items()
    return {"name": name, "arguments": {p: next(v for v in ok if v != "") for p, ok in params.items()
                                        if any(v != "" for v in ok)}}


def perturb(call: dict, i: int, func: dict) -> dict:
    """Every 3rd prediction drops a required argument, every 5th changes a value."""
    args = dict(call["arguments"])
    req = func["parameters"].get("required", [])
    if i % 3 == 0 and req and req[0] in args:
        args.pop(req[0])
    elif i % 5 == 0 and args:
        k = next(iter(args))
        args[k] = (args[k] + 1) if isinstance(args[k], (int, float)) and not isinstance(args[k], bool) else "xyz"
    return {"name": call["name"], "arguments": args}


def score(questions, answers, predictions) -> dict:
    by_id = {q["id"]: q for q in questions}
    errs, ok = Counter(), 0
    for a in answers:
        q = by_id[a["id"]]
        calls = predictions.get(a["id"], [])
        if len(calls) != 1:
            errs["wrong_count"] += 1
            continue
        func = next((f for f in q["function"] if f["name"].replace(".", "_") == calls[0]["name"].replace(".", "_")),
                    q["function"][0])
        good, why = ast_match(func, calls[0], a["ground_truth"][0])
        ok += good
        if not good:
            errs[why] += 1
    return {"n": len(answers), "accuracy": ok / max(1, len(answers)), "errors": dict(errs)}


def selection_top1(questions, answers) -> dict:
    gold = {a["id"]: next(iter(a["ground_truth"][0])) for a in answers}
    hit1 = hit2 = n = 0
    n_funcs = []
    for q in questions:
        fs = q["function"]
        if q["id"] not in gold or len(fs) < 2:
            continue
        bm = BM25()
        for i, f in enumerate(fs):
            props = f["parameters"].get("properties", {})
            bm.add(i, " ".join([f["name"].replace(".", " ").replace("_", " "), f.get("description", "")]
                               + [f"{k} {v.get('description', '')}" for k, v in props.items()]))
        text = " ".join(m["content"] for turn in q["question"] for m in turn if m["role"] == "user")
        s = bm.scores(text, len(fs))
        order = sorted(range(len(fs)), key=lambda i: -s[i])
        names = [fs[i]["name"] for i in order]
        n += 1
        hit1 += names[0] == gold[q["id"]]
        hit2 += gold[q["id"]] in names[:2]
        n_funcs.append(len(fs))
    return {"n": n, "mean_candidates": sum(n_funcs) / max(1, n), "bm25_top1": hit1 / max(1, n),
            "bm25_top2": hit2 / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200, help="questions of simple_python to score")
    ap.add_argument("--predictions", help="JSONL of model predictions to score instead of the demo ones")
    a = ap.parse_args()
    if (RAW / "BFCL_v4_simple_python.json").exists():
        qs = load_jsonl(RAW / "BFCL_v4_simple_python.json")[: a.n]
        ans = load_jsonl(RAW / "possible_answer" / "BFCL_v4_simple_python.json")[: a.n]
        print(f"BFCL v4 simple_python from {RAW} ({len(qs)} questions)")
    else:
        qs, ans = INLINE_Q, INLINE_A
        print("BFCL data not found; using 3 inline examples in BFCL format (run scripts/fetch_external.py bfcl)")
    funcs = {q["id"]: q["function"][0] for q in qs}
    if a.predictions:
        preds = {r["id"]: r["calls"] for r in load_jsonl(Path(a.predictions))}
        print("model predictions:", json.dumps(score(qs, ans, preds)))
    else:
        perfect_preds = {x["id"]: [perfect(x["ground_truth"][0])] for x in ans}
        print("perfect predictions:  ", json.dumps(score(qs, ans, perfect_preds)))
        noisy = {i: [perturb(c[0], k, funcs[i])] for k, (i, c) in enumerate(perfect_preds.items())}
        print("perturbed predictions:", json.dumps(score(qs, ans, noisy)))
    if (RAW / "BFCL_v4_multiple.json").exists():
        mq = load_jsonl(RAW / "BFCL_v4_multiple.json")
        ma = load_jsonl(RAW / "possible_answer" / "BFCL_v4_multiple.json")
        print("selection on BFCL multiple:", json.dumps(selection_top1(mq, ma)))


if __name__ == "__main__":
    main()
