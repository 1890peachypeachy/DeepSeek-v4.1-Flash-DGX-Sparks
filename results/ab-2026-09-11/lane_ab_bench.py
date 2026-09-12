#!/usr/bin/env python3
"""Apples-to-apples lane bench: clean decode + agent-sim over OpenAI-compatible SSE.

Runs the identical protocol against any /v1/chat/completions endpoint so
SGLang (DSV4.1 TP3) and vLLM (Qwen3.8 Flash single) lanes are measured with
one yardstick on our fabric (methodology fix from the EXL3/NVFP4 round:
never compare lanes across different bench harnesses).

Modes:
  clean  -- thinking OFF, temperature 0. Prose + code categories.
            C in --clean-c. Per-stream tok/s after first token, TTFT.
  agent  -- thinking ON (production config), get_weather tool schema,
            tool-call loop replayed locally, multi-turn history.
            C in --agent-c. End-to-end per-stream tok/s, TTFT, turns.

Warmup: 2 untimed requests before anything is measured.
Output: JSONL records appended to --out + summary lines on stdout.
"""
import argparse, json, statistics, threading, time, urllib.request

FILLER = ("Entry {i:06d}: the quarterly logistics audit recorded a routine "
          "variance in the northbound depot inventory.\n")

def build_ctx(tokens):
    return "".join(FILLER.format(i=i) for i in range(max(1, int(tokens / 25))))

PROSE_REQ = ("Write a flowing, continuous essay about the history of maritime "
             "navigation, from Polynesian wayfinding to the clipper era. Use "
             "ordinary narrative prose, no lists, no headings.")
CODE_REQ = ("Write a complete Python module implementing an LRU cache with a "
            "doubly linked list and dictionary. Include get, put, delete, "
            "iteration, resize, clear, invariant validation and detailed "
            "docstrings. Then ten usage examples. Return code only.")
TOOL_REQ = ("What is the weather right now in Tokyo, and then again in "
            "Reykjavik, then in Cairo? Use the get_weather tool for each city, "
            "one call at a time, then summarize in one sentence per city.")
TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}}, "required": ["city"]}}}]
TOOL_RESULTS = {"tokyo": "22C light rain", "reykjavik": "4C wind",
                "cairo": "31C clear"}

def stream_request(base, model, messages, max_tokens, thinking, tools=None):
    payload = {"model": model, "messages": messages, "stream": True,
               "temperature": 0, "max_tokens": max_tokens,
               "stream_options": {"include_usage": True}}
    if not thinking:
        # Verified 2026-09-11 on both engines: this is the accepted knob that
        # zeroes reasoning (reasoning field comes back null). "thinking":False
        # is silently ignored by vLLM's ChatTemplateParams.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    start = time.monotonic()
    events, first_tok, tool_calls, usage = [], None, None, None
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500] if e.fp else ""
        print(f"HTTP_{e.code} on {base}: {body}", flush=True)
        raise
    with resp_ctx as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            n = sum(len(delta.get(f) or "") for f in
                    ("content", "reasoning_content", "reasoning"))
            tc = delta.get("tool_calls")
            if tc:
                # Streaming tool_calls arrive as per-index argument fragments;
                # concatenate, do not replace (a replaced value is an
                # unterminated JSON fragment -> server 400 on replay).
                if tool_calls is None:
                    tool_calls = []
                for frag in tc:
                    i = frag.get("index", 0)
                    while len(tool_calls) <= i:
                        tool_calls.append({"id": None, "type": "function",
                                           "function": {"name": None, "arguments": ""}})
                    call = tool_calls[i]
                    if frag.get("id"):
                        call["id"] = frag["id"]
                    fn = frag.get("function") or {}
                    if fn.get("name"):
                        call["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        call["function"]["arguments"] += fn["arguments"]
            if n:
                now = time.monotonic() - start
                if first_tok is None:
                    first_tok = now
                events.append({"t": now, "n": n})
    return {"events": events, "ttft": first_tok, "usage": usage,
            "tool_calls": tool_calls, "elapsed": time.monotonic() - start}

def run_agent_loop(base, model, msgs, max_tokens, thinking):
    """Run the request; replay tool calls locally up to 6 turns."""
    rec = stream_request(base, model, msgs, max_tokens, thinking, TOOLS)
    turns = 1
    msgs = list(msgs)
    while rec["tool_calls"] and turns < 6:
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [
                         {"id": f"call{turns}_{i}", "type": "function",
                          "function": {
                              "name": (c.get("function") or {}).get("name", "get_weather"),
                              "arguments": (c.get("function") or {}).get("arguments", "{}")}}
                         for i, c in enumerate(rec["tool_calls"])]})
        for i, c in enumerate(rec["tool_calls"]):
            try:
                args = json.loads((c.get("function") or {}).get("arguments") or "{}")
            except ValueError:
                args = {}
            city = str(args.get("city", "tokyo")).lower().strip()
            msgs.append({"role": "tool", "tool_call_id": f"call{turns}_{i}",
                         "content": TOOL_RESULTS.get(city, "unknown")})
        turns += 1
        rec = stream_request(base, model, msgs, max_tokens, thinking, TOOLS)
    rec["turns"] = turns
    return rec

def server_tokens(rec):
    u = rec.get("usage") or {}
    return u.get("completion_tokens") or u.get("completion_tokens_details", {}).get("total")

def window_rate(rec):
    """Decode tok/s after first token; prefer server token counts."""
    if not rec:
        return None
    ev = rec.get("events") or []
    if not ev or rec.get("ttft") is None or len(ev) < 2:
        return None
    span = ev[-1]["t"] - ev[0]["t"]
    if span <= 0:
        return None
    toks = server_tokens(rec)
    if not toks:
        toks = sum(e["n"] for e in ev) / 4.0
    return toks / span

def wave(base, model, conc, prompt, thinking, agent, max_tokens, context):
    results = [None] * conc
    def go(i):
        msgs = [{"role": "user", "content": build_ctx(context) + prompt}]
        results[i] = (run_agent_loop(base, model, msgs, max_tokens, thinking)
                      if agent else
                      stream_request(base, model, msgs, max_tokens, thinking))
    ts = [threading.Thread(target=go, args=(i,)) for i in range(conc)]
    t0 = time.monotonic()
    [t.start() for t in ts]
    [t.join() for t in ts]
    return results, time.monotonic() - t0

def emit(out, rec):
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(json.dumps(rec), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["clean", "agent"], required=True)
    ap.add_argument("--clean-c", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--agent-c", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # warmup (untimed, absorbs JIT/graph/cold-cache effects)
    for _ in range(2):
        stream_request(args.base, args.model,
                       [{"role": "user", "content": "Count to 20."}],
                       64, False)

    out = open(args.out, "a")
    levels = ([(c, "prose") for c in args.clean_c] +
              [(c, "code") for c in args.clean_c]) if args.mode == "clean" \
        else [(c, "agent") for c in args.agent_c]

    for conc, kind in levels:
        agent = kind == "agent"
        thinking = agent
        prompt = {"prose": PROSE_REQ, "code": CODE_REQ, "agent": TOOL_REQ}[kind]
        for rep in range(args.repeats):
            res, wall = wave(args.base, args.model, conc, prompt, thinking,
                             agent, args.max_tokens, args.context)
            rates = [r for r in (window_rate(r) for r in res) if r]
            ttfts = [r["ttft"] for r in res if r["ttft"] is not None]
            toks = [t for t in (server_tokens(r) for r in res) if t]
            agg = round(sum(toks) / wall, 2) if toks and wall > 0 else None
            emit(out, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "lane": args.base,
                "model": args.model, "mode": args.mode, "category": kind,
                "concurrency": conc, "repeat": rep, "context": args.context,
                "max_tokens": args.max_tokens,
                "per_stream_tok_s": [round(r, 2) for r in rates],
                "median_tok_s": round(statistics.median(rates), 2) if rates else None,
                "ttft_s": [round(t, 3) for t in ttfts],
                "aggregate_tok_s": agg, "wall_s": round(wall, 2),
                "turns": [r.get("turns", 1) for r in res]})
            time.sleep(3)
    out.close()
    print("BENCH_DONE", args.mode, args.base, flush=True)

if __name__ == "__main__":
    main()
