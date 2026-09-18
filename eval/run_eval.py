"""检索效果评估：对比 纯向量 / +混合检索 / +混合检索+重排序 三种模式。

【候选预算对齐】三种模式统一按 top-5 计算指标。
渐进式检索在 stage1 常常只返回 1 篇，列表长度 1 和 5 之间的 MRR@5 / Recall@1
不可比，所以 baseline 在这里不用渐进式，而是取纯向量的 top-5（用户确认的方案）。
这样三行数字才是同一个口径。

【成本】默认只跑检索，不调大模型、不调重排之外的外部服务，因此基本免费。
要算「关键词命中率」和端到端延迟必须显式加 --with-answers，那会按题数调用大模型。
带重排的模式每次会调一次重排接口（输入篇数受 config.rerank_max_candidates 限制）。

用法：
    python eval/run_eval.py                       # 三种模式，检索指标（默认）
    python eval/run_eval.py --dataset hard        # 用困难集
    python eval/run_eval.py --mode baseline       # 只跑一种
    python eval/run_eval.py --baseline            # 同上（别名）
    python eval/run_eval.py --no-rerank           # 只跑 +hybrid
    python eval/run_eval.py --with-answers        # 额外算关键词命中率（会调大模型）
    python eval/run_eval.py --api                 # 走 HTTP（模式以服务端为准）
    python eval/run_eval.py --limit 5 --json out.json
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = {
    "base": os.path.join(PROJECT_ROOT, "eval", "eval_dataset.json"),
    "hard": os.path.join(PROJECT_ROOT, "eval", "eval_dataset_hard.json"),
}
API_PATH = "/qa/ask"
RECALL_KS = (1, 3, 5)
MRR_K = 5

# 所有模式统一按这个候选数计算指标，保证可比
CANDIDATE_BUDGET = 5

# (显示名, hybrid, rerank, session 前缀)
ALL_MODES = [
    ("baseline", False, False, "base"),
    ("+hybrid", True, False, "hyb"),
    ("+hybrid+rerank", True, True, "hybrr"),
]


def load_dataset(path: str, limit=None) -> list:
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset[:limit] if limit else dataset


# ---------------- 指标 ----------------

def first_hit_rank(references: list, expected_sources: list, k: int = MRR_K) -> int:
    for index, ref in enumerate(references[:k]):
        if ref in expected_sources:
            return index + 1
    return 0


def recall_at(references: list, expected_sources: list, k: int) -> int:
    return 1 if any(ref in expected_sources for ref in references[:k]) else 0


def keyword_hit(answer: str, expected_keywords: list) -> int:
    if not expected_keywords:
        return 0
    lowered = (answer or "").lower()
    return 1 if all(kw.lower() in lowered for kw in expected_keywords) else 0


def dedup_sources(docs: list, limit: int = CANDIDATE_BUDGET) -> list:
    out = []
    for doc in docs:
        source = (doc.metadata or {}).get("source", "")
        if source and source not in out:
            out.append(source)
        if len(out) >= limit:
            break
    return out


# ---------------- 检索（统一预算，不调大模型） ----------------

def retrieve_sources(app, query: str, mode: str, category: str | None = None) -> list:
    """返回该模式下按相关度排序的来源列表，统一取 top-CANDIDATE_BUDGET。"""
    import config_data as config

    svc = app.rag.vector_store_service
    if mode == "baseline":
        # 纯向量 top-5。渐进式在 stage1 可能只给 1 篇，长度不同会让指标不可比。
        pairs = svc.similarity_search_with_relevance_scores(
            query, k=CANDIDATE_BUDGET, category=category
        )
        return dedup_sources([doc for doc, _ in pairs])
    if mode == "+hybrid":
        return dedup_sources(svc.hybrid_search(query, k=CANDIDATE_BUDGET, category=category))
    # +hybrid+rerank
    coarse = svc.hybrid_search(
        query, k=max(int(config.bm25_top_k), int(config.vector_top_k)), category=category
    )
    return dedup_sources(svc.rerank(query, coarse, int(config.rerank_top_n)))


def evaluate_retrieval(app, dataset, mode: str, category, verbose: bool, label: str) -> list:
    rows = []
    for i, item in enumerate(dataset):
        start = time.time()
        try:
            refs = retrieve_sources(app, item["question"], mode, category)
        except Exception as exc:
            print(f"  [{i + 1}/{len(dataset)}] 检索失败：{exc}")
            continue
        latency = time.time() - start

        expected = item.get("expected_sources", [])
        rank = first_hit_rank(refs, expected)
        row = {
            "question": item["question"],
            "references": refs,
            "rank": rank,
            "source_hit": 1 if rank else 0,
            "recall": {k: recall_at(refs, expected, k) for k in RECALL_KS},
            "retrieval_latency": latency,
        }
        rows.append(row)
        print(f"  [{label} {i + 1}/{len(dataset)}] {'命中' if rank else '未中'} "
              f"rank={rank or '-'} {latency:.2f}s  {item['question'][:26]}")
        if verbose and not rank:
            print(f"      期望={expected}")
            print(f"      实际={refs}")
    return rows


def evaluate_answers(app, dataset, mode: str, hybrid: bool, rerank: bool,
                     slug: str, verbose: bool):
    """走生产链路（app.ask）算关键词命中率与端到端延迟。会调用大模型。"""
    import config_data as config

    prev = (config.hybrid_search_enabled, config.rerank_enabled)
    config.hybrid_search_enabled, config.rerank_enabled = hybrid, rerank
    try:
        session_ids = [f"eval_{slug}_{i}" for i in range(len(dataset))]
        try:
            from file_history_store import get_history
            for sid in session_ids:
                get_history(sid).clear()
        except Exception as exc:
            print(f"  [warn] 清理会话历史失败：{exc}")

        out = {}
        for i, item in enumerate(dataset):
            start = time.time()
            try:
                result = app.ask(query=item["question"], session_id=session_ids[i])
            except Exception as exc:
                print(f"  [answer {i + 1}/{len(dataset)}] 失败：{exc}")
                continue
            latency = time.time() - start
            hit = keyword_hit(result.get("answer", ""), item.get("expected_keywords", []))
            out[item["question"]] = {"keyword_hit": hit, "answer_latency": latency}
            print(f"  [answer {i + 1}/{len(dataset)}] 关键词={'Y' if hit else 'N'} {latency:.1f}s")
            if verbose and not hit:
                print(f"      期望关键词={item.get('expected_keywords')}")
                print(f"      回答={(result.get('answer') or '')[:200]}")
        return out
    finally:
        config.hybrid_search_enabled, config.rerank_enabled = prev


# ---------------- 汇总与输出 ----------------

def summarize(rows: list, answers: dict) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    summary = {
        "n": n,
        "source_hit": sum(r["source_hit"] for r in rows) / n,
        "mrr": sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / n,
        "recall": {k: sum(r["recall"][k] for r in rows) / n for k in RECALL_KS},
        "retrieval_latency": sum(r["retrieval_latency"] for r in rows) / n,
    }
    if answers:
        matched = [answers[r["question"]] for r in rows if r["question"] in answers]
        if matched:
            summary["keyword_hit"] = sum(m["keyword_hit"] for m in matched) / len(matched)
            summary["answer_latency"] = sum(m["answer_latency"] for m in matched) / len(matched)
    return summary


def print_table(results: list, with_answers: bool) -> None:
    header = ["模式", "召回命中率", "MRR@5", "Recall@1", "Recall@3", "Recall@5", "平均检索延迟"]
    if with_answers:
        header += ["关键词命中率", "平均端到端延迟"]

    lines = []
    for name, s in results:
        if not s:
            lines.append([name] + ["-"] * (len(header) - 1))
            continue
        row = [
            name,
            f"{s['source_hit']:.1%}",
            f"{s['mrr']:.3f}",
            f"{s['recall'][1]:.1%}",
            f"{s['recall'][3]:.1%}",
            f"{s['recall'][5]:.1%}",
            f"{s['retrieval_latency'] * 1000:.0f}ms",
        ]
        if with_answers:
            row += [
                f"{s.get('keyword_hit', 0):.1%}" if "keyword_hit" in s else "-",
                f"{s.get('answer_latency', 0):.1f}s" if "answer_latency" in s else "-",
            ]
        lines.append(row)

    widths = [max([len(header[i])] + [len(r[i]) for r in lines]) for i in range(len(header))]
    print()
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("-+-".join("-" * w for w in widths))
    for row in lines:
        print(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description="检索效果评估（三种模式，统一 top-5 预算）")
    parser.add_argument("--dataset", choices=list(DATASETS), default="base",
                        help="base=原 20 题，hard=困难集")
    parser.add_argument("--baseline", action="store_true", help="只跑纯向量")
    parser.add_argument("--no-hybrid", action="store_true", help="关闭混合检索（等价于 --baseline）")
    parser.add_argument("--no-rerank", action="store_true", help="关闭重排序（只跑 +hybrid）")
    parser.add_argument("--mode", choices=["baseline", "hybrid", "hybrid+rerank"], help="只跑指定模式")
    parser.add_argument("--category", default="market", help="限定数据源，默认 market")
    parser.add_argument("--with-answers", action="store_true",
                        help="额外调大模型算关键词命中率与端到端延迟（会产生 token 消耗）")
    parser.add_argument("--api", action="store_true", help="走 HTTP（仅支持 --with-answers 路径）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--json", dest="json_out", default=None, help="逐题明细写到 JSON")
    parser.add_argument("--verbose", action="store_true", help="打印未命中题目的细节")
    args = parser.parse_args()

    path = DATASETS[args.dataset]
    if not os.path.exists(path):
        print(f"找不到评估集：{path}")
        return 1
    dataset = load_dataset(path, args.limit)
    print(f"评估集：{args.dataset} / {len(dataset)} 题（{path}）")
    print(f"候选预算：统一 top-{CANDIDATE_BUDGET}")

    if args.mode:
        picked = {"baseline": "baseline", "hybrid": "+hybrid", "hybrid+rerank": "+hybrid+rerank"}[args.mode]
        modes = [m for m in ALL_MODES if m[0] == picked]
    elif args.baseline or args.no_hybrid:
        modes = [m for m in ALL_MODES if m[0] == "baseline"]
    elif args.no_rerank:
        modes = [m for m in ALL_MODES if m[0] == "+hybrid"]
    else:
        modes = list(ALL_MODES)

    print("模式：" + "、".join(m[0] for m in modes))
    if args.with_answers:
        print(f"⚠️ --with-answers 会对每题调用大模型（{len(modes)} 模式 x {len(dataset)} 题）")
    else:
        print("只跑检索指标，不调用大模型")

    app = None
    if not args.api:
        from app_core import RAGApp
        app = RAGApp()
        # 预热 BM25，避免把冷启动算进第一条的延迟
        try:
            app.rag.vector_store_service.build_bm25_index(args.category or None)
        except Exception as exc:
            print(f"  [warn] BM25 预热失败：{exc}")

    results = []
    detail = {}
    for name, hybrid, rerank, slug in modes:
        print(f"\n===== {name} =====")
        rows = evaluate_retrieval(app, dataset, name, args.category or None, args.verbose, name)
        answers = {}
        if args.with_answers:
            answers = evaluate_answers(app, dataset, name, hybrid, rerank, slug, args.verbose)
        results.append((name, summarize(rows, answers)))
        detail[name] = rows

    print_table(results, args.with_answers)
    print("\n列说明：召回命中率=前 5 个来源里含任一期望来源；MRR@5=首个命中来源排名的倒数均值；")
    print(f"Recall@K=前 K 个来源里有任一期望来源。三种模式候选预算统一为 {CANDIDATE_BUDGET} 篇。")
    print("说明：baseline 这里取「纯向量 top-5」而不是渐进式链路——渐进式 stage1 常只返回 1 篇，")
    print("      列表长度不同会让 MRR@5 / Recall@1 失去可比性。")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"summary": {n: s for n, s in results}, "detail": detail},
                      f, ensure_ascii=False, indent=2)
        print(f"\n逐题明细已写入：{args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
