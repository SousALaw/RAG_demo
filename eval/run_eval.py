import json
import time
import requests

API_URL = "http://127.0.0.1:8000/qa/ask"
EVAL_PATH = "eval/eval_dataset.json"

def evaluate():
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    hit_sources = 0
    hit_keywords = 0
    total_latency = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        expected_keywords = item.get("expected_keywords", [])

        start = time.time()
        try:
            resp = requests.post(API_URL, json={
                "query": question,
                "session_id": f"eval_{i}",  # 每条问题独立会话
            }, timeout=60)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"❌ [{i+1}/{total}] {question} -> 请求失败: {e}")
            continue

        latency = time.time() - start
        total_latency += latency

        answer = result.get("answer", "")
        references = result.get("references", [])

        # 召回命中：expected_sources 里至少有一个在 references 里
        source_hit = any(src in references for src in expected_sources)
        if source_hit:
            hit_sources += 1

        # 关键词命中：所有期望关键词都出现在回答里
        answer_lower = answer.lower()
        keyword_hit = all(kw.lower() in answer_lower for kw in expected_keywords) if expected_keywords else False
        if keyword_hit:
            hit_keywords += 1

        print(f"[{i+1}/{total}] {question}")
        print(f"  references: {references}")
        print(f"  召回命中: {'✅' if source_hit else '❌'}  关键词命中: {'✅' if keyword_hit else '❌'}  耗时: {latency:.1f}s")
        if not keyword_hit:
            print(f"  ⚠️ 回答全文：{answer}")
            print(f"  ⚠️ 期望关键词：{expected_keywords}")
        print()

    print("=" * 50)
    print(f"总问题数: {total}")
    print(f"召回命中率: {hit_sources}/{total} = {hit_sources/total:.1%}")
    print(f"关键词命中率: {hit_keywords}/{total} = {hit_keywords/total:.1%}")
    print(f"平均延迟: {total_latency/total:.1f}s")

if __name__ == "__main__":
    evaluate()