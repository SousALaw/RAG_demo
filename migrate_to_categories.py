"""
一次性迁移：把改造前的单一知识库（data/*.txt + 无 category 的向量）迁移到 category="market"。

幂等，可以重复运行。**执行前请先停掉 uvicorn 与 streamlit**：Chroma 的 sqlite 被占用时
备份与就地更新都不安全。

步骤：
1. 复制备份 chroma_db/ -> chroma_db.bak/（已存在则跳过）
2. data/ 顶层的 *.txt 移动到 data/market/
3. 就地为 Chroma 里已有的向量补 category / content_md5（只改 metadata，不重新嵌入）
4. 用「已入库的文件集合」重写 md5.text 为 "content_md5<TAB>market"
   （会保留 md5.text 里其他分类的行，避免重跑时误删别的源）
5. 对 data/market 下仍未入库的 *.txt 调 add_new_file(category="market")
   —— 已入库的会因为 md5 命中返回 [跳过]，只有缺口那部分才真正嵌入

用法：
    python migrate_to_categories.py --dry-run        # 只报告，不改任何数据
    python migrate_to_categories.py                  # 真正执行
    python migrate_to_categories.py --skip-backfill  # 只做 1~4（不补嵌入）
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

import config_data as config
from knowledge_base import KnowledgeBaseService, get_string_md5

# 与 app_core 保持一致：构造 DashScopeEmbeddings 时就需要 DASHSCOPE_API_KEY。
load_dotenv()

CATEGORY = "market"
BATCH = 500
PROGRESS_EVERY = 100


def report(msg: str) -> None:
    print(msg, flush=True)


def backup_chroma(dry_run: bool) -> None:
    """复制备份 chroma_db（是复制不是移动：第 3 步要基于原库就地更新）。"""
    src = config.persist_directory
    dst = src.rstrip("/\\") + ".bak"

    if os.path.exists(dst):
        report(f"[跳过] 备份已存在：{dst}")
        return
    if not os.path.exists(src):
        report(f"[跳过] {src} 不存在，无需备份")
        return
    if dry_run:
        report(f"[dry-run] 会复制备份 {src} -> {dst}")
        return

    shutil.copytree(src, dst)
    report(f"[完成] 备份 {src} -> {dst}")


def move_top_level_txt(data_dir: str, category: str, dry_run: bool) -> None:
    """把 data/ 顶层的 .txt 移进 data/{category}/。"""
    target_dir = os.path.join(data_dir, category)
    entries = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    txt_files = [f for f in entries if f.lower().endswith(".txt")]
    leftovers = [f for f in entries if not f.lower().endswith(".txt")]

    report(f"data/ 顶层：{len(entries)} 个文件（.txt {len(txt_files)}，非 .txt {len(leftovers)}）")
    if leftovers:
        report(f"  注意：这些非 txt 文件不会被移动 -> {leftovers[:10]}")

    if not txt_files:
        report("[跳过] 顶层没有待移动的 .txt")
        return
    if dry_run:
        report(f"[dry-run] 会把 {len(txt_files)} 个 .txt 移动到 {target_dir}")
        return

    os.makedirs(target_dir, exist_ok=True)
    moved = 0
    for filename in txt_files:
        src = os.path.join(data_dir, filename)
        dst = os.path.join(target_dir, filename)
        if os.path.exists(dst):
            report(f"  [跳过] {filename} 在 {category}/ 已存在，不覆盖")
            continue
        shutil.move(src, dst)
        moved += 1
    report(f"[完成] 移动 {moved} 个文件到 {target_dir}")


def _content_md5_of(data_dir: str, category: str, source: str, cache: dict) -> str:
    """读取文件内容算 md5；文件不存在返回空串。

    dry-run 时文件还没移动到 data/{category}/，所以额外回退到 data/ 顶层看一眼，
    这样 dry-run 的预测才和真实执行一致。
    """
    if source in cache:
        return cache[source]

    for path in (os.path.join(data_dir, category, source), os.path.join(data_dir, source)):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                cache[source] = get_string_md5(f.read())
            return cache[source]

    cache[source] = ""
    return ""


def _candidate_files(data_dir: str, category: str, dry_run: bool) -> list:
    """迁移后应归属该分类的 .txt 文件名；dry-run 时把顶层尚未移动的也算进来。"""
    names = set()
    target_dir = os.path.join(data_dir, category)
    if os.path.isdir(target_dir):
        names.update(f for f in os.listdir(target_dir) if f.lower().endswith(".txt"))

    if dry_run:
        names.update(
            f for f in os.listdir(data_dir)
            if f.lower().endswith(".txt") and os.path.isfile(os.path.join(data_dir, f))
        )
    return sorted(names)


def tag_existing_vectors(kb: KnowledgeBaseService, data_dir: str, category: str, dry_run: bool) -> tuple:
    """就地为已有向量补 category / content_md5。

    返回 (库里的 source 集合, 库里的 content_md5 集合)。
    注意：Chroma 的 update 是**整体替换** metadata，所以要带上原有的
    create_time / operator / source，否则这些字段会丢。
    """
    coll = kb.chroma._collection
    got = coll.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    report(f"Chroma 现有向量：{len(ids)} 条")

    if not ids:
        return set(), set()

    cache: dict = {}
    sources = set()
    new_metas = []
    for meta in metas:
        meta = dict(meta or {})
        source = meta.get("source", "")
        sources.add(source)
        meta["category"] = category
        meta["content_md5"] = _content_md5_of(data_dir, category, source, cache)
        new_metas.append(meta)

    known_md5 = {m["content_md5"] for m in new_metas if m.get("content_md5")}
    missing = sum(1 for m in new_metas if not m.get("content_md5"))
    if missing:
        report(f"  其中 {missing} 条向量找不到对应文件（content_md5 留空）")

    if dry_run:
        report(f"[dry-run] 会给 {len(ids)} 条向量打上 category={category}")
        return sources, known_md5

    for start in range(0, len(ids), BATCH):
        coll.update(ids=ids[start:start + BATCH], metadatas=new_metas[start:start + BATCH])
        report(f"  已更新 {min(start + BATCH, len(ids))}/{len(ids)}")

    report(f"[完成] 就地更新 {len(ids)} 条向量的 metadata")
    return sources, known_md5


def _keep_other_category_lines() -> list:
    """保留 md5.text 里属于其他分类的行（重跑时不能误删别的源）。"""
    keep = []
    if not os.path.exists(config.md5_path):
        return keep

    with open(config.md5_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            # 旧格式（没有制表符的裸 hash）直接丢弃，其余分类的原样保留
            if len(parts) == 2 and parts[1] in config.knowledge_categories and parts[1] != CATEGORY:
                keep.append(line)
    return keep


def rebuild_md5(data_dir: str, category: str, sources: set, dry_run: bool) -> int:
    """用「已入库文件集合」重写 md5.text 的 market 部分。"""
    cache: dict = {}
    market_lines = []
    for source in sorted(sources):
        if not source:
            continue
        md5_value = _content_md5_of(data_dir, category, source, cache)
        if not md5_value:
            continue
        market_lines.append(f"{md5_value}\t{category}")

    other_lines = _keep_other_category_lines()
    report(f"md5.text 将重写：{category} {len(market_lines)} 行，保留其他分类 {len(other_lines)} 行")

    if dry_run:
        report("[dry-run] 不会写 md5.text")
        return len(market_lines)

    with open(config.md5_path, "w", encoding="utf-8") as f:
        for line in other_lines + market_lines:
            f.write(line + "\n")
    report(f"[完成] md5.text 已重写：{config.md5_path}")
    return len(market_lines)


def backfill(kb: KnowledgeBaseService, data_dir: str, category: str, dry_run: bool, known_md5: set) -> None:
    """把该分类下仍未入库的 .txt 补进去（已入库的会被 md5 跳过）。"""
    names = _candidate_files(data_dir, category, dry_run)
    report(f"待处理 .txt：{len(names)} 个（含 dry-run 时尚未移动的顶层文件）")

    if not names:
        return

    if dry_run:
        cache: dict = {}
        need = 0
        for name in names:
            md5_value = _content_md5_of(data_dir, category, name, cache)
            if md5_value and md5_value not in known_md5:
                need += 1
        report(
            f"[dry-run] 预计需要嵌入 {need} 个，"
            f"{len(names) - need} 个已入库会被跳过"
        )
        return

    embedded = skipped = failed = 0
    failures = []
    for index, filename in enumerate(names, 1):
        path = os.path.join(data_dir, category, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            failed += 1
            failures.append((filename, f"读取失败：{exc}"))
            continue

        try:
            result = kb.add_new_file(filename=filename, data=content, category=category)
        except Exception as exc:
            failed += 1
            failures.append((filename, str(exc)))
            continue

        if "[跳过]" in result:
            skipped += 1
        elif "[成功]" in result:
            embedded += 1
        else:
            failed += 1
            failures.append((filename, result))

        if index % PROGRESS_EVERY == 0:
            report(f"  进度 {index}/{len(names)}：新嵌入 {embedded}，跳过 {skipped}，失败 {failed}")

    report(f"[完成] 新嵌入 {embedded}，跳过（已入库）{skipped}，失败 {failed}")
    if failures:
        report("失败清单（直接重跑本脚本即可自动重试这些文件）：")
        for name, err in failures[:20]:
            report(f"  - {name}: {err}")
        if len(failures) > 20:
            report(f"  ... 其余 {len(failures) - 20} 条省略")


def main() -> int:
    parser = argparse.ArgumentParser(description="把旧知识库迁移到 category=market")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不改任何数据")
    parser.add_argument("--skip-backfill", action="store_true", help="不执行第 5 步（不补嵌入）")
    args = parser.parse_args()

    if CATEGORY not in config.knowledge_categories:
        report(f"错误：config.knowledge_categories 里没有 {CATEGORY}")
        return 1

    report(f"=== 迁移到 category='{CATEGORY}'{'（dry-run）' if args.dry_run else ''} ===")

    backup_chroma(args.dry_run)
    move_top_level_txt(config.data_directory, CATEGORY, args.dry_run)

    kb = KnowledgeBaseService()
    sources, known_md5 = tag_existing_vectors(kb, config.data_directory, CATEGORY, args.dry_run)
    rebuild_md5(config.data_directory, CATEGORY, sources, args.dry_run)

    if args.skip_backfill:
        report("[跳过] --skip-backfill：不补嵌入")
    else:
        backfill(kb, config.data_directory, CATEGORY, args.dry_run, known_md5)

    report("=== 迁移结束 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
