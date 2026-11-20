"""Score saved DeepNote reports using the scoring rules from run_test.py.

The judge and summary functions follow run_test.py. Embedding recall maps
retrieved chunks back to reference documents before scoring.
This module never calls RetrievalAgent/IterativeRAG or regenerates answers.
"""
import argparse
import asyncio
import bz2
import copy
import hashlib
import json
import os
import re
from pathlib import Path

cache_data = {}


class OpenAI:
    """Keep run_test's chat interface, using the installed official SDK."""
    def __init__(self, model, base_url, api_key):
        from openai import OpenAI as SDK
        self.model = model
        self.client = SDK(base_url=base_url, api_key=api_key or "EMPTY", timeout=120, max_retries=2)

    def chat(self, messages, temperature=0):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        message = response.choices[0].message
        if not isinstance(message.content, str) or not message.content.strip():
            raise ValueError("Judge returned no answer")
        return message


async def check_result(base_url, api_key, question, answer, llm_answer, gold_docs: list, model="deepseek-r1") -> tuple:

    # 初始化LLM
    llm = OpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key
    )

    # 模板定义
    answer_integrity_template = """
    {
    "reason": "",
    "result": "right"
    }
    或者
    {
    "reason": "",
    "result": "wrong"
    }
    """

    answerable_template = """
    {
    "reason": "",
    "answerable": true
    }
    或者
    {
    "reason": "",
    "answerable": false
    }
    """

    # 判断问题能否被回答
    async def check_answerable():
        # 生成answerable的缓存key
        answerable_prompt = f"""
        我将给你一个问题(question)，这个问题的标准答案(answer) ，以及这个问题的参考文档(gold_docs)。
        请回答
        这个问题能不能在仅参考 gold_docs 的情况下被回答，并且得到答案answer？（gold_docs中是否包含足够明确的信息回答这个问题并得到answer）
        question = {question}
        answer = {answer}
        gold_docs = {gold_docs}

        请用json输出答案，格式为 {answerable_template}
        请在reason中给出你的判断的理由。理由请用中文给出。
        """

        cache_key = hashlib.md5(f"answerable|{answerable_prompt}|{model}|{base_url}".encode('utf-8')).hexdigest()

        # 尝试从缓存读取
        global cache_data
        if cache_key in cache_data:
            cached_result = cache_data[cache_key]
            print(f"mengyao_debug check_answerable hit cache")
            return cached_result["answerable"], cached_result["reason"]

        print(f"mengyao_debug check_answerable NOT hit cache, answerable_prompt=\n{answerable_prompt}")

        # 缓存中没有，调用大模型
        def sync_chat_answerable():
            messages = [{"role": "user", "content": answerable_prompt}]
            response = llm.chat(messages, temperature=0)
            return response

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, sync_chat_answerable)

        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()

        try:
            response_json = json.loads(content.replace("```json", "").replace("```", "").strip())
            answerable = response_json["answerable"]
            if not isinstance(answerable, bool):
                raise ValueError("answerable must be a boolean")
            reason = response_json.get("reason", "")

            # 将结果存入缓存
            cache_data[cache_key] = {
                "answerable": answerable,
                "reason": reason,
                "question": question,
                "answer": answer,
                "gold_docs": gold_docs,
            }

            return answerable, reason
        except Exception as E:
            raise ValueError("评判模型返回的 answerable 不是有效 JSON") from E

    # 判断答案是否正确
    async def check_correctness():
        # 先检查是否可以直接匹配
        if (not str(llm_answer).isdigit()) and (not str(answer).isdigit()):
            if str(llm_answer) in str(answer) or str(answer) in str(llm_answer):
                return True, "答案包含"

        # 生成correctness的缓存key
        correctness_prompt = f"""
        我将给你一个问题(question)，这个问题的标准答案(answer) 以及一个大模型对他的回答(llm_answer)。
        请回答
        大模型对他的回答 llm_answer 与 标准答案是否相同或基本相同？
        判断最终回答是否完整、正确地回答了问题。允许同义表达；仅包含标准答案但同时否定它、
        给出互相矛盾的结论或缺少必要答案要点，应判为 wrong。
        question = {question}
        answer = {answer}
        llm_answer = {llm_answer}

        请用json输出答案，格式为 {answer_integrity_template}
        请在reason中给出你的判断的理由。理由请用中文给出。
        """

        cache_key = hashlib.md5(f"correctness|{correctness_prompt}|{model}|{base_url}".encode('utf-8')).hexdigest()

        global cache_data
        if cache_key in cache_data:
            cached_result = cache_data[cache_key]
            print(f"mengyao_debug check_correctness hit cache")
            return cached_result["result"] == "right", cached_result["reason"]

        # 缓存中没有，调用大模型
        def sync_chat_correctness():
            messages = [{"role": "user", "content": correctness_prompt}]
            response = llm.chat(messages, temperature=0)
            return response

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, sync_chat_correctness)

        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()

        try:
            response_json = json.loads(content.replace("```json", "").replace("```", "").strip())
            if response_json.get("result") not in ("right", "wrong"):
                raise ValueError("result must be right or wrong")
            result = response_json["result"] == "right"
            reason = response_json.get("reason", "")

            # 将结果存入缓存
            cache_data[cache_key] = {
                "question": question,
                "result": "right" if result else "wrong",
                "reason": reason
            }

            return result, reason
        except Exception as E:
            raise ValueError("评判模型返回的 correctness 不是有效 JSON") from E

    # 分别检查参考证据的可回答性和最终答案正确性。
    answerable, answerable_reason = await check_answerable()
    result, result_reason = await check_correctness()

    return result, result_reason, answerable, answerable_reason


async def calc_recal_rate(gold_doc: list, retrieve_docs: list) -> float:
    gold_doc = [doc.replace("\n", " ") for doc in gold_doc]
    retrieve_docs = [doc.replace("\n", " ") for doc in retrieve_docs]
    recall_count = 0
    for doc in gold_doc:
        for retrieve_doc in retrieve_docs:
            if doc in retrieve_doc:
                recall_count += 1
                break
    return recall_count / len(gold_doc) if gold_doc else 0.0


def source_paragraphs(dataset, corpus_dir, index_dir=None):
    if dataset == "musique":
        for variant in ("ans", "full"):
            for split in ("train", "dev", "test"):
                path = corpus_dir / f"musique_{variant}_v1.0_{split}.jsonl"
                with path.open(encoding="utf-8-sig") as source:
                    for line in source:
                        for paragraph in json.loads(line)["paragraphs"]:
                            yield paragraph["title"], paragraph["paragraph_text"]
    elif dataset == "2wikimultihopqa":
        for split in ("train", "dev", "test"):
            data = json.loads((corpus_dir / f"{split}.json").read_text(encoding="utf-8-sig"))
            for item in data:
                for title, sentences in item["context"]:
                    yield title, " ".join(sentences)
    elif dataset == "hotpotqa":
        paths = sorted(corpus_dir.glob("*/wiki_*.bz2"))
        if not paths:
            raise FileNotFoundError(f"No HotpotQA corpus files found in {corpus_dir}")
        for path in paths:
            with bz2.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line)
                    yield record["title"], "".join(record["text"])
    elif dataset == "military":
        from build_index.emb.military_data import load_military_records
        stats_path = (index_dir or corpus_dir) / "military.index.stats.json"
        input_path = (Path(__file__).resolve().parents[1] / "data" / "military_input.json")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
            if stats.get("input_path"):
                input_path = Path(stats["input_path"])
        for record in load_military_records(input_path):
            yield "", record["text"]
    else:
        raise ValueError(f"Embedding recall mapping is unsupported for dataset: {dataset}")


def map_chunks(rows, corpus_dir, dataset, index_dir=None):
    chunk_path = (index_dir or corpus_dir) / "chunk.json"
    chunk_data = json.loads(chunk_path.read_text(encoding="utf-8-sig"))
    if not isinstance(chunk_data, list) or any(not isinstance(chunk, str) for chunk in chunk_data):
        raise ValueError(f"Invalid embedding chunks: {chunk_path}")
    chunks = set(chunk_data)
    retrieved = {doc for row in rows if row["status"] != "qa_error" for doc in row["retrieved_results"]}
    if not retrieved <= chunks:
        raise ValueError(f"Retrieved embedding text is missing from {chunk_path}; check that the report and index match")
    normalized = {doc: doc.replace("\n", " ").strip() for doc in retrieved}
    if any(not text for text in normalized.values()):
        raise ValueError("Retrieved embedding chunks must not be empty")
    anchors = {}
    anchor_length = min(32, min((len(text) for text in normalized.values()), default=32))
    for doc, text in normalized.items():
        anchors.setdefault(text[:anchor_length], []).append(doc)
    if not anchors:
        return {}
    pattern = re.compile("(?=(" + "|".join(re.escape(anchor) for anchor in anchors) + "))")
    mapped = {doc: set() for doc in retrieved}
    for title, original in source_paragraphs(dataset, corpus_dir, index_dir):
        combined = (title + "\n" + original if title else original).replace("\n", " ")
        for match in pattern.finditer(combined):
            for doc in anchors[match.group(1)]:
                if normalized[doc] in combined:
                    mapped[doc].add(original)
    missing = [doc for doc, originals in mapped.items() if not originals]
    if missing:
        raise ValueError(f"Could not map {len(missing)} retrieved chunks to {dataset} paragraphs; check the corpus and index")
    return mapped


def summarize_results(results):
    """Keep every selected question in the primary accuracy denominator."""
    total = len(results)
    judged = [q for q in results if q.get("status") == "ok"]
    correct = sum(q.get("llm_judge") is True for q in judged)
    answerable = [q for q in judged if q.get("llm_judge_answerable") is True]
    answerable_correct = sum(q.get("llm_judge") is True for q in answerable)
    total_tokens = sum(q.get("n_token_retrieval", 0) for q in results)
    return {
        "total": total, "judged": len(judged), "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "qa_errors": sum(q.get("status") == "qa_error" for q in results),
        "judge_errors": sum(q.get("status") == "judge_error" for q in results),
        "blocked": sum(q.get("status") == "blocked" for q in results),
        "answerable": len(answerable),
        "accuracy_on_answerable": answerable_correct / len(answerable) if answerable else None,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / total if total else 0.0,
        "avg_qa_elapsed_s": sum(q.get("qa_elapsed_s", 0) for q in results) / total if total else 0.0,
        "avg_multiround_recall_on_answerable": (
            sum(q.get("retrieved_rate", 0) for q in answerable) / len(answerable) if answerable else None
        ),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def reference_documents(question):
    """run_test permits an empty reference list, but not a missing field."""
    if "references" in question:
        refs = question["references"]
        if not isinstance(refs, list) or any(
            not isinstance(ref, (list, tuple)) or not ref or not isinstance(ref[0], str)
            for ref in refs
        ):
            raise ValueError("references must be a list of [document text, ...] entries")
        return [ref[0].replace("...", "") for ref in refs]
    docs = question.get("gold_docs")
    if not isinstance(docs, list) or any(not isinstance(doc, str) for doc in docs):
        raise ValueError("gold_docs must be a list of strings (empty list is allowed), or supply references")
    return docs


def skip_empty_references(questions):
    kept, skipped = [], []
    for position, question in enumerate(questions):
        field = "references" if "references" in question else "gold_docs"
        value = question.get(field)
        empty = value is None or value == [] or value == "" or (
            field == "gold_docs" and isinstance(value, list)
            and all(isinstance(doc, str) and not doc.strip() for doc in value)
        )
        if empty:
            skipped.append({"original_index": position, "question": question.get("question"),
                            "reason": f"Missing or empty {field}"})
        else:
            record = copy.deepcopy(question)
            record["original_question_index"] = position
            kept.append(record)
    if not kept:
        raise ValueError("No questions remain after skipping empty reference documents")
    validate_questions(kept)  # Malformed nonempty fields still fail; do not silently hide errors.
    return kept, skipped


def validate_questions(questions):
    for position, question in enumerate(questions):
        try:
            if question.get("answer") is None or not str(question["answer"]).strip():
                raise ValueError("missing or empty answer")
            reference_documents(question)
        except ValueError as exc:
            raise ValueError(f"Question #{position + 1} (index {position}), {question.get('question', '')[:100]!r}: {exc}") from exc


def prepare_report(path, questions):
    from eval_jybigdata import retrieved_documents
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not report.get("summary", {}).get("complete"):
        raise ValueError(f"Report is incomplete: {path}")
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Report contains no results: {path}")
    expected = report.get("run", {}).get("question_count")
    if expected != len(rows) or [r.get("id") for r in rows] != list(range(len(rows))):
        raise ValueError("Report must contain all selected questions in original positional order")
    if len(questions) < len(rows):
        raise ValueError("Question file is shorter than report")
    prepared = []
    for row in rows:
        question = questions[row["id"]]
        if row.get("question", row.get("query")) != question.get("question"):
            raise ValueError(f"Question mismatch at position {row['id']}")
        if question.get("answer") is None or not str(question["answer"]).strip():
            raise ValueError(f"Missing gold answer at position {row['id']}")
        if row.get("answer") is not None and row["answer"] != question["answer"]:
            raise ValueError(f"Gold answer mismatch at position {row['id']}")
        item = copy.deepcopy(row)
        item["answer"] = question["answer"]
        # Match run_test: references overrides gold_docs; empty gold_docs is permitted.
        item["gold_docs"] = reference_documents(question)
        item["qa_elapsed_s"] = row.get("performance", {}).get("total_seconds", 0)
        item["llm_answer"] = row.get("output")
        item["retrieved_results"] = retrieved_documents(row)
        if row.get("status") == "error" or not isinstance(row.get("output"), str) or not row["output"].strip():
            item["status"] = "qa_error"
        else:
            if "ref_log" not in row:
                raise ValueError("Successful predictions require ref_log for recall scoring")
            item["status"] = "pending"
        prepared.append(item)
    return report, prepared


async def score_report(path, questions, settings, output_path):
    report, rows = prepare_report(path, questions)
    chunk_paragraphs = None
    if (report.get("run", {}).get("retrieve_method") == "emb"
            and report["run"].get("retrieval_unit") != "record"):
        dataset = report["run"].get("dataset")
        if dataset not in ("musique", "2wikimultihopqa", "hotpotqa", "military"):
            raise ValueError(f"Embedding recall mapping is unsupported for dataset: {dataset}")
        corpus_dir = Path(__file__).resolve().parents[1] / "data" / "corpus" / dataset
        index_dir = Path(__file__).resolve().parents[1] / "data" / "index" / dataset
        chunk_paragraphs = map_chunks(rows, corpus_dir, dataset, index_dir)
    if not settings.get("model") or not settings.get("base_url"):
        raise ValueError("Configure evaluation.model and evaluation.base_url")
    workers = int(settings.get("max_workers", 2))
    if workers < 1:
        raise ValueError("evaluation.max_workers must be positive")
    semaphore = asyncio.Semaphore(workers)
    global cache_data
    cache_data = {}  # Fresh per-report cache, using the exact run_test cache keys.
    completed = {}

    async def score(row):
        if row["status"] == "qa_error":
            return row
        retrieved = row["retrieved_results"]
        if chunk_paragraphs is not None:
            retrieved = [paragraph for chunk in retrieved for paragraph in chunk_paragraphs[chunk]]
            row["retrieved_paragraphs"] = retrieved
        row["retrieved_rate"] = await calc_recal_rate(row["gold_docs"], retrieved)
        async with semaphore:
            try:
                right, reason, answerable, answerable_reason = await check_result(
                    base_url=settings["base_url"],
                    api_key=os.environ.get("EVAL_JUDGE_API_KEY") or settings.get("api_key") or "EMPTY",
                    model=settings["model"], question=row["question"], answer=row["answer"],
                    llm_answer=row["llm_answer"], gold_docs=row["gold_docs"],
                )
                row.update(llm_judge=right, llm_judge_reason=reason,
                           llm_judge_answerable=answerable, answerable_reason=answerable_reason, status="ok")
            except Exception as exc:
                row.update(status="judge_error", error=str(exc))
        return row

    def payload(complete):
        # Include unfinished rows as pending so the denominator is always the full selection.
        ordered = [completed.get(row["id"], row) for row in rows]
        summary = summarize_results(ordered)
        # The source runner does not collect SDK tokens. Do not misreport missing usage as zero.
        summary["total_tokens"] = summary["avg_tokens"] = None
        summary.update(complete=complete, completed=len(completed), token_usage_note="Not recorded by DeepNote")
        return {"source_report": str(Path(path).resolve()), "scoring_protocol": "src/run_test.py",
                "judge": {"model": settings["model"], "base_url": settings["base_url"], "temperature": 0},
                "generation_run": report["run"], "generation_summary": report["summary"],
                "summary": summary, "results": ordered}

    for task in asyncio.as_completed([score(copy.deepcopy(row)) for row in rows]):
        row = await task
        completed[row["id"]] = row
        write_json(output_path, payload(False))
        print(f"{Path(path).name}: scored {len(completed)}/{len(rows)}", flush=True)
    result = payload(True)
    write_json(output_path, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return result



def comparison_payload(summaries):
    metrics = []
    for item in summaries:
        summary = item["summary"]
        metrics.append({"retrieve_method": item["retrieve_method"],
                        "accuracy": summary["accuracy"],
                        "recall_on_answerable": summary["avg_multiround_recall_on_answerable"],
                        "total_questions": summary["total"], "answerable_questions": summary["answerable"],
                        "qa_errors": summary["qa_errors"], "judge_errors": summary["judge_errors"]})
    return {"metrics": metrics,
            "metric_definitions": {
                "accuracy": "Correct answers / all selected questions, including errors.",
                "recall_on_answerable": "Average multiround reference recall for judge-accepted answerable questions; null if none.",
            },
            "scoring_protocol": "src/run_test.py", "reports": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/config.yaml"))
    parser.add_argument("--comparison-output", help="Combined summaries; default beside first report")
    args = parser.parse_args()
    import yaml
    questions = json.loads(Path(args.question_file).read_text(encoding="utf-8-sig"))
    if not isinstance(questions, list):
        parser.error("Question file must be a JSON array")
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["evaluation"]
    # Validate ALL inputs before any judge requests; compare the exact same question selection.
    selections = [prepare_report(path, questions)[1] for path in args.reports]
    if any([r["id"] for r in selection] != [r["id"] for r in selections[0]] for selection in selections[1:]):
        parser.error("Reports must cover identical question selections")
    destination = args.comparison_output or str(Path(args.reports[0]).parent / "deepnote_comparison.json")
    protected = {Path(path).resolve() for path in [*args.reports, args.question_file, args.config]}
    output_paths = [str(Path(path).with_suffix(".scored.json")) for path in args.reports]
    resolved = [Path(path).resolve() for path in [*output_paths, destination]]
    if len(set(resolved)) != len(resolved) or any(path in protected for path in resolved):
        parser.error("Scoring outputs must be distinct and must not overwrite inputs/config")
    summaries = []
    for path, output in zip(args.reports, output_paths):
        result = asyncio.run(score_report(path, questions, settings, output))
        summaries.append({"source_report": path, "scored_report": output,
                          "summary": result["summary"], "generation_summary": result["generation_summary"],
                          "retrieve_method": result["generation_run"].get("retrieve_method", "unknown")})
    comparison = comparison_payload(summaries)
    write_json(destination, comparison)
    print("Accuracy and recall (answerable questions):")
    for metric in comparison["metrics"]:
        recall = metric["recall_on_answerable"]
        print(f"  {metric['retrieve_method']}: accuracy={metric['accuracy']:.4f}, "
              f"recall={recall:.4f}" if recall is not None else
              f"  {metric['retrieve_method']}: accuracy={metric['accuracy']:.4f}, recall=N/A")
    print(f"Comparison saved: {destination}")


if __name__ == "__main__":
    main()
