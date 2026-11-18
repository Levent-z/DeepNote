"""jybigdata scoring rules adapted to saved DeepNote predictions.

Judge prompts, substring shortcut, parse fallbacks and recall denominators are
retained from the supplied evaluator. No jybigdata RAG/memory modules required.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path

cache_data = {}


class OpenAI:
    """Adapt the original llm.chat(messages) call to the official SDK."""
    def __init__(self, model, base_url, api_key):
        from openai import OpenAI as SDKClient
        self.model = model
        self.client = SDKClient(base_url=base_url, api_key=api_key or "EMPTY",
                                timeout=120, max_retries=3)

    def chat(self, messages):
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        message = response.choices[0].message
        if not message.content:
            raise ValueError("Judge returned no text content")
        return message


async def check_result(base_url, api_key, question, answer, llm_answer, gold_docs: list, model="deepseek-r1") -> (
bool, str, bool, str):

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
    "result": "right",
    }
    或者
    {
    "reason": "",
    "result": "wrong",
    }
    """

    answerable_template = """
    {
    "reason": "",
    "answerable": true/false,
    }
    或者
    {
    "reason": "",
    "answerable": true/false,
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
            return cached_result["answerable"], cached_result["reason"]


        # 缓存中没有，调用大模型
        def sync_chat_answerable():
            messages = [{"role": "user", "content": answerable_prompt}]
            response = llm.chat(messages)
            return response

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, sync_chat_answerable)

        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()

        try:
            response_json = json.loads(content.replace("```json", "").replace("```", "").strip())
            answerable = response_json.get("answerable", True)
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
            print(f"解析answerable响应失败: {E}")
            return True, "解析失败，默认返回可回答"

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
        注意这两个问题是独立的。
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
            return cached_result["result"] == "right", cached_result["reason"]

        # 缓存中没有，调用大模型
        def sync_chat_correctness():
            messages = [{"role": "user", "content": correctness_prompt}]
            response = llm.chat(messages)
            return response

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, sync_chat_correctness)

        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()

        try:
            response_json = json.loads(content.replace("```json", "").replace("```", "").strip())
            result = response_json.get("result", "wrong") == "right"
            reason = response_json.get("reason", "")

            # 将结果存入缓存
            cache_data[cache_key] = {
                "question": question,
                "result": "right" if result else "wrong",
                "reason": reason
            }

            return result, reason
        except Exception as E:
            print(f"解析correctness响应失败: {E}")
            return False, "解析失败，默认返回错误"

    # 并行执行两个检查
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
    return float(recall_count) / float(len(gold_doc)+0.001)


def gold_documents(question):
    # Match the original evaluator: references takes precedence over gold_docs.
    if "references" in question:
        refs = question["references"]
        if not isinstance(refs, list) or any(
            not isinstance(ref, (list, tuple)) or not ref or not isinstance(ref[0], str)
            for ref in refs
        ):
            raise ValueError("references must be a list of [document text, ...] entries")
        docs = [ref[0].replace("...", "") for ref in refs]
    else:
        docs = question.get("gold_docs")
    if not isinstance(docs, list) or not docs or any(not isinstance(d, str) or not d.strip() for d in docs):
        raise ValueError("Each evaluation question needs nonempty gold_docs or references")
    return docs


def retrieved_documents(prediction):
    documents = []
    for step in prediction.get("ref_log", []):
        for ref in step.get("refs", []):
            if isinstance(ref, str):
                documents.append(ref)
            elif isinstance(ref, dict):
                text = ref.get("paragraph_text", ref.get("text"))
                if not isinstance(text, str):
                    raise ValueError("Retrieved document is missing text/paragraph_text")
                documents.append(text)
            else:
                raise ValueError("Unsupported retrieved document format")
    return documents


def load_predictions(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [row for row in rows if isinstance(row, dict) and "id" in row and "output" in row]


def summarize(results):
    answerable = [r for r in results if r.get("llm_judge_answerable") is True]
    correct = sum(r.get("llm_judge") is True for r in answerable)
    # Keep the supplied evaluator's epsilon and answerable-only recall average.
    return {
        "evaluation": "jybigdata",
        "total_questions": len(results),
        "total_answerable": len(answerable),
        "total_correct": correct,
        "accuracy": correct / len(answerable) if answerable else 0,
        "recall_rate": sum(r["retrieved_rate"] for r in answerable) / (len(answerable) + 0.001),
        "evaluation_errors": sum("evaluation_error" in r for r in results),
        "token_usage": None,
        "token_usage_note": "DeepNote does not currently record generation token usage; not estimated",
    }


async def evaluate_predictions(predictions, settings, output_path, questions=None):
    if not predictions:
        raise ValueError("No DeepNote predictions to evaluate")
    if not settings.get("base_url") or not settings.get("model"):
        raise ValueError("Set evaluation.base_url and evaluation.model for the judge service")
    workers = int(settings.get("max_workers", 2))
    if workers < 1:
        raise ValueError("evaluation.max_workers must be positive")
    prepared = []
    seen = set()
    for pred in predictions:
        idx = pred["id"]
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx in seen:
            raise ValueError("DeepNote prediction IDs must be unique nonnegative question positions")
        seen.add(idx)
        source = pred if questions is None else questions[idx]
        query = source.get("question", source.get("query"))
        if not isinstance(query, str) or not query.strip() or "answer" not in source:
            raise ValueError(f"Missing question/answer for prediction {idx}")
        if pred.get("question", pred.get("query")) != query:
            raise ValueError(f"Question mismatch for prediction {idx}; check question file/order")
        if pred["answer"] != source["answer"]:
            raise ValueError(f"Gold answer mismatch for prediction {idx}")
        if "ref_log" not in pred:
            raise ValueError("jybigdata recall requires DeepNote ref_log; use --method deepnote")
        docs = gold_documents(source)
        retrieved = retrieved_documents(pred)
        prepared.append((pred, query, source["answer"], docs, retrieved))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(str(output_path) + ".judge_cache.json")
    global cache_data
    cache_data = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    semaphore = asyncio.Semaphore(workers)
    results = []

    async def score(item):
        pred, query, answer, docs, retrieved = item
        row = copy.deepcopy(pred)
        row.update(question=query, gold_docs=docs, llm_answer=pred["output"], retrieved_results=retrieved)
        row["retrieved_rate"] = await calc_recal_rate(docs, retrieved)
        async with semaphore:
            try:
                right, reason, answerable, answerable_reason = await check_result(
                    base_url=settings["base_url"], api_key=settings.get("api_key", "EMPTY"),
                    question=query, answer=answer, llm_answer=pred["output"], gold_docs=docs,
                    model=settings["model"],
                )
                row.update(llm_judge=right, llm_judge_reason=reason,
                           llm_judge_answerable=answerable, answerable_reason=answerable_reason)
            except Exception as exc:
                row["evaluation_error"] = str(exc)
        return row

    for task in asyncio.as_completed([score(item) for item in prepared]):
        results.append(await task)
        ordered = sorted(results, key=lambda r: r["id"])
        output_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Judge progress: {len(results)}/{len(prepared)}", flush=True)
    summary = summarize(results)
    summary["judge"] = {"model": settings["model"], "base_url": settings["base_url"]}
    Path(str(output_path) + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_path", required=True, help="DeepNote JSONL predictions")
    parser.add_argument("--question_file", help="Same original question JSON array used for prediction, including gold_docs/references")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/config.yaml"))
    parser.add_argument("--output_path", help="Scored result JSON; default: <pred_path>.jybigdata.json")
    args = parser.parse_args()
    import yaml
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")).get("evaluation", {})
    questions = json.loads(Path(args.question_file).read_text(encoding="utf-8")) if args.question_file else None
    asyncio.run(evaluate_predictions(load_predictions(args.pred_path), settings,
                                     args.output_path or args.pred_path + ".jybigdata.json", questions))


if __name__ == "__main__":
    main()
