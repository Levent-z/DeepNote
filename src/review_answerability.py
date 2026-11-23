"""Rejudge questions marked unanswerable in one or more scored reports."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from score_deepnote_reports import OpenAI, write_json


def collect_questions(paths):
    questions = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = report.get("results")
        if not report.get("summary", {}).get("complete") or not isinstance(rows, list):
            raise ValueError(f"Expected a complete scored report: {path}")
        for row in rows:
            if row.get("status") != "ok" or not isinstance(row.get("llm_judge_answerable"), bool):
                continue
            question, answer, docs = row.get("question"), row.get("answer"), row.get("gold_docs")
            if not isinstance(question, str) or answer is None or not isinstance(docs, list) or any(not isinstance(doc, str) for doc in docs):
                raise ValueError(f"Invalid question, answer or gold_docs in {path}, id={row.get('id')}")
            key = json.dumps([question, answer, docs], ensure_ascii=False, sort_keys=True)
            entry = questions.setdefault(key, {"question": question, "answer": answer, "gold_docs": docs, "original_judgments": []})
            entry["original_judgments"].append({"report": str(path), "id": row.get("id"),
                "retrieve_method": report.get("generation_run", {}).get("retrieve_method"),
                "answerable": row["llm_judge_answerable"], "reason": row.get("answerable_reason")})
    return [entry for entry in questions.values() if any(not item["answerable"] for item in entry["original_judgments"])]


async def review_question(client, question):
    template = '''
    {
    "reason": "",
    "answerable": true
    }
    或者
    {
    "reason": "",
    "answerable": false
    }
    '''
    prompt = f'''
        我将给你一个问题(question)，这个问题的标准答案(answer) ，以及这个问题的参考文档(gold_docs)。
        请回答
        这个问题能不能在仅参考 gold_docs 的情况下被回答，并且得到答案answer？（gold_docs中是否包含足够明确的信息回答这个问题并得到answer）
        question = {question["question"]}
        answer = {question["answer"]}
        gold_docs = {question["gold_docs"]}

        请用json输出答案，格式为 {template}
        请在reason中给出你的判断的理由。理由请用中文给出。
        '''
    response = await asyncio.to_thread(client.chat, [{"role": "user", "content": prompt}], temperature=0)
    content = response.content
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    result = json.loads(content.replace("```json", "").replace("```", "").strip())
    if not isinstance(result.get("answerable"), bool):
        raise ValueError("Judge answerable must be a boolean")
    return {"answerable": result["answerable"], "reason": result.get("reason", "")}


async def run(paths, output, settings):
    paths = [Path(path).resolve() for path in paths]
    output = Path(output).resolve()
    if not paths or len(set(paths)) != len(paths) or output in paths:
        raise ValueError("Report paths must be distinct and different from output")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not settings.get("model") or not settings.get("base_url"):
        raise ValueError("Configure evaluation.model and evaluation.base_url")
    questions = collect_questions(paths)
    client = OpenAI(model=settings["model"], base_url=settings["base_url"],
                    api_key=os.environ.get("EVAL_JUDGE_API_KEY") or settings.get("api_key") or "EMPTY") if questions else None
    results = []
    for question in questions:
        try:
            review = await review_question(client, question)
            question["review"] = {"status": "ok", **review}
        except Exception as exc:
            question["review"] = {"status": "error", "error": str(exc)}
        results.append(question)
        print(f"Reviewed {len(results)}/{len(questions)}: {question['review']['status']}", flush=True)
    reviewed = [item for item in results if item["review"]["status"] == "ok"]
    result = {"judge": {"model": settings["model"], "base_url": settings["base_url"], "temperature": 0},
              "source_reports": [str(path) for path in paths],
              "summary": {"selected_questions": len(questions), "reviewed": len(reviewed),
                          "answerable": sum(item["review"]["answerable"] for item in reviewed),
                          "unanswerable": sum(not item["review"]["answerable"] for item in reviewed),
                          "errors": len(results) - len(reviewed)}, "results": results}
    write_json(output, result)
    return result


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, help="Existing scored reports")
    parser.add_argument("--output", required=True, help="New review JSON file")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/config.yaml"))
    args = parser.parse_args()
    if Path(args.output).resolve() == Path(args.config).resolve():
        parser.error("Output must not overwrite the config")
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["evaluation"]
    result = asyncio.run(run(args.reports, args.output, settings))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
