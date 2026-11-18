import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import hashlib
import asyncio
import time
import argparse
from datetime import datetime
from pathlib import Path

from llm.openai_llm import OpenAI
from rag.agent.rag_agent import RetrievalAgent
from rag.rag_flow.iterative_rag import IterativeRAG
import traceback

_MEMORY_CLIENT = None
REPO_ROOT = Path(__file__).resolve().parents[2]


answer_exception = "EXCEPTION"
cache_data = {}


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


async def run_query(model, base_url, rag_url, api_key, dataset, query, max_iterations: int, topk: int, topk_first_stage:int,
                    reflect_final_answer=False,
                    reflect_intermediate_answer=False,
                    rerank_LLM=False,
                    add_decompose_guideline=True,
                    skip_first_attempt=True,
                    with_memory=False):
    """针对指定dataset执行查询"""
    llm = OpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key
    )

    agent = RetrievalAgent(
        llm=llm,
        func_name="big_data_base",
        dataset=dataset,
        url=rag_url,
        memory_client=_MEMORY_CLIENT,
    )

    iterative_rag_flow = IterativeRAG(
        agent=agent,
        max_iterations=max_iterations,
        topk=topk,
        topk_first_stage=topk_first_stage
    )

    try:
        all_retrieved_docs = []
        all_hybrid_retrieval_results = []
        all_retrieved_docs_supported = []
        (answer, reason, retrieved_results, n_token_retrieval, intermediate_context, reflect_result,
         reflect_reason, decompose_reasoning, inappropriate_content, work_memory_list,
         hybrid_retrieval_results, proc_payload) = \
            await iterative_rag_flow.arun_with_reason(query,
                                                      ws_client=None,
                                                      all_retrieved_docs=all_retrieved_docs,
                                                      all_hybrid_retrieval_results=all_hybrid_retrieval_results,
                                                      all_retrieved_docs_supported=all_retrieved_docs_supported,
                                                      reflect_final_answer=reflect_final_answer,
                                                      reflect_intermediate_answer=reflect_intermediate_answer,
                                                      rerank_LLM=rerank_LLM,
                                                      add_decompose_guideline=add_decompose_guideline,
                                                      skip_first_attempt=skip_first_attempt,
                                                      llm_model_rerank=model,
                                                      with_memory=with_memory)

        if inappropriate_content:
            print(f"[run_query] inappropriate_content")
            return (answer, reason, retrieved_results, n_token_retrieval, intermediate_context,
                    all_retrieved_docs, all_retrieved_docs_supported, reflect_result, reflect_reason,
                    decompose_reasoning, inappropriate_content, work_memory_list, hybrid_retrieval_results, None)

        retrieved_docs = [
            {"text": r.text, "reference": r.reference}
            for r in retrieved_results
        ]

        print("\n==================== 查询结果 ====================")
        print(f"数据集: {dataset}")
        print(f"问题: {query}")
        print(f"\n答案:\n{answer}\n")

        print("---------- 中间上下文 ----------")
        print(intermediate_context)

        print("\n---------- 检索文档 ----------")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"[{i}] 来源: {doc['reference']}")
            print(f"内容: {doc['text'][:200]}...\n")

        print("---------- Token 使用 ----------")
        print(n_token_retrieval)
        print("=================================================\n")
        return (answer, reason, retrieved_results, n_token_retrieval, intermediate_context,
                all_retrieved_docs, all_retrieved_docs_supported, reflect_result, reflect_reason,
                decompose_reasoning, inappropriate_content, work_memory_list, hybrid_retrieval_results, proc_payload)

    except Exception as e:
        print(f"❌ 查询失败: {str(e)}")
        traceback.print_exc()
        return answer_exception, str(e), [], 0, [], [], [], None, None, None, False, [], [], None
    finally:
        agent.thread_pool.shutdown(wait=False)


async def get_lasttime_wrong_questions(result_dir: str, total_run: int):
    directory = Path(result_dir)

    if not directory.exists() or not directory.is_dir():
        return None

    # 获取所有文件的修改时间
    files_with_mtime = [
        (file, file.stat().st_mtime)
        for file in directory.iterdir()
        if file.is_file() and file.name.startswith("result_") and file.suffix == ".json"
    ]

    if not files_with_mtime:
        return None

    # 找到修改时间最大的文件
    latest_file, latest_mtime = max(files_with_mtime, key=lambda x: x[1])
    with open(latest_file, 'r') as f:
        questions = json.loads(f.read())
    questions = [q for q in questions if q.get("llm_judge") is False]

    return questions[:total_run]


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


async def query_question_set(dataset: str, total_run: int, question_file: str,
                             model="deepseek-v3",
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                             base_judge_url=None,
                             topk=10,
                             api_key="",
                             api_key_judge="",
                             topk_first_stage=200,
                             max_iterations=4,
                             start_index: int = 0,
                             rag_url="http://127.0.0.1:8011/passages",
                             max_workers=8,
                             last_time_wrong=False,
                             reflect_intermediate_answer=False,
                             reflect_final_answer=False,
                             skip_first_attempt=True,
                             add_decompose_guideline=True,
                             keyword="",
                             enable_thinking=False,
                             rerank_LLM=True,
                             with_memory=False,
                             model_judge=None,
                             output_dir=None):

    if not 1 <= topk <= 30:
        raise ValueError("Hybrid /passages 要求 1 <= topk <= 30")
    if max_workers < 1 or total_run < 1 or start_index < 0 or max_iterations < 1:
        raise ValueError("并发、题数和迭代轮数必须为正数，起始下标不能为负数")
    base_judge_url = base_judge_url or base_url
    api_key_judge = api_key_judge or api_key
    model_judge = model_judge or model

    os.environ["enable_thinking"] = str(enable_thinking)
    result_dir = str(output_dir or REPO_ROOT / "data" / dataset / "result_")
    os.makedirs(result_dir, exist_ok=True)
    keyword = keyword or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if Path(keyword).name != keyword:
        raise ValueError("keyword 不能包含路径分隔符")
    result_file = os.path.join(result_dir, f"result_{keyword}.json")
    with open(question_file, 'r') as f:
        questions = json.load(f)
        if not isinstance(questions, list):
            raise ValueError("题目文件必须为 JSON 数组")
        questions = questions[start_index: start_index + total_run]
        print(f"questions total is {len(questions)}")

    if last_time_wrong:
        questions = await get_lasttime_wrong_questions(result_dir, total_run=total_run) or []
        print(f"questions total is {len(questions)}")

    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("每题必须为 JSON 对象")
        if "references" in question:
            question["gold_docs"] = [
                ref[0].replace("...", "") for ref in question["references"]
            ]
        if (not isinstance(question.get("question"), str) or not question["question"].strip()
                or question.get("answer") is None or not str(question["answer"]).strip()
                or not isinstance(question.get("gold_docs"), list)
                or not all(isinstance(doc, str) for doc in question["gold_docs"])):
            raise ValueError("每题必须包含非空 question、answer，以及字符串列表 gold_docs（或 references）")

    if not questions:
        raise ValueError("没有可评测的问题，请检查题目文件和切片范围")

    global _MEMORY_CLIENT
    _MEMORY_CLIENT = None
    if with_memory:
        from memory.client import AsyncMemoryClient
        _MEMORY_CLIENT = AsyncMemoryClient(host=os.environ.get("MEMORY_SERVICE_HOST", "http://localhost:8088"))
        try:
            await _MEMORY_CLIENT.fetch_status()
        except Exception:
            await _MEMORY_CLIENT.aclose()
            _MEMORY_CLIENT = None
            raise
        print(f"[memory] enabled: work_memory={_MEMORY_CLIENT.has_work_memory} "
              f"procedural={_MEMORY_CLIENT.has_procedural} reflection={_MEMORY_CLIENT.has_reflection}")

    total_questions_count = len(questions)
    print(f"total_questions_count is {total_questions_count}")

    actual_max_workers = max_workers
    # 统一的并发控制策略
    semaphore = asyncio.Semaphore(actual_max_workers)

    async def process_question(question):
        question_copy = copy.deepcopy(question)
        query = question["question"]
        answer = question["answer"]
        print(f"mengyao_debug query = {query}")

        started = time.monotonic()
        query_kwargs = dict(
            model=model, base_url=base_url, rag_url=rag_url, api_key=api_key,
            topk_first_stage=topk_first_stage, dataset=dataset, query=query,
            max_iterations=max_iterations, topk=topk,
            reflect_final_answer=reflect_final_answer, rerank_LLM=rerank_LLM,
            reflect_intermediate_answer=reflect_intermediate_answer,
            skip_first_attempt=skip_first_attempt,
            add_decompose_guideline=add_decompose_guideline, with_memory=with_memory,
        )
        for attempt in range(4):
            (llm_answer, reason, retrieved_results, n_token_retrieval, intermediate_context,
             all_retrieved_docs, all_retrieved_docs_supported, reflect_result, reflect_reason,
             decompose_reasoning, inappropriate_content, work_memory_list,
             hybrid_retrieval_results, proc_payload) = await run_query(**query_kwargs)
            if inappropriate_content or llm_answer != answer_exception:
                break

        question_copy["attempts"] = attempt + 1
        question_copy["qa_elapsed_s"] = round(time.monotonic() - started, 4)
        question_copy["llm_answer"] = llm_answer
        question_copy["hybrid_retrieval_results"] = hybrid_retrieval_results
        question_copy["n_token_retrieval"] = n_token_retrieval
        if inappropriate_content:
            question_copy["status"] = "blocked"
            return question_copy
        if llm_answer == answer_exception:
            question_copy["status"] = "qa_error"
            question_copy["error"] = reason
            return question_copy

        print(f"mengyao_debug llm_answer = {llm_answer}")
        print(f"mengyao_debug retrieved_results = {retrieved_results}")
        print(f"mengyao_debug retrieved_results = {[r.text for r in retrieved_results]}")
        print(f"mengyao_debug n_token_retrieval = {n_token_retrieval}")
        print(f"mengyao_debug intermediate_context = {len(intermediate_context)}, "
              f"all_retrieved_docs = {len(all_retrieved_docs)}")

        all_retrieved_docs_list = []
        for retrive_result in all_retrieved_docs:
            all_retrieved_docs_list.extend([r.text for r in retrive_result])

        question_copy["llm_answer"] = llm_answer
        question_copy["llm_reason"] = reason
        try:
            question_copy["decompose_reasoning"] = json.loads(decompose_reasoning)
        except:
            question_copy["decompose_reasoning"] = decompose_reasoning
        question_copy["reflect_result"] = reflect_result
        question_copy["reflect_reason"] = reflect_reason
        question_copy["n_token_retrieval"] = n_token_retrieval
        question_copy["intermediate_context"] = []
        for i, ic in enumerate(intermediate_context):
            entry = {
                "context": ic,
                "work_memory": work_memory_list[i] if i < len(work_memory_list) else "",
                "retrieve docs": [doc.text for doc in all_retrieved_docs[i]] if i < len(all_retrieved_docs) else [],
                "retrieve docs supported": (
                    [doc.text for doc in all_retrieved_docs_supported[i]] if i < len(all_retrieved_docs_supported) else []
                ),
            }
            parts = ic.split("\n", 2)
            if len(parts) == 3:
                entry.update(query=parts[0], answer=parts[1], reason=parts[2])
            question_copy["intermediate_context"].append(entry)

        question_copy["retrieved_results"] = [r.text for r in retrieved_results]
        question_copy["retrieved_rate"] = await calc_recal_rate(question["gold_docs"], all_retrieved_docs_list)
        try:
            (question_copy["llm_judge"], question_copy["llm_judge_reason"],
             question_copy["llm_judge_answerable"], question_copy["answerable_reason"]) = await check_result(
                base_url=base_judge_url, api_key=api_key_judge, model=model_judge,
                gold_docs=question_copy["gold_docs"], question=query, answer=answer, llm_answer=llm_answer,
            )
            question_copy["status"] = "ok"
        except Exception as exc:
            question_copy["status"] = "judge_error"
            question_copy["error"] = str(exc)
            return question_copy

        # 用标准答案对比结果作为 success，一次性存入 procedural memory
        if with_memory and _MEMORY_CLIENT is not None and _MEMORY_CLIENT.has_procedural and proc_payload is not None:
            try:
                llm_judge = bool(question_copy["llm_judge"])
                reflection_advice = ""
                try:
                    ref_result = await _MEMORY_CLIENT.reflect(
                        query=proc_payload["query"],
                        intermediate_context=proc_payload.get("intermediate_context", []),
                        decompose_reasoning=proc_payload.get("decompose_reasoning", ""),
                        success=llm_judge,
                        language=proc_payload.get("language", "CHINESE"),
                    )
                    reflection_advice = ref_result.get("reflection_advice", "")
                except Exception as _re:
                    print(f"[memory] reflect failed: {_re}")
                await _MEMORY_CLIENT.save_procedural(
                    query=proc_payload["query"],
                    steps=proc_payload["steps"],
                    success=llm_judge,
                    final_answer=proc_payload["final_answer"],
                    reflection_advice=reflection_advice,
                )
                print(f"[memory] procedural saved with llm_judge={llm_judge} reflection={reflection_advice[:60]!r}")
            except Exception as _fe:
                print(f"[memory] save_procedural failed: {_fe}")

        # print(f"question_copy = {question_copy}")
        return question_copy

    async def limited_process_question(question):
        async with semaphore:
            try:
                return await process_question(question)
            except Exception as e:
                print(f"处理问题失败: {question.get('question', 'Unknown')}, 错误: {e}")
                traceback.print_exc()
                question_copy = copy.deepcopy(question)
                question_copy["llm_answer"] = answer_exception
                question_copy["error"] = str(e)
                question_copy["status"] = "qa_error"
                return question_copy

    # 创建所有任务
    tasks = [limited_process_question(question) for question in questions]
    final_res = []

    print(f"开始处理 {len(tasks)} 个问题，最大并发数: {actual_max_workers}")

    # 每个任务捕获自身异常，避免按完成顺序错误关联失败题目。
    try:
        for future in asyncio.as_completed(tasks):
            final_res.append(await future)
            if len(final_res) % 10 == 0 or len(final_res) == len(tasks):
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(final_res, f, ensure_ascii=False, indent=2)
                print(f"进度: {len(final_res)}/{len(tasks)}")
    finally:
        if _MEMORY_CLIENT is not None:
            await _MEMORY_CLIENT.aclose()
            _MEMORY_CLIENT = None

    summary = summarize_results(final_res)
    summary["config"] = {
        "dataset": dataset, "question_file": str(Path(question_file).resolve()),
        "start_index": start_index, "total_run": total_run,
        "model": model, "base_url": base_url, "rag_url": rag_url,
        "judge_model": model_judge, "judge_base_url": base_judge_url,
        "topk": topk, "max_iterations": max_iterations, "max_workers": max_workers,
        "with_memory": with_memory, "enable_thinking": enable_thinking,
        "reflect_final_answer": reflect_final_answer,
        "reflect_intermediate_answer": reflect_intermediate_answer,
        "skip_first_attempt": skip_first_attempt, "add_decompose_guideline": add_decompose_guideline,
        "topk_first_stage": topk_first_stage, "rerank_LLM": rerank_LLM,
        "note": "Hybrid 召回量和重排由服务端配置；旧 topk_first_stage/rerank_LLM 不控制 Hybrid。",
    }
    summary_file = os.path.join(result_dir, f"summary_{keyword}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"逐题结果: {result_file}\n汇总指标: {summary_file}")

    return final_res


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


def main(argv=None):
    parser = argparse.ArgumentParser(description="通过 Hybrid /passages 评测多轮 RAG 的最终问答效果")
    parser.add_argument("--dataset", required=True, help="Hybrid registry 中的数据集 ID")
    parser.add_argument("--question-file", required=True, help="question/answer/gold_docs JSON 数组")
    parser.add_argument("--model", default=os.getenv("ITER_RAG_MODEL_NAME"))
    parser.add_argument("--base-url", default=os.getenv("ITER_RAG_MODEL_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("ITER_RAG_MODEL_KEY", "dummy"))
    parser.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL"))
    parser.add_argument("--judge-base-url", default=os.getenv("EVAL_JUDGE_BASE_URL"))
    parser.add_argument("--judge-api-key", default=os.getenv("EVAL_JUDGE_API_KEY"))
    hybrid_url = os.getenv("HYBRID_RETRIEVAL_URL", "http://127.0.0.1:8011").rstrip("/")
    parser.add_argument("--rag-url", default=hybrid_url if hybrid_url.endswith("/passages") else hybrid_url + "/passages")
    parser.add_argument("--total-run", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--topk", type=int, choices=range(1, 31), default=15)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--output-dir", help="默认 data/<dataset>/result_")
    parser.add_argument("--keyword", default="", help="输出文件标识，默认当前时间")
    parser.add_argument("--with-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=os.getenv("enable_thinking", "false").lower() in ("true", "1", "yes", "on"))
    parser.add_argument("--reflect-final-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reflect-intermediate-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-first-attempt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-decompose-guideline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not args.model or not args.base_url:
        parser.error("请设置 --model/--base-url 或 ITER_RAG_MODEL_NAME/ITER_RAG_MODEL_BASE_URL")
    options = vars(args)
    options["model_judge"] = options.pop("judge_model")
    options["base_judge_url"] = options.pop("judge_base_url")
    options["api_key_judge"] = options.pop("judge_api_key")
    asyncio.run(query_question_set(**options))


if __name__ == "__main__":
    main()
