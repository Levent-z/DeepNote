import os
import json
import argparse
import logging
import datetime
import asyncio
import time
from pathlib import Path
import yaml
import faiss
import pandas as pd
import backoff
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from eval import acc_score, F1_scorer, compute_exact, eval_asqa, acc_choice
from utils import seed_everything
from embedding_api import EmbeddingAPI
from prediction_report import load_questions, limit_questions, PerformanceTracker, save_report
from generation_api import GenerationAPI

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model to use",
)
parser.add_argument(
    "--max_step", type=int, default=3, help="Maximum number of update steps"
)
parser.add_argument(
    "--max_fail_step", type=int, default=2, help="Maximum number of failed steps"
)
parser.add_argument(
    "--MaxClients", type=int, default=1, help="Number of concurrent clients"
)
parser.add_argument(
    "--retrieve_top_k",
    type=int,
    default=5,
    help="Number of documents to retrieve per query",
)
parser.add_argument(
    "--max_top_k",
    type=int,
    default=15,
    help="Total maximum number of documents to retrieve",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "2wikimultihopqa",
        "hotpotqa",
        "musique",
        "asqa",
        "strategyqa",
        "military",
    ],
    default="hotpotqa",
    help="Dataset to use",
)
parser.add_argument(
    "--method",
    type=str,
    default="deepnote",
    choices=["deepnote", "base", "base_wo_retri"],
    help="Method to use",
)
parser.add_argument(
    "--resume_path",
    type=str,
    default="",
    help="Path to the file for resuming generation",
)
parser.add_argument(
    "--retrieve_method",
    type=str,
    default="es",
    help="Retrieval method to use (es: ElasticSearch, emb: Dense Retrieval)",
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to run inference on"
)
parser.add_argument("--eval_backend", choices=["jybigdata", "original", "none"], default="jybigdata",
                    help="Unified LLM judge evaluation or original Acc/F1/EM evaluation")
parser.add_argument("--question_file", default="",
                    help="Question JSON array or JSONL; none mode needs only question")
parser.add_argument("--output_path", default="", help="Single JSON report path in --eval_backend none mode")
parser.add_argument("--max_questions", type=int, default=None,
                    help="Process at most the first N questions; omit to use all questions")
parser.add_argument("--model_backend", choices=["auto", "api", "local"], default="auto",
                    help="api calls generation service; local loads vLLM weights; auto keeps legacy behavior")
args = parser.parse_args()
if args.eval_backend == "jybigdata" and args.method != "deepnote":
    parser.error("jybigdata evaluation currently requires --method deepnote for retrieval logs")
if args.dataset == "military" and args.retrieve_method != "emb":
    parser.error("military currently requires --retrieve_method emb")
if args.MaxClients < 1:
    parser.error("--MaxClients must be positive")
if args.max_questions is not None and args.max_questions < 1:
    parser.error("--max_questions must be a positive integer; omit it to use all questions")
if args.eval_backend == "none" and args.resume_path:
    parser.error("Generation-only reports do not yet support --resume_path")
if args.output_path and args.eval_backend != "none":
    parser.error("--output_path currently requires --eval_backend none")

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
if args.model_backend == "api":
    args.model = args.model or config.get("generation", {}).get("model")
    if not args.model:
        parser.error("API mode requires generation.model or --model")
else:
    args.model = args.model or "llama-3.1-8b-instruct"


@backoff.on_exception(backoff.expo, (Exception), max_time=500)
def call_gpt(prompt_file, variable_dict):
    with open(prompt_file, "r") as fin:
        prompt = fin.read().format(**variable_dict)
    res = gpt_gen(args.model, prompt)
    assert res is not None
    return res


def call_local(prompt_file, variable_dict):

    with open(prompt_file, "r") as fin:
        prompt = fin.read()
    if "llama" in args.model:
        model_template = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if "qwen" in args.model:
        model_template = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt = model_template.format(prompt=prompt.format(**variable_dict))
    response = llm.generate(prompt, sampling_params)[0].outputs[0].text
    return response


if args.model_backend == "api":
    call_llm = GenerationAPI(config.get("generation", {}), args.model)
elif args.model_backend == "auto" and "gpt" in args.model:
    from utils import gpt_gen

    call_llm = call_gpt

else:
    from vllm import LLM, SamplingParams
    if args.model not in config["model"] or not any(name in args.model for name in ("llama", "qwen")):
        parser.error("Unknown local model; use --model_backend api for a deployed Jiugе service")
    llm = LLM(
        model=config["model"][args.model],
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(max_tokens=1280, temperature=0.1, top_p=0.9)

    call_llm = call_local


def get_context(data):

    if retrieve_method == "emb":
        text = "\n".join(data)
    else:
        text = ""
        for i in range(len(data)):
            text += f"Title: {data[i]['title']}\nText: {data[i]['paragraph_text']} \n\n"
    return text


def save_log_to_file(logger, log_file="my_log", log_folder="logs"):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_date = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    log_file_name = f"{log_file}-{current_date}.log"
    file_handler = logging.FileHandler(os.path.join(log_folder, log_file_name))
    logger.addHandler(file_handler)


def is_directory_empty(directory_path: str) -> bool:
    try:
        return len(os.listdir(directory_path)) == 0
    except OSError:
        return False


def call_llm_template(template, variables):
    return call_llm(f"../prompts/{LAUGUAGE}/{template}", variables)


def init_note(query, refs):
    return call_llm_template("init_note", {"query": query, "refs": refs})


def gen_new_query(query, note, query_log):
    return call_llm_template(
        "gen_new_query", {"query": query, "note": note, "query_log": query_log}
    )


def refine_note(note, query, refs):
    return call_llm_template(
        "refine_note", {"note": note, "query": query, "refs": refs}
    )


def gen_answer(query, note_str):
    if args.dataset in ["asqa", "strategyqa"]:
        template = f"gen_answer_{args.dataset}"
    else:
        template = "gen_answer"
    return call_llm_template(template, {"query": query, "note": note_str})


def compare_note(query, best_note, new_note):
    response = call_llm_template(
        "compare", {"query": query, "best_note": best_note, "new_note": new_note}
    )

    try:
        if "json" in response:
            response = response[response.index("{") : response.rindex("}") + 1]
        status = json.loads(response)["status"]
        return status.lower() == "true"
    except:
        return "true" in response.lower()


def retrieve_note(doc_id, query, answer, top_k=2):
    ref_log, llm_times, note_log, query_log, query_list = [], 0, [], [], []
    refs = retrieve(args.dataset, query=query, topk=top_k)
    note = best_note = init_note(query, get_context(refs))
    ref_log.append({"refs": refs, "step": 0, "flag": "init_refs"})
    note_log.append({"note": note, "step": 0, "flag": "init_note"})
    llm_times += 1
    step, notes_status = 0, []

    while step < args.max_step:
        all_refs = {
            ref
            for d in ref_log
            for ref in (
                d["refs"]
                if retrieve_method == "emb"
                else [d["title"] + d["paragraph_text"] for d in d["refs"]]
            )
        }
        if len(all_refs) > args.max_top_k:
            break
        max_ref = (
            args.max_top_k - len(all_refs)
            if (len(all_refs) + top_k) > args.max_top_k
            else 0
        )

        new_query = gen_new_query(query, best_note, str(query_list))
        query_list.append(new_query)
        llm_times += 1

        refs = retrieve(
            args.dataset, query=(new_query + "\n" + query)[:500], topk=top_k
        )
        if max_ref > 0:
            refs = (
                [d for d in refs if d not in all_refs][:max_ref]
                if retrieve_method == "emb"
                else [
                    d for d in refs if d["title"] + d["paragraph_text"] not in all_refs
                ][:max_ref]
            )

        note = refine_note(best_note, query, get_context(refs)).replace("\n", "")
        llm_times += 1
        status = compare_note(query, best_note, note)
        flag = "True" if status else "False"

        ref_log.append({"refs": refs, "step": step, "flag": flag})
        note_log.append({"note": note, "step": step, "flag": flag})
        query_log.append({"query": new_query, "step": step, "flag": flag})

        if status:
            best_note = note
        notes_status.append(status)
        if notes_status.count(False) >= args.max_fail_step:
            break
        step += 1

    llm_times += 1
    return {
        "id": doc_id,
        "question": query,
        "answer": answer,
        "output": gen_answer(query, best_note),
        "deepnote": best_note,
        "query_log": query_log,
        "note_log": note_log,
        "ref_log": ref_log,
    }


def process_doc_cell(idx, doc_cell, args):
    id_new, query, answer = idx, doc_cell["question"], doc_cell.get("answer")

    if args.method != "deepnote":
        prompt_path = f"../prompts/{LAUGUAGE}"
        if args.dataset in ["asqa", "strategyqa"]:
            gen_name = f"{args.method}_{args.dataset}"
        else:
            gen_name = args.method
        output = (
            call_llm(f"{prompt_path}/{gen_name}", {"query": query})
            if "wo_retri" in args.method
            else call_llm(
                f"{prompt_path}/{gen_name}",
                {
                    "query": query,
                    "refs": get_context(
                        retrieve(args.dataset, query=query, topk=args.retrieve_top_k)
                    ),
                },
            )
        )

        return {"id": id_new, "query": query, "answer": answer, "output": output}
    else:
        return retrieve_note(id_new, query, answer, top_k=args.retrieve_top_k)


if __name__ == "__main__":

    LAUGUAGE = "en"
    seed_everything(66)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    save_log_to_file(
        logger,
        log_file=f"{args.dataset}_{args.method}_{args.model}",
        log_folder="../log",
    )
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    dataset_name = args.dataset
    vector_path = f"../data/corpus/{dataset_name}/{dataset_name}.index"
    use_embedding_api = config.get("embedding", {}).get("provider", "local") == "api"
    if args.dataset == "asqa" or args.dataset == "strategyqa":

        vector = faiss.read_index(f"../data/corpus/wiki/wiki.index")
        if use_embedding_api:
            emb_model = EmbeddingAPI(config["embedding"])
            emb_model.validate_index("../data/corpus/wiki/wiki.index")
        else:
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer(config["model"]["gtr-t5-xxl"], device="cpu")
        raw_data = pd.read_csv("../data/corpus/wiki/psgs_w100.tsv", sep="\t")

        def retrieve(_, query, topk):
            feature = emb_model.encode([query], query=True) if use_embedding_api else emb_model.encode([query])
            _, match_id = vector.search(feature, topk)
            return [
                raw_data.iloc[i]["title"] + "\n" + raw_data.iloc[i]["text"]
                for i in match_id[0] if 0 <= i < len(raw_data)
            ]

    elif args.retrieve_method == "emb":

        if use_embedding_api:
            emb_model = EmbeddingAPI(config["embedding"])
            emb_model.validate_index(vector_path)
        else:
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer(
                config["model"]["bge-base-en-v1.5"], device=args.device
            )
        with open(f"../data/corpus/{args.dataset}/chunk.json", encoding="utf-8") as f:
            raw_data = json.load(f)
        vector = faiss.read_index(vector_path)

        def retrieve(_, query, topk):
            feature = emb_model.encode([query], query=True) if use_embedding_api else emb_model.encode([query])
            _, match_id = vector.search(feature, topk)
            return [raw_data[i] for i in match_id[0] if 0 <= i < len(raw_data)]

    else:
        from es_retrieve import retrieve

    formatted_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")

    question_path = args.question_file or f"../data/eval/{args.dataset}/test.json"
    qa_data = load_questions(question_path, require_answer=args.eval_backend != "none")
    input_question_count = len(qa_data)
    qa_data = limit_questions(qa_data, args.max_questions)
    logger.info("Question file: %d total; this run: %d (max_questions=%s)",
                input_question_count, len(qa_data), args.max_questions)
    if args.eval_backend == "jybigdata":
        from eval_jybigdata import gold_documents, evaluate_predictions
        if not config.get("evaluation", {}).get("base_url") or not config.get("evaluation", {}).get("model"):
            raise ValueError("Configure evaluation.base_url and evaluation.model before running")
        for question in qa_data:
            gold_documents(question)  # Fail before generating answers if references are missing.

    retrieve_method = args.retrieve_method
    if args.dataset in ["asqa", "strategyqa"]:
        retrieve_method = "emb"

    save_path = f"../output/{args.dataset}/{retrieve_method}/{args.method}/{args.model}"
    os.makedirs(save_path, exist_ok=True)

    all_result = []

    if args.resume_path:
        with open(args.resume_path, "r", encoding="utf-8") as fin:
            resume_data = [row for i in fin if i.strip()
                           for row in [json.loads(i)] if "id" in row and "output" in row]
            if [row["id"] for row in resume_data] != list(range(len(resume_data))):
                raise ValueError("Resume predictions must be a contiguous prefix of question positions")
            if len(resume_data) > len(qa_data):
                raise ValueError("Resume file exceeds the selected question limit")
            for row in resume_data:
                source = qa_data[row["id"]]
                if row.get("question", row.get("query")) != source["question"] or row["answer"] != source["answer"]:
                    raise ValueError("Resume predictions do not match the selected question file")
            all_result = resume_data
            filepath = args.resume_path
    else:
        resume_data = []
        filepath = (
            f"{save_path}/topk-{args.retrieve_top_k}-{formatted_time}.jsonl"
            if args.method != "deepnote"
            else f"{save_path}/topk-{args.retrieve_top_k}__max_step-{args.max_step}__max_fail_step-{args.max_fail_step}-{formatted_time}.jsonl"
        )
    logger.info(f"The predicted results will be saved in '{filepath}'.")
    report_path = args.output_path or filepath + ".report.json"
    if args.eval_backend == "none" and Path(report_path).resolve() == Path(question_path).resolve():
        raise ValueError("Output report must not overwrite the question file")
    tracker = PerformanceTracker()
    retrieve = tracker.wrap("retrieval", retrieve)
    call_llm = tracker.wrap("generation", call_llm)
    run_info = {"dataset": args.dataset, "method": args.method, "model": args.model,
                "model_backend": args.model_backend,
                "retrieve_method": retrieve_method, "question_file": str(Path(question_path).resolve()),
                "input_question_count": input_question_count, "max_questions": args.max_questions,
                "question_count": len(qa_data), "max_step": args.max_step,
                "max_fail_step": args.max_fail_step, "max_top_k": args.max_top_k,
                "retrieve_top_k": args.retrieve_top_k, "MaxClients": args.MaxClients}
    last_id = len(resume_data)
    logger.info("Already completed: %d; remaining: %d", last_id, len(qa_data) - last_id)
    batch_size = args.MaxClients
    logger.info("start predicting ...")
    generation_started = time.perf_counter()
    if args.eval_backend == "none":
        save_report(report_path, [], run_info, 0)
    for i in tqdm(range(last_id, len(qa_data), batch_size)):
        pool = ThreadPool(processes=args.MaxClients)
        current_batch = qa_data[i : i + batch_size]
        tasks = [
            (idx + i, doc_cell, args) for idx, doc_cell in enumerate(current_batch)
        ]

        try:
            results = pool.starmap(
                lambda idx, cell, options: tracker.run(process_doc_cell, idx, cell, options), tasks
            )
        finally:
            pool.close()
            pool.join()

        for result in results:
            if result:
                if args.eval_backend == "jybigdata":
                    result["gold_docs"] = gold_documents(qa_data[result["id"]])
                all_result.append(result)
                if args.eval_backend != "none":
                    with open(filepath, "a", buffering=1) as fout:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        if args.eval_backend == "none":
            save_report(report_path, all_result, run_info, time.perf_counter() - generation_started)

    if args.eval_backend == "none":
        summary = save_report(report_path, all_result, run_info,
                              time.perf_counter() - generation_started, complete=True)
        logger.info("Generation report saved to %s; summary: %s", report_path, summary)
        raise SystemExit(0)

    logger.info("Generated %d question results; start evaluating ...", len(all_result))

    if args.eval_backend == "jybigdata":
        eval_result = asyncio.run(evaluate_predictions(
            all_result, config["evaluation"], filepath + ".jybigdata.json", qa_data
        ))
        logger.info("eval result: %s", eval_result)
        raise SystemExit(0)

    predictions = [data["output"] for data in all_result]
    answers = [data["answer"] for data in all_result]

    if "asqa" in args.dataset:
        eval_result = eval_asqa(all_result)
    elif "strategyqa" in args.dataset:
        acc = acc_choice(predictions, answers)
        eval_result = {"Acc": acc}
    else:
        acc = acc_score(predictions, answers)
        f1 = F1_scorer(predictions, answers)
        em = compute_exact(predictions, answers)
        eval_result = {"Acc": acc, "F1": f1, "EM": em}

    if eval_result:
        eval_result["evaluated_questions"] = len(all_result)
        with open(filepath, "a", buffering=1) as fout:
            fout.write(json.dumps(eval_result, ensure_ascii=False) + "\n")

    logger.info(f"eval result: {eval_result}")
