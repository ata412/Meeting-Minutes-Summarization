"""
exp03 — exp01 + GEN_K=8 + prompt ให้ใช้คำต้นฉบับ + few-shot (ua047 shot2)
"""
from pathlib import Path
import os
import gc
import re
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from vllm import LLM, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

GEN_K          = 8     # paragraphs fed to LLM for context
POOL_N         = 20    # top-N from each stage-1 retriever before rerank
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
MODEL_NAME     = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B-AWQ")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Few-shot examples from doc_050 (held-out document)
_SHOT1_QUERY = "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด"
_SHOT1_PARAS = [
    "คณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน",
    "ครั้งที่ ๔๙",
    "วันพุธที่ ๑๙ มีนาคม ๒๕๖๘",
    "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
    "เมื่อกรรมาธิการมาครบองค์ประชุมแล้ว ประธานคณะกรรมาธิการได้กล่าวเปิดประชุม และดำเนินการประชุมตามระเบียบวาระการประชุม สรุปสาระสำคัญได้ ดังนี้",
]
_SHOT1_ANSWER = "การประชุมสถาบันการเงินครั้งที่ ๔๙ มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา [อ้างอิง: 4]"

_SHOT2_QUERY = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมาธิการจัดขึ้นเพื่ออะไร"
_SHOT2_PARAS = [
    "เริ่มประชุมเวลา ๐๙.๔๖ นาฬิกา",
    "เมื่อกรรมาธิการมาครบองค์ประชุมแล้ว ประธานคณะกรรมาธิการได้กล่าวเปิดประชุม และดำเนินการประชุมตามระเบียบวาระการประชุม สรุปสาระสำคัญได้ ดังนี้",
    "ระเบียบวาระที่ ๑ เรื่องที่ประธานแจ้งต่อที่ประชุม",
    "สำนักงานเลขาธิการสภาผู้แทนราษฎรขอความอนุเคราะห์ตอบแบบสำรวจความพึงพอใจและความไม่พึงพอใจของคณะกรรมาธิการต่อการบริหารจัดการด้านการประชุม การศึกษาดูงาน และการจัดสัมมนา เพื่อนำผลการประเมินความพึงพอใจและความไม่พึงพอใจมาเป็นข้อมูลในการทบทวน ปรับปรุง และพัฒนาการปฏิบัติงานให้มีประสิทธิภาพต่อไป",
    "ที่ประชุมรับทราบ",
]
_SHOT2_ANSWER = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมการในครั้งนี้ มีการจัดทำขึ้นเพื่อนำข้อมูลที่ได้มาทบทวน ปรับปรุง รวมถึงนำไปพัฒนาการปฏิบัติงานให้มีประสิทธิภาพยิ่งขึ้น [อ้างอิง: 4]"


def benchmark_lib(i):
    os.system(f"{PROGRESS_LIB} {i}")


def load_data(test_dir):
    for name in ("test.json", "sample_test_set.json"):
        p = Path(test_dir) / name
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    raise FileNotFoundError(f"No test.json or sample_test_set.json in {test_dir}")


def filter_valid_paragraphs(paragraphs):
    def is_valid(p):
        text = p["text"].strip()
        return bool(text) and not (set(text) <= set("_-=. \t\n"))
    return [p for p in paragraphs if is_valid(p)]


def tokenize_th(text):
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def encode_texts(model, texts, batch_size=EMBED_BATCH):
    return model.encode(
        texts, batch_size=batch_size, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


def build_prompt(query, retrieved_texts):
    """E5: numbered paragraphs + ask LLM to cite which ones it used."""
    lines = [f"[{i+1}] {text}" for i, text in enumerate(retrieved_texts)]
    context = "\n".join(lines)
    return (
        f"คำถาม: {query}\n\n"
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับ 1-3 ประโยค "
        f"โดยใช้ถ้อยคำจากเอกสารต้นฉบับให้มากที่สุด "
        f"อ้างอิงจากข้อมูลที่ให้มาเท่านั้น ห้ามแต่งเติม "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def parse_citation(text, n_paras):
    """Extract 0-indexed paragraph indices from LLM citation tag."""
    m = re.search(r'\[อ้างอิง[:\s]+([0-9,\s]+)\]', text)
    if m:
        nums = [int(x.strip()) for x in re.findall(r'\d+', m.group(1))]
        valid = [n - 1 for n in nums if 1 <= n <= n_paras]
        if valid:
            return valid
    return [0]  # fallback: top-ranked paragraph


def split_answer_citation(text):
    """Split LLM output into (answer, raw_citation_tag)."""
    idx = text.rfind('[อ้างอิง')
    if idx != -1:
        return text[:idx].strip(), text[idx:]
    return text.strip(), ""


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"{n} queries, {len(doc_index)} docs", flush=True)

    # ── Stage 1a: embed paragraphs + queries, build BM25 ─────────────────
    # spawn (not fork) is used for vLLM workers so GPU use here is safe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    doc_para_data = {}
    for doc_id, paragraphs in doc_index.items():
        valid = filter_valid_paragraphs(paragraphs)
        if valid:
            embs = encode_texts(embed_model, [p["text"] for p in valid])
            bm25 = BM25Okapi([tokenize_th(p["text"]) for p in valid])
        else:
            embs = torch.zeros((0, embed_model.get_sentence_embedding_dimension()))
            bm25 = None
        doc_para_data[doc_id] = (valid, embs, bm25)

    query_embs = encode_texts(embed_model, [q["query"] for q in queries])
    del embed_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Embeddings + BM25 done ({device}).", flush=True)

    # ── Stage 1b: build candidate pools ───────────────────────────────────
    pools = []        # per-query list of para_ids (reranked order, up to GEN_K)
    pair_texts, pair_qidx = [], []
    for i, query in enumerate(queries):
        valid, para_embs, bm25 = doc_para_data.get(
            query["doc_id"], ([], torch.zeros((0, 1)), None))
        if not valid:
            pools.append([])
            continue
        sims = F.cosine_similarity(query_embs[i].unsqueeze(0), para_embs, dim=1)
        dense_idx = torch.topk(sims, k=min(POOL_N, len(valid))).indices.tolist()
        bm25_scores = np.asarray(bm25.get_scores(tokenize_th(query["query"])))
        bm25_idx = np.argsort(-bm25_scores)[:POOL_N].tolist()
        pool_idx = list(dict.fromkeys(dense_idx + bm25_idx))
        pools.append([valid[j]["para_id"] for j in pool_idx])
        for j in pool_idx:
            pair_texts.append((query["query"], valid[j]["text"]))
            pair_qidx.append(i)

    # ── Stage 1c: cross-encoder rerank (GPU) ──────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=device)
    pair_scores = reranker.predict(pair_texts, batch_size=64, show_progress_bar=False)
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Reranked {len(pair_texts)} pairs ({device}).", flush=True)

    scores_by_q = {}
    for qi, s in zip(pair_qidx, pair_scores):
        scores_by_q.setdefault(qi, []).append(float(s))

    # ── assemble top-GEN_K paragraphs per query ────────────────────────────
    items = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        pool_pids = pools[i]
        q_text = query["query"]

        if pool_pids:
            scores = scores_by_q.get(i, [])
            order = sorted(range(len(pool_pids)), key=lambda j: -scores[j])
            top_pids = [pool_pids[j] for j in order[:GEN_K]]
        else:
            top_pids = []

        para_text_map = {p["para_id"]: p["text"]
                         for p in doc_index.get(query["doc_id"], [])}
        retrieved_texts = [para_text_map[pid] for pid in top_pids
                           if para_text_map.get(pid, "").strip()]

        if retrieved_texts:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": build_prompt(_SHOT1_QUERY, _SHOT1_PARAS)},
                {"role": "assistant", "content": _SHOT1_ANSWER},
                {"role": "user", "content": build_prompt(_SHOT2_QUERY, _SHOT2_PARAS)},
                {"role": "assistant", "content": _SHOT2_ANSWER},
                {"role": "user", "content": build_prompt(q_text, retrieved_texts)},
            ]
        else:
            messages = None
        items.append((query["ID"], top_pids, messages, q_text, retrieved_texts))

    # ── Stage 2: vLLM batch generation ────────────────────────────────────
    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=16384,
              gpu_memory_utilization=0.90, dtype="half", enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
                              repetition_penalty=1.05)

    prompts = []
    for it in items:
        msgs = it[2] if it[2] is not None else [{"role": "user", "content": it[3]}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    outputs = llm.generate(prompts, sampling)

    # ── parse answers + citations ──────────────────────────────────────────
    results = []
    for it, out in zip(items, outputs):
        query_id, top_pids, _, q_text, retrieved_texts = it
        raw = out.outputs[0].text.strip()
        answer, _ = split_answer_citation(raw)
        summary = answer or q_text

        # map cited indices → para_ids
        cited_idx = parse_citation(raw, len(retrieved_texts))
        cited_refs = [top_pids[j] for j in cited_idx if j < len(top_pids)]
        if not cited_refs:
            cited_refs = top_pids[:1] if top_pids else []

        results.append({"ID": query_id, "abstractive": summary,
                         "refs": ",".join(cited_refs)})

    out_path = Path(RESULT_DIR) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} rows to {out_path}", flush=True)
    return n


if __name__ == "__main__":
    n = main()
    benchmark_lib(n)
