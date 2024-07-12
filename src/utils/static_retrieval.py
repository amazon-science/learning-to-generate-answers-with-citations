"""
Based on https://github.com/princeton-nlp/ALCE/blob/main/retrieval.py
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

TOPK = 100


def bm25_sphere_retrieval(data, index_path):
    from pyserini.search import LuceneSearcher

    print("loading bm25 index, this may take a while...")
    searcher = LuceneSearcher(index_path)

    print("running bm25 retrieval...")
    for d in tqdm(data):
        query = d["question"]
        try:
            hits = searcher.search(query, TOPK)
        except Exception as e:
            # https://github.com/castorini/pyserini/blob/1bc0bc11da919c20b4738fccc020eee1704369eb/scripts/kilt/anserini_retriever.py#L100
            if "maxClauseCount" in str(e):
                query = " ".join(query.split())[:950]
                hits = searcher.search(query, TOPK)
            else:
                raise e

        docs = []
        for hit in hits:
            h = json.loads(str(hit.docid).strip())
            docs.append(
                {
                    "title": h["title"],
                    "text": hit.raw,
                    "url": h["url"],
                }
            )
        d["docs"] = docs

    return data


def gtr_build_index(encoder, docs, index_path):
    with torch.inference_mode():
        embs = encoder.encode(docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        embs = embs.astype("float16")

    with open(index_path, "wb") as f:
        pickle.dump(embs, f)
    return embs


def gtr_wiki_retrieval(data, index_path, passages_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading GTR encoder...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device=device)

    questions = [d["question"] for d in data]
    with torch.inference_mode():
        queries = encoder.encode(questions, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        queries = torch.tensor(queries, dtype=torch.float32, device="cpu")

    # the wikipedia split from DPR repo: https://github.com/facebookresearch/DPR
    docs = []
    print("loading wikipedia file...")
    with open(passages_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            docs.append(row[2] + "\n" + row[1])

    if not os.path.exists(index_path):
        print("gtr embeddings not found, building...")
        embs = gtr_build_index(encoder, docs, index_path)
    else:
        print("gtr embeddings found, loading...")
        with open(index_path, "rb") as f:
            embs = pickle.load(f)

    del encoder  # save gpu mem

    gtr_emb = torch.tensor(embs, dtype=torch.float32, device="cpu")

    print("running GTR retrieval...")
    for qi, q in enumerate(tqdm(queries)):
        q = q.to("cpu")
        scores = torch.matmul(gtr_emb, q)
        score, idx = torch.topk(scores, TOPK)
        ret = []
        for i in range(idx.size(0)):
            title, text = docs[idx[i].item()].split("\n")
            ret.append({"id": str(idx[i].item() + 1), "title": title, "text": text, "score": score[i].item()})
        print(ret)
        data[qi]["docs"] = ret
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--data_file", type=str, default=None, help="path to the data file")
    parser.add_argument("--passages_path", type=str, default=None, help="path to the data file")
    parser.add_argument("--index_path", type=str, default=None, help="path to the data file")
    parser.add_argument(
        "--output_file", type=str, default=None, help="same format as the data file but with the retrieved docs."
    )
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--retriever", default="gtr", type=str, choices=["gtr", "bm25"])
    args = parser.parse_args()

    with open(args.data_file) as f:
        lines = f.readlines()
        data = []
        for line in lines:
            sample = json.loads(line)
            data.append(sample)

    # with open(args.data_file) as f:
    #     data = json.load(f)[args.start_index:10000] # Only index 10000 samples at a time

    if args.retriever == "gtr":
        data = gtr_wiki_retrieval(data, args.index_path, args.passages_path)
    elif args.retriever == "bm25":
        data = bm25_sphere_retrieval(data, args.index_path)
    else:
        print("Retriever method not known, aborting...")
        sys.exit()

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
