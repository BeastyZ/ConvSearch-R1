#!/usr/bin/env python3

# This script is adapted from DPR CC-BY-NC 4.0 licensed repo (https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This script is copied from https://github.com/McGill-NLP/topiocqa/blob/main/download_data.py

import argparse
import logging
import os
import pathlib
import wget

from typing import Tuple

logger = logging.getLogger(__name__)

RESOURCES_MAP = {
    "data.topiocqa_dataset.train": {
        "url": "https://zenodo.org/records/7709644/files/topiocqa_train.json",
        "original_ext": ".json",
        "desc": "TopicQA train data",
    },
    "data.topiocqa_dataset.dev": {
        "url": "https://zenodo.org/records/7709644/files/topiocqa_dev.json",
        "original_ext": ".json",
        "desc": "TopicQA development data",
    },
    "data.wikipedia_split.full_wiki": {
        "url": "https://zenodo.org/records/6173228/files/data/wikipedia_split/full_wiki.jsonl",
        "original_ext": ".jsonl",
        "desc": "Entire wikipedia corpus",
    },
    "data.wikipedia_split.full_wiki_segments": {
        "url": "https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv",
        "original_ext": ".tsv",
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 200-word segments (no overlap)"
    },
    "data.gold_passages_info.all_history.train": {
        "url": "https://zenodo.org/records/6151011/files/data/gold_passages_info/all_history/train.json",
        "original_ext": ".json",
        "desc": "Gold passages info for all_history train passages",
    },
    "data.gold_passages_info.all_history.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/gold_passages_info/all_history/dev.json",
        "original_ext": ".json",
        "desc": "Gold passages info for all_history dev passages",
    },
    "data.gold_passages_info.original.train": {
        "url": "https://zenodo.org/records/6151011/files/data/gold_passages_info/original/train.json",
        "original_ext": ".json",
        "desc": "Gold passages info for train examples of original variant",
    },
    "data.gold_passages_info.original.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/gold_passages_info/original/dev.json",
        "original_ext": ".json",
        "desc": "Gold passages info for dev examples of original variant",
    },
    "data.gold_passages_info.rewrites_t5_qrecc.train": {
        "url": "https://zenodo.org/records/6151011/files/data/gold_passages_info/rewrites_t5_qrecc/train.json",
        "original_ext": ".json",
        "desc": "Gold passages info for train examples of rewrites variant",
    },
    "data.gold_passages_info.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/gold_passages_info/rewrites_t5_qrecc/dev.json",
        "original_ext": ".json",
        "desc": "Gold passages info for dev examples of rewrites variant",
    },
    "data.retriever.all_history.train": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/all_history/train.json",
        "original_ext": ".json",
        "desc": "Retriever training data for all_history train passages",
    },
    "data.retriever.all_history.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/all_history/dev.json",
        "original_ext": ".json",
        "desc": "Retriever training data for all_history dev passages",
    },
    "data.retriever.original.train": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/original/train.json",
        "original_ext": ".json",
        "desc": "Retriever training data for original train passages",
    },
    "data.retriever.original.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/original/dev.json",
        "original_ext": ".json",
        "desc": "Retriever training data for original dev passages",
    },
    "data.retriever.rewrites_t5_qrecc.train": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/rewrites_t5_qrecc/train.json",
        "original_ext": ".json",
        "desc": "Retriever training data for rewrites train passages",
    },
    "data.retriever.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/rewrites_t5_qrecc/dev.json",
        "original_ext": ".json",
        "desc": "Retriever training data for rewrites dev passages",
    },
    "data.retriever.qas.all_history.train": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/qas/all_history/train.csv",
        "original_ext": ".csv",
        "desc": "Retriever training data for all_history train passages (CSV format)",
    },
    "data.retriever.qas.all_history.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/qas/all_history/dev.csv",
        "original_ext": ".csv",
        "desc": "Retriever training data for all_history dev passages (CSV format)",
    },
    "data.retriever.qas.original.train": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/qas/original/train.csv",
        "original_ext": ".csv",
        "desc": "Retriever training data for original train passages (CSV format)",
    },
    "data.retriever.qas.original.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/qas/original/dev.csv",
        "original_ext": ".csv",
        "desc": "Retriever training data for original dev passages (CSV format)",
    },
    "data.retriever.qas.rewrites_t5_qrecc.train": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/qas/rewrites_t5_qrecc/train.csv",
        "original_ext": ".csv",
        "desc": "Retriever training data for rewrites train passages (CSV format)",
    },
    "data.retriever.qas.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6151011/files/data/retriever/qas/rewrites_t5_qrecc/dev.csv",
        "original_ext": ".csv",
        "desc": "Retriever dev data for rewrites variant in csv format",
    },
    "passage_embeddings.all_history.wikipedia_passages": {
        "url": [
            "https://zenodo.org/records/6153453/files/passage_embeddings/all_history/wikipedia_passages_{}.pkl".format(i)
            for i in range(25)
        ] + [
            "https://zenodo.org/records/6153959/files/passage_embeddings/all_history/wikipedia_passages_{}.pkl".format(i)
            for i in range(25, 50)
        ],
        "original_ext": ".pkl",
        "desc": "Wikipedia passage embeddings for all_history passages",
    },
    "passage_embeddings.original.wikipedia_passages": {
        "url": [
            "https://zenodo.org/records/6157968/files/passage_embeddings/original/wikipedia_passages_{}.pkl".format(i)
            for i in range(25)
        ] + [
            "https://zenodo.org/records/6158757/files/passage_embeddings/original/wikipedia_passages_{}.pkl".format(i)
            for i in range(25, 50)
        ],
        "original_ext": ".pkl",
        "desc": "Wikipedia passage embeddings for original passages",
    },
    "passage_embeddings.rewrites_t5_qrecc.wikipedia_passages": {
        "url": [
            "https://zenodo.org/records/6154952/files/passage_embeddings/rewrites_t5_qrecc/wikipedia_passages_{}.pkl".format(i)
            for i in range(25)
        ] + [
            "https://zenodo.org/records/6156282/files/passage_embeddings/rewrites_t5_qrecc/wikipedia_passages_{}.pkl".format(i)
            for i in range(25, 50)
        ],
        "original_ext": ".pkl",
        "desc": "Wikipedia passage embeddings for rewrites passages",
    },
    "model_checkpoints.retriever.dpr.all_history.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/retriever/dpr_retriever_all_history",
        "original_ext": ".pt",
        "desc": "Retriever DPR model checkpoint for all_history passages",
    },
    "model_checkpoints.retriever.dpr.original.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/retriever/dpr_retriever_original",
        "original_ext": ".pt",
        "desc": "Retriever DPR model checkpoint for original passages",
    },
    "model_checkpoints.retriever.dpr.rewrites_t5_qrecc.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/retriever/dpr_retriever_rewrites_t5_qrecc",
        "original_ext": ".pt",
        "desc": "Retriever DPR model checkpoint for rewrites passages",
    },
    "model_checkpoints.reader.dpr_reader.dpr_retriever.all_history.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/dpr_reader/dpr_retriever/all_history",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoint for all_history passages, trained on the Retriever DPR results",
    },
    "model_checkpoints.reader.dpr_reader.dpr_retriever.original.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/dpr_reader/dpr_retriever/original",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoint for original passages, trained on the Retriever DPR results",
    },
    "model_checkpoints.reader.dpr_reader.dpr_retriever.rewrites_t5_qrecc.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/dpr_reader/dpr_retriever/rewrites_t5_qrecc",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoint for rewrites passages, trained on the Retriever DPR results",
    },
    "model_checkpoints.reader.dpr_reader.bm25_retriever.all_history.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/dpr_reader/bm25_retriever/all_history",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoint for all_history passages, trained on the BM25 Retriever results",
    },
    "model_checkpoints.reader.dpr_reader.bm25_retriever.original.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/dpr_reader/bm25_retriever/original",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoint for original passages, trained on the BM25 Retriever results",
    },
    "model_checkpoints.reader.dpr_reader.bm25_retriever.rewrites_t5_qrecc.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/dpr_reader/bm25_retriever/rewrites_t5_qrecc",
        "original_ext": ".pt",
        "desc": "Reader DPR model checkpoint for rewrites passages, trained on the BM25 Retriever results",
    },
    "model_checkpoints.reader.fid.dpr_retriever.all_history.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/fid/dpr_retriever/all_history",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for all_history passages, trained on the Retriever DPR results",
    },
    "model_checkpoints.reader.fid.dpr_retriever.original.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/fid/dpr_retriever/original",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for original passages, trained on the Retriever DPR results",
    },
    "model_checkpoints.reader.fid.dpr_retriever.rewrites_t5_qrecc.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/fid/dpr_retriever/rewrites_t5_qrecc",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for rewrites passages, trained on the Retriever DPR results",
    },
    "model_checkpoints.reader.fid.bm25_retriever.all_history.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/fid/bm25_retriever/all_history",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for all_history passages, trained on the BM25 Retriever results",
    },
    "model_checkpoints.reader.fid.bm25_retriever.original.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/fid/bm25_retriever/original",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for original passages, trained on the BM25 Retriever results",
    },
    "model_checkpoints.reader.fid.bm25_retriever.rewrites_t5_qrecc.checkpoint": {
        "url": "https://zenodo.org/records/6151076/files/model_checkpoints/reader/fid/bm25_retriever/rewrites_t5_qrecc",
        "original_ext": ".zip",
        "desc": "FiD model checkpoint for rewrites variant with BM25 Retriever",
    },
    "results.retriever.dpr.all_history.train": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_all_history_train.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for all_history variant on train set",
    },
    "results.retriever.dpr.all_history.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for all_history variant on dev set",
    },
    "results.retriever.dpr.original.train": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_original_train.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for original variant on train set",
    },
    "results.retriever.dpr.original.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for original variant on dev set",
    },
    "results.retriever.dpr.rewrites_t5_qrecc.train": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_rewrites_t5_qrecc_train.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for rewrites variant on train set",
    },
    "results.retriever.dpr.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/dpr/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Retriever DPR results for rewrites variant on dev set",
    },
    "results.retriever.bm25.all_history.train": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/bm25/topiocqa_all_history_train.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for all_history variant on train set",
    },
    "results.retriever.bm25.all_history.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/bm25/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for all_history variant on dev set",
    },
    "results.retriever.bm25.original.train": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/bm25/topiocqa_original_train.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for original variant on train set",
    },
    "results.retriever.bm25.original.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/bm25/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for original variant on dev set",
    },
    "results.retriever.bm25.rewrites_t5_qrecc.train": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/bm25/topiocqa_rewrites_t5_qrecc_train.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for rewrites variant on train set",
    },
    "results.retriever.bm25.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/retriever/bm25/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Retriever BM25 results for rewrites variant on dev set",
    },
    "results.reader.dpr_reader.dpr_retriever.all_history.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/dpr_reader/dpr_retriever/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "DPR Reader results using DPR Retriever results for original variant on dev set",
    },
    "results.reader.dpr_reader.dpr_retriever.original.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/dpr_reader/dpr_retriever/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "DPR Reader results using BM25 Retriever results for all history variant on dev set",
    },
    "results.reader.dpr_reader.dpr_retriever.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/dpr_reader/dpr_retriever/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "DPR Reader results using BM25 Retriever results for rewrites variant on dev set",
    },
    "results.reader.dpr_reader.bm25_retriever.all_history.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/dpr_reader/bm25_retriever/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "DPR Reader results using BM25 Retriever results for all_history variant on dev set",
    },
    "results.reader.dpr_reader.bm25_retriever.original.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/dpr_reader/bm25_retriever/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "DPR Reader results using BM25 Retriever results for original variant on dev set",
    },
    "results.reader.dpr_reader.bm25_retriever.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/dpr_reader/bm25_retriever/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "DPR Reader results using BM25 Retriever results for rewrites variant on dev set",
    },
    "results.reader.fid.dpr_retriever.all_history.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/fid/dpr_retriever/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Reader FID results using DPR Retriever results for all_history variant on dev set",
    },
    "results.reader.fid.dpr_retriever.original.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/fid/dpr_retriever/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Reader FID results using DPR Retriever results for original variant on dev set",
    },
    "results.reader.fid.dpr_retriever.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/fid/dpr_retriever/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Reader FID results using DPR Retriever results for rewrites variant on dev set",
    },
    "results.reader.fid.bm25_retriever.all_history.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/fid/bm25_retriever/topiocqa_all_history_dev.json",
        "original_ext": ".json",
        "desc": "Reader FID results using BM25 Retriever results for all_history variant on dev set",
    },
    "results.reader.fid.bm25_retriever.original.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/fid/bm25_retriever/topiocqa_original_dev.json",
        "original_ext": ".json",
        "desc": "Reader FID results using BM25 Retriever results for original variant on dev set",
    },
    "results.reader.fid.bm25_retriever.rewrites_t5_qrecc.dev": {
        "url": "https://zenodo.org/records/6153434/files/results/reader/fid/bm25_retriever/topiocqa_rewrites_t5_qrecc_dev.json",
        "original_ext": ".json",
        "desc": "Reader FID results using BM25 Retriever results for rewrites variant on dev set",
    },
}


def download_resource(
    url: str, original_ext: str, resource_key: str, out_dir: str
) -> Tuple[str, str]:
    logger.info("Requested resource from %s", url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    logger.info("Download root_dir %s", root_dir)

    save_root = os.path.join(root_dir, "downloads", *path_names[:-1]) 

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(os.path.join(save_root, path_names[-1] + original_ext))
    logger.info("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        logger.info("File already exist %s", local_file_uncompressed)
        return save_root, local_file_uncompressed

    local_file = os.path.abspath(os.path.join(save_root, path_names[-1] +  original_ext))

    wget.download(url, out=local_file)

    logger.info("Downloaded to %s", local_file)

    return save_root, local_file


def download_file(url: str, out_dir: str, file_name: str):
    logger.info("Loading from %s", url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        logger.info("File already exist %s", local_file)
        return

    wget.download(url, out=local_file)
    logger.info("Downloaded to %s", local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            logger.info("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    url = download_info["url"]

    save_root_dir = None
    data_files = []
    if isinstance(url, list):
        for i, item_url in enumerate(url):
            save_root_dir, local_file = download_resource(
                item_url,
                download_info["original_ext"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
            data_files.append(local_file)
    else:
        save_root_dir, local_file = download_resource(
            url,
            download_info["original_ext"],
            resource_key,
            out_dir,
        )
        data_files.append(local_file)

    return data_files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print(f"Resource key={k}  :  {v['desc']}")


if __name__ == "__main__":
    main()
