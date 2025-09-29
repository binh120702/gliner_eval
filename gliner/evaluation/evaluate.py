import glob
import json
import os
import os
import numpy as np
import argparse
import torch
from tqdm import tqdm
import random

def open_content(path):
    paths = glob.glob(os.path.join(path, "*.json"))
    train, dev, test, labels = None, None, None, None
    for p in paths:
        if "train" in p:
            with open(p, "r", encoding='utf-8') as f:
                train = json.load(f)
        elif "dev" in p:
            with open(p, "r", encoding='utf-8') as f:
                dev = json.load(f)
        elif "test" in p:
            with open(p, "r", encoding='utf-8') as f:
                test = json.load(f)
        elif "labels" in p:
            with open(p, "r", encoding='utf-8') as f:
                labels = json.load(f)
    return train, dev, test, labels


def process(data):
    words = data['sentence'].split()
    entities = []  # List of entities (start, end, type)

    for entity in data['entities']:
        start_char, end_char = entity['pos']

        # Initialize variables to keep track of word positions
        start_word = None
        end_word = None

        # Iterate through words and find the word positions
        char_count = 0
        for i, word in enumerate(words):
            word_length = len(word)
            if char_count == start_char:
                start_word = i
            if char_count + word_length == end_char:
                end_word = i
                break
            char_count += word_length + 1  # Add 1 for the space

        # Append the word positions to the list
        entities.append((start_word, end_word, entity['type'].lower()))

    # Create a list of word positions for each entity
    sample = {
        "tokenized_text": words,
        "ner": entities
    }

    return sample


# create dataset
def create_dataset(path):
    train, dev, test, labels = open_content(path)
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for data in train:
        train_dataset.append(process(data))
    for data in dev:
        dev_dataset.append(process(data))
    for data in test:
        test_dataset.append(process(data))
    labels = [label.lower() for label in labels]
    return train_dataset, dev_dataset, test_dataset, labels


@torch.no_grad()
def get_for_one_path(path, model):
    # load the dataset
    _, _, test_dataset, entity_types = create_dataset(path)

    data_name = path.split("/")[-1]  # get the name of the dataset

    # check if the dataset is flat_ner
    flat_ner = True
    if any([i in data_name for i in ["ACE", "GENIA", "Corpus"]]):
        flat_ner = False

    # evaluate the model
    results, f1 = model.evaluate(test_dataset, flat_ner=flat_ner, threshold=0.5, batch_size=12,
                                 entity_types=entity_types, decoding_algo="mwis")
    return data_name, results, f1


def get_for_all_path(model, steps, log_dir, data_paths):
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    # set the model to eval mode
    model.eval()

    # log the results
    save_path = os.path.join(log_dir, "results.txt")

    with open(save_path, "a") as f:
        f.write("##############################################\n")
        # write step
        f.write("step: " + str(steps) + "\n")

    zero_shot_benc = ["mit-movie", "mit-restaurant", "CrossNER_AI", "CrossNER_literature", "CrossNER_music",
                      "CrossNER_politics", "CrossNER_science"]

    zero_shot_benc_results = {}
    all_results = {}  # without crossNER

    for p in tqdm(all_paths):
        if "sample_" not in p:
            data_name, results, f1 = get_for_one_path(p, model)
            # write to file
            with open(save_path, "a") as f:
                f.write(data_name + "\n")
                f.write(str(results) + "\n")

            if data_name in zero_shot_benc:
                zero_shot_benc_results[data_name] = f1
            else:
                all_results[data_name] = f1

    avg_all = sum(all_results.values()) / len(all_results)
    avg_zs = sum(zero_shot_benc_results.values()) / len(zero_shot_benc_results)

    save_path_table = os.path.join(log_dir, "tables.txt")

    # results for all datasets except crossNER
    table_bench_all = ""
    for k, v in all_results.items():
        table_bench_all += f"{k:20}: {v:.1%}\n"
    # (20 size aswell for average i.e. :20)
    table_bench_all += f"{'Average':20}: {avg_all:.1%}"

    # results for zero-shot benchmark
    table_bench_zeroshot = ""
    for k, v in zero_shot_benc_results.items():
        table_bench_zeroshot += f"{k:20}: {v:.1%}\n"
    table_bench_zeroshot += f"{'Average':20}: {avg_zs:.1%}"

    # write to file
    with open(save_path_table, "a") as f:
        f.write("##############################################\n")
        f.write("step: " + str(steps) + "\n")
        f.write("Table for all datasets except crossNER\n")
        f.write(table_bench_all + "\n\n")
        f.write("Table for zero-shot benchmark\n")
        f.write(table_bench_zeroshot + "\n")
        f.write("##############################################\n\n")


def compare_decoding_algorithms(model, steps, log_dir, data_paths, iou_threshold=0.5):
    """
    Run evaluation across all datasets for three decoding algorithms (greedy, nms, mwis),
    and write a side-by-side comparison table to file. Also writes per-algorithm results.
    """
    all_paths = glob.glob(f"{data_paths}/*")
    all_paths = sorted(all_paths)

    # move the model to the device
    device = next(model.parameters()).device
    model.to(device)
    model.eval()

    # prepare log files
    os.makedirs(log_dir, exist_ok=True)
    combined_save_path = os.path.join(log_dir, "tables_compare.txt")

    algo_list = [
        # ("greedy", {"decoding_algo": "greedy"}),
        ("nms", {"decoding_algo": "nms"}),
        ("mwis", {"decoding_algo": "mwis"}),
    ]

    # results structure: {algo: {dataset_name: f1}}
    algo_to_results = {name: {} for name, _ in algo_list}

    # zero-shot grouping (reused from get_for_all_path)
    zero_shot_benc = [
        "mit-movie", "mit-restaurant", "CrossNER_AI", "CrossNER_literature",
        "CrossNER_music", "CrossNER_politics", "CrossNER_science"
    ]

    # per-algorithm raw results files
    per_algo_files = {
        name: os.path.join(log_dir, f"results_{name}.txt") for name, _ in algo_list
    }
    for pth in per_algo_files.values():
        with open(pth, "a") as f:
            f.write("##############################################\n")
            f.write("step: " + str(steps) + "\n")

    for p in tqdm(all_paths):
        if "sample_" in p:
            continue
        # build once per dataset
        _, _, test_dataset, entity_types = create_dataset(p)
        data_name = p.split("/")[-1]

        # detect flat/non-flat
        flat_ner = True
        if any([i in data_name for i in ["ACE", "GENIA", "Corpus"]]):
            flat_ner = False

        for algo_name, algo_kwargs in algo_list:
            results, f1 = model.evaluate(
                test_dataset,
                flat_ner=flat_ner,
                threshold=0.5,
                batch_size=12,
                entity_types=entity_types,
                **algo_kwargs,
            )
            algo_to_results[algo_name][data_name] = f1
            # write per-algo raw
            with open(per_algo_files[algo_name], "a") as f:
                f.write(data_name + "\n")
                f.write(str(results) + "\n")

    # compute averages
    def compute_avgs(results_dict):
        # exclude CrossNER for main average
        plain = {k: v for k, v in results_dict.items() if k not in zero_shot_benc}
        zs = {k: v for k, v in results_dict.items() if k in zero_shot_benc}
        avg_plain = (sum(plain.values()) / len(plain)) if plain else 0.0
        avg_zs = (sum(zs.values()) / len(zs)) if zs else 0.0
        return avg_plain, avg_zs

    greedy_avg_all, greedy_avg_zs = compute_avgs(algo_to_results["greedy"])
    nms_avg_all, nms_avg_zs = compute_avgs(algo_to_results["nms"])
    mwis_avg_all, mwis_avg_zs = compute_avgs(algo_to_results["mwis"])

    # build a union of all dataset names to print rows consistently
    all_dataset_names = set()
    for res in algo_to_results.values():
        all_dataset_names.update(res.keys())
    all_dataset_names = sorted(all_dataset_names)

    # compose comparison table text
    lines = []
    lines.append("##############################################")
    lines.append("step: " + str(steps))
    lines.append("Decoding algorithms comparison (F1)")
    lines.append("")
    header = f"{'Dataset':20} | {'Greedy':>8} | {'NMS':>8} | {'MWIS':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in all_dataset_names:
        g = algo_to_results["greedy"].get(name, float('nan'))
        n = algo_to_results["nms"].get(name, float('nan'))
        m = algo_to_results["mwis"].get(name, float('nan'))
        def fmt(x):
            try:
                return f"{x:.1%}"
            except Exception:
                return "NA"
        lines.append(f"{name:20} | {fmt(g):>8} | {fmt(n):>8} | {fmt(m):>8}")

    lines.append("")
    lines.append("Averages (excluding CrossNER):")
    lines.append(f"{'Average':20} | {greedy_avg_all:.1%:>8} | {nms_avg_all:.1%:>8} | {mwis_avg_all:.1%:>8}")
    lines.append("Zero-shot benchmark averages:")
    lines.append(f"{'Average-ZS':20} | {greedy_avg_zs:.1%:>8} | {nms_avg_zs:.1%:>8} | {mwis_avg_zs:.1%:>8}")
    lines.append("##############################################\n")

    with open(combined_save_path, "a") as f:
        f.write("\n".join(lines) + "\n")


def sample_train_data(data_paths, sample_size=10000):
    all_paths = glob.glob(f"{data_paths}/*")

    all_paths = sorted(all_paths)

    # to exclude the zero-shot benchmark datasets
    zero_shot_benc = ["CrossNER_AI", "CrossNER_literature", "CrossNER_music",
                      "CrossNER_politics", "CrossNER_science", "ACE 2004"]

    new_train = []
    # take 10k samples from each dataset
    for p in tqdm(all_paths):
        if any([i in p for i in zero_shot_benc]):
            continue
        train, dev, test, labels = create_dataset(p)

        # add label key to the train data
        for i in range(len(train)):
            train[i]["label"] = labels

        random.shuffle(train)
        train = train[:sample_size]
        new_train.extend(train)

    return new_train