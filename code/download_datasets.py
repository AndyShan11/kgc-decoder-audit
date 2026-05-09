"""
Download small KG benchmark datasets (Kinship, UMLS) from public sources.
These are standard in KG completion literature and run very fast for analysis.

Outputs to data/Kinship/ and data/UMLS/ with train.txt, valid.txt, test.txt.
"""
import os, urllib.request, ssl

# Bypass SSL verification if needed
ssl._create_default_https_context = ssl._create_unverified_context

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "..", "data")

DATASETS = {
    # YAGO3-10 - DeepGraphLearning/KnowledgeGraphEmbedding repo has it
    "YAGO3-10": {
        "train.txt": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/train.txt",
        "valid.txt": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/valid.txt",
        "test.txt":  "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/YAGO3-10/test.txt",
    },
    # Kinship (lowercase in many repos)
    "Kinship": {
        "train.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/Kinship/original/train.txt",
        "valid.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/Kinship/original/valid.txt",
        "test.txt":  "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/Kinship/original/test.txt",
    },
    # UMLS
    "UMLS": {
        "train.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/original/train.txt",
        "valid.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/original/valid.txt",
        "test.txt":  "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/original/test.txt",
    },
}

ALT_URLS = {
    # Alternative sources
    "Kinship": {
        "train.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/Kinship/Kinship/train.txt",
        "valid.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/Kinship/Kinship/valid.txt",
        "test.txt":  "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/Kinship/Kinship/test.txt",
    },
    "UMLS": {
        "train.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/UMLS/train.txt",
        "valid.txt": "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/UMLS/valid.txt",
        "test.txt":  "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/UMLS/test.txt",
    },
}


def try_download(url, target):
    try:
        urllib.request.urlretrieve(url, target)
        return True
    except Exception as e:
        print(f"  Failed: {url} ({e})")
        return False


for name, files in DATASETS.items():
    d = os.path.join(DATA_ROOT, name)
    os.makedirs(d, exist_ok=True)
    print(f"\n{name} -> {d}")
    for f, url in files.items():
        target = os.path.join(d, f)
        if os.path.exists(target):
            n_lines = sum(1 for _ in open(target, 'r', encoding='utf-8'))
            print(f"  {f}: already exists ({n_lines} lines)")
            continue

        ok = try_download(url, target)
        if not ok and name in ALT_URLS and f in ALT_URLS[name]:
            print(f"  Trying alternate URL...")
            ok = try_download(ALT_URLS[name][f], target)

        if ok:
            n_lines = sum(1 for _ in open(target, 'r', encoding='utf-8'))
            print(f"  {f}: downloaded ({n_lines} lines)")
        else:
            print(f"  {f}: ALL SOURCES FAILED — you may need to download manually")

print("\nDone. Datasets should be in: " + DATA_ROOT)
