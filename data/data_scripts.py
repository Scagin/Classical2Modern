
import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='./CCTC', help="raw dataset directory")
args = parser.parse_args()

files = os.listdir(args.datadir)

source_sentences = []
target_sentences = []

for file in files:
    if not os.path.isdir(file) or file == "static":
        continue

    jsonfile_list = os.listdir(os.path.join(args.datadir, file))
    for json_file in jsonfile_list:
        with open(os.path.join(args.datadir, file, json_file), "r", encoding="utf-8") as f:
            json_data = json.load(f)

        for article in json_data:
            pairs = article.get("contents", [])

            for pair in pairs:
                source_sentences.append(pair.get("source"))
                target_sentences.append(pair.get("target"))

assert len(source_sentences) == len(target_sentences)

datas_length = len(source_sentences)

indices = np.arange(datas_length)
random_ind = np.random.permutation(indices)
source_sentences = np.array(source_sentences)[random_ind].tolist()
target_sentences = np.array(target_sentences)[random_ind].tolist()
train_source, dev_source, test_source = source_sentences[:-1000], source_sentences[-1000:], source_sentences[-1000:]
train_target, dev_target, test_target = target_sentences[:-1000], target_sentences[-1000:], target_sentences[-1000:]

with open("train.src", "w", encoding="utf-8") as src_f, \
    open("train.dst", "w", encoding="utf-8") as dst_f:
    src_f.write("\n".join(train_source))
    dst_f.write("\n".join(train_target))

with open("eval.src", "w", encoding="utf-8") as src_f, \
    open("eval.dst", "w", encoding="utf-8") as dst_f:
    src_f.write("\n".join(dev_source))
    dst_f.write("\n".join(dev_target))

with open("test.src", "w", encoding="utf-8") as src_f, \
    open("test.dst", "w", encoding="utf-8") as dst_f:
    src_f.write("\n".join(test_source))
    dst_f.write("\n".join(test_target))


