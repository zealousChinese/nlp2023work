# coding: UTF-8

import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from train import train
from config import Config
from preprocess import DataProcessor
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

parser = argparse.ArgumentParser(description="Bert Text Classification")
parser.add_argument("--mode", type=str, help="train/demo/predict", default="train")
parser.add_argument("--data_dir", type=str, default="./data", help="训练数据和模型保存路径")
parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrainedmodel", help="预训练模型路径")
parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
parser.add_argument("--input_file", type=str, default="./data/input.txt", help="预测输入的一整个文件")
args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(args.seed)
    config = Config(args.data_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)
    bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(args.pretrained_bert_dir, "pytorch_model.bin"),
        config=bert_config
    )
    model.to(config.device)

    if args.mode == "train":
        print("loading data...")
        start_time = time.time()
        train_iterator = DataProcessor(config.train_file, config.device, tokenizer, config.batch_size, config.max_seq_len, args.seed)
        dev_iterator = DataProcessor(config.val_file, config.device, tokenizer, config.batch_size, config.max_seq_len, args.seed)

        # train
        train(model, config, train_iterator, dev_iterator)
    else:
        res = []
        model.load_state_dict(torch.load(config.saved_model, map_location=torch.device('cpu')))
        model.eval()

        text = []
        with open(args.input_file, mode="r", encoding="UTF-8") as f:
            for line in tqdm(f):
                sentence = line.strip()
                if not sentence:    continue
                text.append(sentence)

        num_samples = len(text)
        num_batches = (num_samples - 1) // config.batch_size + 1
        for i in range(num_batches):
            start = i * config.batch_size
            end = min(num_samples, (i + 1) * config.batch_size)
            inputs = tokenizer.batch_encode_plus(
                text[start: end],
                padding=True,
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt")
            inputs = inputs.to(config.device)

            outputs = model(**inputs)
            logits = outputs[0]

            preds = torch.max(logits.data, 1)[1].tolist()
            labels = [config.label_list[_] for _ in preds]
            for j in range(start, end):
                res.append(labels[j - start])
                print("%s\t%s" % (text[j], labels[j - start]))


if __name__ == "__main__":
    main()
