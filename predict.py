# import tkinter as tk
# from config import Config
# from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
# import os
# import torch
# import argparse
#
# parser = argparse.ArgumentParser(description="Bert Chinese Text Classification")
# parser.add_argument("--data_dir", type=str, default="./data", help="training data and saved model path")
# parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrainedmodel", help="pretrained bert model path")
# args = parser.parse_args()
#
# # 创建一个窗口对象
# window = tk.Tk()
# # 设置一下窗口标题
# window.title("标题分类")
# # 设置窗口的大小
# window.geometry("450x260")
#
#
# e = tk.Entry(window, width=35,font=("",15))
#
# e.pack()
#
# def insert_point():
#     # 获取输入文本对象输入的内容
#     var = e.get()
#     print(var)
#     config = Config(args.data_dir)
#
#     tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)
#     bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
#     model = BertForSequenceClassification.from_pretrained(
#         os.path.join(args.pretrained_bert_dir, "pytorch_model.bin"),
#         config=bert_config
#     )
#     model.to(config.device)
#     model.load_state_dict(torch.load(config.saved_model,map_location='cpu'))
#     model.eval()
#     inputs = tokenizer(
#         var,
#         max_length=config.max_seq_len,
#         truncation="longest_first",
#         return_tensors="pt")
#     inputs = inputs.to(config.device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs[0]
#         label = torch.max(logits.data, 1)[1].tolist()
#         print("分类结果:" + config.label_list[label[0]])
#     t.insert("insert",config.label_list[label[0]])
#
#
# b = tk.Button(window, text="预测类别 ", width=15, height=2,
#               command=insert_point)
# b.pack()
#
# t = tk.Text(window, width=20,height=5,font=("",20))
# t.pack()
#
# window.mainloop()
