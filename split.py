from sklearn.model_selection import train_test_split
import pandas as pd
import csv
# f = open('train_data.txt',encoding='utf-8')
dataset = pd.read_csv('information.txt',sep="\t",header=None,names=["text","label"],encoding="utf-8",engine='python')
# dataset.to_csv('test_data_1.txt',sep='\t',header=False,index=False)

# print(dataset)
train_set, x = train_test_split(dataset,
		stratify=dataset['label'],
		test_size=0.1,
		random_state=42)


train_set.to_csv('secondtrain.txt',sep='\t',header=False,index=False)
x.to_csv('secondval.txt',sep='\t',header=False,index=False)
