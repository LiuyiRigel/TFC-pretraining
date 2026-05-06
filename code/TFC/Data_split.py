"""本文件旨在生成不同的数据划分，以评估模型的稳定性。
在每个划分中，我们选择 30 个正样本和 30 个负样本组成训练集（共 60 个样本）。
这是针对 Epilepsy 数据集的示例。-- Xiang Zhang, 2023 年 1 月 16 日"""

import torch
import os
import numpy as np

targetdata_path = f"../../datasets/Epilepsy/"

finetune_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
train_data = torch.load(os.path.join(targetdata_path, "train.pt"))
Samples = torch.cat((finetune_dataset['samples'], train_data['samples']), dim=0)
Labels = torch.cat((finetune_dataset['labels'], train_data['labels']), dim=0)

train_size = 30

"""生成平衡训练集"""

id0 = Labels==0
id1 = Labels==1

Samples_0, Samples_1 = Samples[id0], Samples[id1]
Labels_0, Labels_1 = Labels[id0], Labels[id1]

Samples_train = torch.cat((Samples_0[:train_size], Samples_1[:train_size]))
Samples_test = torch.cat((Samples_0[train_size:], Samples_1[train_size:]))

Labels_train = torch.cat((Labels_0[:train_size], Labels_1[:train_size]))
Labels_test = torch.cat((Labels_0[train_size:], Labels_1[train_size:]))

# """生成不平衡训练集"""
# data = list(zip(Samples, Labels))
# np.random.shuffle(data)
# X, y = zip(*data)
# X = torch.stack(list(X), dim=0)
# y = torch.stack(list(y), dim=0)
# Samples_train, Samples_test = X[:train_size], X[train_size:]
# Labels_train, Labels_test = y[:train_size], y[train_size:]

train_dic = {'samples':Samples_train, 'labels':Labels_train}
test_dic = {'samples':Samples_test, 'labels':Labels_test}
torch.save(train_dic, os.path.join(targetdata_path,'train.pt'))
torch.save(test_dic, os.path.join(targetdata_path,'test.pt'))

print('Re-split finished. Dataset saved to folder:', targetdata_path)
