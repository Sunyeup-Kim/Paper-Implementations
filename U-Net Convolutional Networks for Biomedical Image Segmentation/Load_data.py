import pandas as pd
import numpy as np
import os

df = pd.read_csv(r'D:\Brain_MRI(Kaggle)\kaggle_3m\data.csv')
df.head()


path = r'D:\Brain_MRI(Kaggle)\kaggle_3m'
data_list = os.listdir(path)
data_list = data_list[3:] # csv와 메모 파일 버리기
print(len(data_list))


data_tmp = []

for path2 in data_list:
    path_tmp = os.path.join(path, path2)

    for tmp in os.listdir(path_tmp):
        data_tmp.append(path2)
        data_tmp.append(os.path.join(path_tmp, tmp))


filenames = data_tmp[::2]
masks = data_tmp[1::2]

df = pd.DataFrame(data={"patient_id": filenames,"img_path": masks})
print(df.shape)

df.to_csv(r'D:\Brain_MRI(Kaggle)\kaggle_3m\image_path.csv', index=False)