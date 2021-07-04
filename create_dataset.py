import os
import pandas as pd
import numpy as np
import joblib

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

#get the images paths 
image_paths = list(paths.list_images('/Users/trinhvo/Documents/Data/WikiArt'))

#creating an empty dataframe
df = pd.DataFrame()

labels = []
for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
    label = image_path.split(os.path.sep)[-2]
    df.loc[i,'image_path'] = image_path
    labels.append(label)

labels = np.array(labels)
lb = LabelBinarizer()
print(f"The length of image paths: {len(image_paths)}")
print(f"Total instances: {len(labels)}")
labels = lb.fit_transform(labels)


for i in range(len(labels)):
    index = np.argmax(labels[i])
    df.loc[i, 'target'] = int(index)

df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('../data.csv', index=False)


joblib.dump(lb, '../lb.pkl')



