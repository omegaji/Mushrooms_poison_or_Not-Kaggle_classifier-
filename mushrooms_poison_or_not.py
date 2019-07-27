
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
df=pd.read_csv("../input/mushrooms.csv")



from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X=df.drop("class",axis=1)
y=df["class"]
for i in X.columns:
    X[i]=le.fit_transform(X[i])
y=le.fit_transform(y)
X=pd.get_dummies(X,columns=X.columns,drop_first=True)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pca = PCA(n_components=6)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
clf.score(X_test,y_test)

