import pandas as pd

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from sklearn.metrics import classification_report

import joblib

df=pd.read_parquet("embeddings.parquet")

X=df.drop(columns=["label"])

y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf=LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63
)

clf.fit(X_train,y_train)

pred=clf.predict(X_test)

print(classification_report(y_test,pred))

joblib.dump(clf,"lightgbm.pkl")