import json

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from lightgbm import LGBMClassifier

import joblib

df = pd.read_parquet("embeddings.parquet")

X = df.drop(columns=["label"])

# "label" viene serializado como JSON (ej: '["66 - SQL Injection", "248 - Command Injection"]')
y_raw = df["label"].apply(json.loads)

# Convierte la lista de labels por fila en una matriz binaria (una columna por clase)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_raw)

print(f"Clases detectadas ({len(mlb.classes_)}): {list(mlb.classes_)}")

# Nota: train_test_split no soporta stratify nativo para multi-label.
# Si el desbalance es fuerte, considerar iterative-stratification
# (paquete: scikit-multilearn / iterstrat).
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

base_clf = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63
)

# Entrena un clasificador binario por clase (uno-vs-el-resto)
clf = OneVsRestClassifier(base_clf)
clf.fit(X_train, Y_train)

pred = clf.predict(X_test)

print(classification_report(Y_test, pred, target_names=mlb.classes_, zero_division=0))

joblib.dump(clf, "lightgbm.pkl")
joblib.dump(mlb, "mlb.pkl")