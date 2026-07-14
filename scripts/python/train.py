import gc
import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

from lightgbm import LGBMClassifier
import joblib

MODELS_DIR = "class_models"


def load_and_split_data(parquet_path="embeddings.parquet"):
    print("Cargando parquet...")
    df = pd.read_parquet(parquet_path)

    print("Binarizando etiquetas...")
    y_raw = df.pop("label").apply(json.loads)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_raw)
    del y_raw
    gc.collect()

    num_rows, num_cols = df.shape
    print(f"Dimensiones detectadas: {num_rows} filas, {num_cols} columnas")

    print("Guardando matriz completa temporalmente en disco (X_full.npy)...")
    X_full = np.lib.format.open_memmap(
        'X_full.npy', mode='w+', dtype=np.float32, shape=(num_rows, num_cols))

    X_full[:] = df.to_numpy(dtype=np.float32)
    X_full.flush()
    del df, X_full
    gc.collect()

    print("Generando índices para particiones...")
    indices = np.arange(num_rows)
    train_idx, test_idx, Y_train, Y_test = train_test_split(
        indices, Y, test_size=0.2, random_state=42
    )
    del Y, indices
    gc.collect()

    print("Creando particiones X_train y X_test por lotes (Chunks)...")
    X_full_read = np.load('X_full.npy', mmap_mode='r')

    X_train = np.lib.format.open_memmap(
        'X_train.npy', mode='w+', dtype=np.float32, shape=(
            len(train_idx), num_cols))
    X_test = np.lib.format.open_memmap(
        'X_test.npy', mode='w+', dtype=np.float32, shape=(
            len(test_idx), num_cols))

    chunk_size = 10000

    print("  -> Escribiendo X_train (Copiando sin usar RAM)...")
    for i in range(0, len(train_idx), chunk_size):
        batch_indices = train_idx[i:i+chunk_size]
        X_train[i:i+chunk_size] = X_full_read[batch_indices]
    X_train.flush()

    print("  -> Escribiendo X_test...")
    for i in range(0, len(test_idx), chunk_size):
        batch_indices = test_idx[i:i+chunk_size]
        X_test[i:i+chunk_size] = X_full_read[batch_indices]
    X_test.flush()

    print("Limpiando memoria y eliminando temporales...")
    del X_full_read, X_train, X_test
    gc.collect()
    os.remove('X_full.npy')

    print("Cargando X_train y X_test definitivos vía memory mapping...")
    X_train_final = np.load("X_train.npy", mmap_mode='r')
    X_test_final = np.load("X_test.npy", mmap_mode='r')

    return X_train_final, X_test_final, Y_train, Y_test, mlb


def train_one_class(class_idx, class_name,
                    X_train, y_train_col, X_test, model_path):
    if os.path.exists(model_path):
        print(f"  [skip] {class_name} ya entrenada, se reutiliza {model_path}")
        return joblib.load(model_path)

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        verbosity=-1,
        n_jobs=-1,
        class_weight='balanced'
        # device_type='cuda'
    )
    clf.fit(X_train, y_train_col)

    joblib.dump(clf, model_path)
    return clf


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, Y_train, Y_test, mlb = load_and_split_data()

    print(f"Clases detectadas ({len(mlb.classes_)}): {list(mlb.classes_)}")

    all_preds = np.zeros_like(Y_test)

    for class_idx, class_name in enumerate(mlb.classes_):
        print(f"Entrenando clase {class_idx + 1}/{len(mlb.classes_)}:"
              f" {class_name}")

        model_path = os.path.join(MODELS_DIR, f"clf_{class_idx:02d}.pkl")

        clf = train_one_class(
            class_idx,
            class_name,
            X_train,
            Y_train[:, class_idx],
            X_test,
            model_path,
        )

        all_preds[:, class_idx] = clf.predict(X_test)

        del clf
        gc.collect()

    report_text = classification_report(
        Y_test, all_preds, target_names=mlb.classes_, zero_division=0
    )
    print(report_text)

    report_dict = classification_report(
        Y_test, all_preds, target_names=mlb.classes_,
        zero_division=0, output_dict=True
    )

    df_report = pd.DataFrame(report_dict).transpose()

    csv_path = "metricas_reporte_secbert-lightgbm.csv"
    df_report.to_csv(csv_path, index=True, index_label="clase")

    print(f"[OK] Métricas guardadas exitosamente en '{csv_path}'")

    joblib.dump(mlb, "mlb.pkl")
    print(f"[OK] Modelos por clase '{MODELS_DIR}/' binarizador en 'mlb.pkl'")


if __name__ == "__main__":
    main()
