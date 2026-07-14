"""
ETL para preparar logs.csv antes de pasarlo por SecBERT.

Genera:
  - payload: concatenación de request_http_method + request_http_request
  - label:   nombre de la columna de ataque activa (one-hot -> etiqueta multiclase)

Uso:
    python etl.py                       # logs.csv -> logs_clean.csv
    python etl.py --input otros.csv --output otro_clean.csv
"""

import argparse
import json
import pandas as pd

# Columnas one-hot de labels, tal como aparecen en el CSV original
LABEL_COLUMNS = [
    "000 - Normal",
    "272 - Protocol Manipulation",
    "242 - Code Injection",
    "88 - OS Command Injection",
    "126 - Path Traversal",
    "66 - SQL Injection",
    "16 - Dictionary-based Password Attack",
    "310 - Scanning for Vulnerable Software",
    "153 - Input Data Manipulation",
    "248 - Command Injection",
    "274 - HTTP Verb Tampering",
    "194 - Fake the Source of Data",
    "34 - HTTP Response Splitting",
    "33 - HTTP Request Smuggling",
]


def build_payload(df: pd.DataFrame) -> pd.Series:
    """Concatena method + request en un solo string de texto."""
    method = df["request_http_method"].fillna("").astype(str)
    request = df["request_http_request"].fillna("").astype(str)
    return (method + " " + request).str.strip()


def build_label(df: pd.DataFrame, label_columns=LABEL_COLUMNS) -> pd.Series:
    """
    Convierte las columnas one-hot en una columna 'label' que contiene un
    arreglo (lista) con los nombres de todas las categorías activas (valor 1)
    para esa fila. Puede haber 0, 1 o varios labels activos por fila.

    Como CSV no soporta listas nativamente, cada arreglo se serializa como
    JSON (ej: '["66 - SQL Injection", "248 - Command Injection"]'), y se
    puede deserializar después con json.loads().
    """
    missing = [c for c in label_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas de label en el CSV: {missing}")

    sub = df[label_columns].fillna(0).astype(int)

    n_empty = (sub.sum(axis=1) == 0).sum()
    if n_empty > 0:
        print(f"[WARN] {n_empty} filas no tienen ningún label activo "
              f"(se les asigna la lista ['UNKNOWN']).")

    def collect_labels(row):
        active = [c for c in label_columns if row[c] == 1]
        return active if active else ["UNKNOWN"]

    labels = sub.apply(collect_labels, axis=1)
    return labels.apply(json.dumps)


def run_etl(input_csv: str = "logs.csv", output_csv: str = "logs_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    out = pd.DataFrame()
    out["payload"] = build_payload(df)
    out["label"] = build_label(df)

    out.to_csv(output_csv, index=False)

    print(f"[OK] {len(out)} filas procesadas -> {output_csv}")
    print("\nDistribución de combinaciones de labels (como aparecen serializadas):")
    print(out["label"].value_counts())

    # Distribución por label individual (útil para ver desbalance por clase)
    exploded = out["label"].apply(json.loads).explode()
    print("\nDistribución por label individual:")
    print(exploded.value_counts())

    return out


def parse_args():
    parser = argparse.ArgumentParser(description="ETL de logs.csv para SecBERT")
    parser.add_argument("--input", default="logs.csv", help="CSV de entrada")
    parser.add_argument("--output", default="logs_clean.csv", help="CSV de salida")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_etl(args.input, args.output)