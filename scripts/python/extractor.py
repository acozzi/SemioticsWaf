import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import AutoModel

import torch

MODEL = "jackaduma/SecBERT"

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batches(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield i, seq[i:i + batch_size]


def run_extraction(
    input_csv="logs_clean.csv",
    output_parquet="embeddings.parquet",
    checkpoint_dir="embeddings_batches",
    batch_size=64,
    group_size=50,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    model.eval()

    df = pd.read_csv(input_csv)

    # Blindaje contra NaN/valores no-string en payload
    df["payload"] = df["payload"].fillna("").astype(str)

    payloads = df["payload"].tolist()

    batches = list(get_batches(payloads, batch_size))

    for batch_idx, (start, batch_texts) in enumerate(tqdm(batches, desc="Extrayendo embeddings")):

        batch_file = os.path.join(checkpoint_dir, f"batch_{batch_idx:06d}.npy")

        # Si el batch ya fue procesado en una corrida anterior, lo saltamos
        if os.path.exists(batch_file):
            continue

        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)

        cls = outputs.last_hidden_state[:, 0, :]

        # Guardamos el checkpoint de este batch a disco antes de seguir
        np.save(batch_file, cls.cpu().numpy())

    # --- Reensamblado final (streaming, agrupando batches para no generar
    # miles de row groups diminutos en el parquet, que hacen crecer la
    # metadata en memoria sin límite y terminan colgando el proceso) ---
    print("Ensamblando batches en el parquet final (modo streaming agrupado)...")

    labels = df["label"].tolist()
    row_offset = 0
    writer = None

    n_batches = len(batches)

    try:
        for group_start in tqdm(range(0, n_batches, group_size), desc="Escribiendo parquet"):
            group_end = min(group_start + group_size, n_batches)

            arrs = []
            for batch_idx in range(group_start, group_end):
                batch_file = os.path.join(checkpoint_dir, f"batch_{batch_idx:06d}.npy")
                arrs.append(np.load(batch_file))

            arr = np.concatenate(arrs, axis=0)
            n_rows = arr.shape[0]

            group_labels = labels[row_offset:row_offset + n_rows]
            row_offset += n_rows

            group_df = pd.DataFrame(arr)
            group_df["label"] = group_labels

            table = pa.Table.from_pandas(group_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_parquet, table.schema)

            writer.write_table(table)

            # liberamos explícitamente antes del próximo grupo
            del arrs, arr, group_df, table
    finally:
        if writer is not None:
            writer.close()

    print(f"[OK] {row_offset} filas -> {output_parquet}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extracción de embeddings SecBERT con batching y checkpoints")
    parser.add_argument("--input", default="logs_clean.csv")
    parser.add_argument("--output", default="embeddings.parquet")
    parser.add_argument("--checkpoint-dir", default="embeddings_batches")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=50,
                         help="Cuántos batches se agrupan en cada row group del parquet final")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(
        input_csv=args.input,
        output_parquet=args.output,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        group_size=args.group_size,
    )
