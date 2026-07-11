import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import AutoModel

import torch

MODEL="jackaduma/SecBERT"

device="cuda" if torch.cuda.is_available() else "cpu"

tokenizer=AutoTokenizer.from_pretrained(MODEL)
model=AutoModel.from_pretrained(MODEL).to(device)

df=pd.read_csv("logs.csv")

embeddings=[]

model.eval()

with torch.no_grad():

    for text in tqdm(df["payload"]):

        tokens=tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )

        tokens={k:v.to(device) for k,v in tokens.items()}

        outputs=model(**tokens)

        cls=outputs.last_hidden_state[:,0,:]

        embeddings.append(
            cls.cpu().numpy().flatten()
        )

emb=np.array(embeddings)

out=pd.DataFrame(emb)

out["label"]=df["label"]

out.to_parquet("embeddings.parquet")