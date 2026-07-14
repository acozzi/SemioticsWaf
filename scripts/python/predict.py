import joblib
import torch

from transformers import AutoTokenizer
from transformers import AutoModel

MODEL = "jackaduma/SecBERT"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)

clf = joblib.load("lightgbm.pkl")
mlb = joblib.load("mlb.pkl")

payload = "GET /?id=1 UNION SELECT password FROM users"

tokens = tokenizer(
    payload,
    return_tensors="pt",
    truncation=True,
    max_length=128,
    padding="max_length"
)

tokens = {k: v.to(device) for k, v in tokens.items()}

with torch.no_grad():
    outputs = model(**tokens)

embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

prediction = clf.predict(embedding)

# Convierte la matriz binaria de vuelta a nombres de labels
labels = mlb.inverse_transform(prediction)

print(labels[0] if labels else ())