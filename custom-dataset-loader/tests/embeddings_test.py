import math

import torch


emb = torch.load("data/embeddings/page_embeddings.pt")

print(f'Embeddings shape: {emb["embeddings"].shape}')
print(f'Embeddings length: {len(emb["page_ids"])}')
print(f'Page IDs: {emb["page_ids"][:10]}')

print(f'Any NaN in embeddings: {torch.isnan(emb["embeddings"]).any()}')

# page_ids is saved as a Python list of ints (not a tensor), so torch.isnan() is not applicable.
page_ids = emb["page_ids"]
print(f"Page IDs are all ints: {all(isinstance(x, int) for x in page_ids)}")

# model_name is a string; NaN checks are not applicable.
model_name = emb["model_name"]
print(f"Model name present: {isinstance(model_name, str) and len(model_name) > 0}")

# avg_text_length_words is a float; check finiteness.
avg_len = float(emb["avg_text_length_words"])
print(f"Avg text length words is finite: {math.isfinite(avg_len)}")

print(f'Mean embedding norm: {emb["embeddings"].norm(dim=1).mean()}')

