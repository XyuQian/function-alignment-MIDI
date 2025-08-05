import torch
import matplotlib.pyplot as plt
# import seaborn as sns
import math

# --- Copy the get_pe function from cross_attention.py ---
def get_pe(d_model, max_len=10000):
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# --- Verification ---
d_model = 128  # A smaller dimension for easier visualization
seq_len = 100

pe_matrix = get_pe(d_model, max_len=seq_len)

# Plotting the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(pe_matrix, cmap='viridis')
plt.title("Positional Encoding Heatmap")
plt.xlabel("Embedding Dimension")
plt.ylabel("Sequence Position")
plt.show()