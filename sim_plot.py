# paste this into a local Jupyter notebook or a .py file and run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Layers and numeric values from your message
layers = [5, 7, 10, 18, 19]
labels = [str(l) for l in layers]

pairs = {
    (5,7): 0.8046875,
    (10,5): 0.640625,
    (10,7): 0.75390625,
    (18,5): 0.4375,
    (18,7): 0.515625,
    (18,10): 0.60546875,
    (19,5): 0.416015625,
    (19,7): 0.4921875,
    (19,10): 0.58203125,
    (19,18): 0.91796875
}

magnitudes = {5:2.59375, 7:4.125, 10:6.21875, 18:10.8125, 19:13.5625}

# Build symmetric cosine-similarity matrix
n = len(layers)
mat = np.eye(n)
index_of = {layers[i]: i for i in range(n)}
for (a,b), val in pairs.items():
    i = index_of[a]; j = index_of[b]
    mat[i,j] = float(val); mat[j,i] = float(val)

# DataFrame for numeric table
df = pd.DataFrame(mat, index=labels, columns=labels)
print("Cosine similarity matrix:")
print(df)

# Save CSV for the report
df.to_csv("cosine_similarity_matrix.csv")
print("Saved numeric CSV: cosine_similarity_matrix.csv")

# ---- Heatmap (cosine similarity) ----
plt.figure(figsize=(6,5))
plt.imshow(mat, vmin=-1, vmax=1)   # default colormap
plt.colorbar()
plt.xticks(range(n), labels)
plt.yticks(range(n), labels)
plt.xlabel("Layer")
plt.ylabel("Layer")
plt.title("Cosine similarity between 'evil' mean-diff vectors (layers)")
plt.tight_layout()
plt.savefig("cosine_similarity_heatmap.png", dpi=200)
plt.show()
plt.close()
print("Saved heatmap: cosine_similarity_heatmap.png")

# ---- Bar chart (magnitudes) ----
plt.figure(figsize=(6,4))
plt.bar(labels, [magnitudes[l] for l in layers])
plt.xlabel("Layer")
plt.ylabel("Magnitude")
plt.title("Layer magnitudes (||mean-diff||)")
plt.tight_layout()
plt.savefig("layer_magnitudes.png", dpi=200)
plt.show()
plt.close()
print("Saved bar chart: layer_magnitudes.png")
