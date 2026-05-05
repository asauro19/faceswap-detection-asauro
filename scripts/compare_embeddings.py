import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



# 1. Load embeddings + labels

rgb_feats = np.load("rgb_features.npy")
fft_feats = np.load("fft_features.npy")

rgb_labels = np.load("rgb_feature_labels.npy")
fft_labels = np.load("fft_feature_labels.npy")

assert np.array_equal(rgb_labels, fft_labels), "Label mismatch between RGB and FFT!"



# 2. PCA to 2D for visualization

pca = PCA(n_components=2)

rgb_2d = pca.fit_transform(rgb_feats)
fft_2d = pca.fit_transform(fft_feats)



# 3. Plot RGB vs FFT embeddings

plt.figure(figsize=(14, 6))

# RGB plot
plt.subplot(1, 2, 1)
plt.scatter(rgb_2d[:, 0], rgb_2d[:, 1], c=rgb_labels, cmap="coolwarm", s=1)
plt.title("RGB Embedding Space (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

# FFT plot
plt.subplot(1, 2, 2)
plt.scatter(fft_2d[:, 0], fft_2d[:, 1], c=fft_labels, cmap="coolwarm", s=1)
plt.title("FFT Embedding Space (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.savefig("embedding_spaces_pca.png", dpi=300)
plt.close()



# 4. Cosine similarity analysis

cos_sims = np.array([
    cosine_similarity(rgb_feats[i].reshape(1, -1),
                      fft_feats[i].reshape(1, -1))[0][0]
    for i in range(len(rgb_feats))
])

plt.figure(figsize=(8, 5))
plt.hist(cos_sims, bins=50, color="purple")
plt.title("Cosine Similarity Between RGB and FFT Embeddings")
plt.xlabel("Cosine similarity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("cosine_similarity_hist.png", dpi=300)
plt.close()

print("Mean cosine similarity:", cos_sims.mean())
print("Median cosine similarity:", np.median(cos_sims))
