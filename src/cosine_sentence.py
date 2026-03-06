import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# ----------------------
# Load PhoBERT embedding model
# ----------------------
model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

# ----------------------
# Main
# ----------------------
def main():
    input_path = r"D:\TTCK\str_510_sentence_pairs.csv"
    output_path = r"D:\TTCK\cosine_paragraph_results.csv"

    df = pd.read_csv(input_path)

    # Xóa khoảng trắng nếu có
    df.columns = df.columns.str.strip()

    # Lấy đúng tên cột trong file
    sentences1 = df["Sentence1"].astype(str).tolist()
    sentences2 = df["Sentence2"].astype(str).tolist()

    # Encode sentences
    emb1 = model.encode(sentences1, convert_to_tensor=False)
    emb2 = model.encode(sentences2, convert_to_tensor=False)

    # Compute cosine similarity
    cosine_scores = [
        cosine_similarity([e1], [e2])[0][0]
        for e1, e2 in zip(emb1, emb2)
    ]

    df["embedding_cosine"] = [round(s, 3) for s in cosine_scores]

    # Evaluation nếu có nhãn STR
    if "STR" in df.columns:
        pearson = pearsonr(df["STR"], df["embedding_cosine"])[0]
        print(f"Pearson correlation: {pearson:.4f}")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()