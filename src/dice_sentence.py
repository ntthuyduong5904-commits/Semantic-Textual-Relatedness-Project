# đọc / ghi file CSV
import pandas as pd
# xử lý chuỗi bằng biểu thức chính quy
import re
# đánh giá mô hình STR
from scipy.stats import pearsonr

# ----------------------
# Text preprocessing
# ----------------------
def preprocess(sentence):
    if not isinstance(sentence, str):
        return set()

    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    return set(sentence.split())


# ----------------------
# Dice similarity (Baseline STR)
# ----------------------
def dice_similarity(sent_pair):
    s1, s2 = sent_pair

    t1 = preprocess(s1)
    t2 = preprocess(s2)

    if not t1 and not t2:
        return 0.0

    return 2 * len(t1 & t2) / (len(t1) + len(t2))


# ----------------------
# Main
# ----------------------
def main():
    input_path = r"D:\TTCK\str_510_sentence_pairs.csv"
    output_path = r"D:\TTCK\dice_sentence_results.csv"

    # Read CSV
    df = pd.read_csv(input_path)

    # 👉 Tạo cặp câu từ đúng cột của file Data2.csv
    sentence_pairs = df[["Sentence1", "Sentence2"]].values.tolist()

    # Tính baseline score
    df["baseline_score"] = (
        pd.Series(sentence_pairs)
        .apply(dice_similarity)
        .round(2)
    )

    # 👉 Đánh giá nếu có nhãn chuẩn STR
    if "STR" in df.columns:
        pearson = pearsonr(df["STR"], df["baseline_score"])[0]

        print("Evaluation results:")
        print(f"Pearson correlation: {pearson:.4f}")

    # Giữ nguyên các cột cũ + thêm baseline_score
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
