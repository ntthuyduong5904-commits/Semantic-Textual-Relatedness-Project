# đọc / ghi file CSV
import pandas as pd
# xử lý chuỗi bằng biểu thức chính quy
import re
# đánh giá mô hình STR
from scipy.stats import pearsonr


# ----------------------
# Text preprocessing
# ----------------------
def preprocess(text):
    if not isinstance(text, str):
        return set()

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()

    return set(tokens)


# ----------------------
# Dice similarity (Baseline STR cho cặp đoạn)
# ----------------------
def dice_similarity(paragraph_pair):
    p1, p2 = paragraph_pair

    t1 = preprocess(p1)
    t2 = preprocess(p2)

    if not t1 and not t2:
        return 0.0

    return 2 * len(t1 & t2) / (len(t1) + len(t2))


# ----------------------
# Main
# ----------------------
def main():
    input_path = r"D:\TTCK\str_87_paragraph_pairs.csv" 
    output_path = r"D:\TTCK\dice_paragraph_results.csv"

    # Read CSV
    df = pd.read_csv(input_path)

    # 👉 Tạo cặp đoạn từ đúng tên cột mới
    paragraph_pairs = df[["Paragraph1", "Paragraph2"]].values.tolist()

    # Tính baseline score
    df["baseline_score"] = (
        pd.Series(paragraph_pairs)
        .apply(dice_similarity)
        .round(3)
    )

    # 👉 Đánh giá nếu có nhãn STR
    if "STR" in df.columns:
        pearson = pearsonr(df["STR"], df["baseline_score"])[0]

        print("Evaluation results:")
        print(f"Pearson correlation: {pearson:.4f}")

    # Giữ nguyên các cột cũ + thêm baseline_score
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()