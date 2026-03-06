#đọc / ghi file CSV
import pandas as pd
#xử lý chuỗi bằng biểu thức chính quy
import re
#dùng để đánh giá chất lượng mô hình STR, so sánh điểm dự đoán với điểm chuẩn
from scipy.stats import pearsonr, spearmanr

# ----------------------
# Text preprocessing ( Hàm tiền xử lý văn bản) : Đảm bảo dữ liệu đầu vào là chuỗi
#Nếu là số, rỗng -> trả về tập rỗng
# ----------------------
def preprocess(sentence):
    if not isinstance(sentence, str):
        return set()
#chuyển về chữ thường
    sentence = sentence.lower()
    #xóa dấu câu và ký tự đặc biệt
    sentence = re.sub(r"[^\w\s]", "", sentence)
    #tách từ và đưa vào set ,tránh lặp từ
    return set(sentence.split())


# ----------------------
# Dice similarity: Baseline STR
# ----------------------
def dice_similarity(sent_pair):
    #nhận vào 1 cặp câu
    s1, s2 = sent_pair

#tiền xử lý 2 câu thành 2 tập từ
    t1 = preprocess(s1)
    t2 = preprocess(s2)

#nếu 2 câu rỗng thì độ liên quan  0
    if not t1 and not t2:
        return 0.0

    return 2 * len(t1 & t2) / (len(t1) + len(t2))


# ----------------------
# Main
# ----------------------
def main():
    input_path = r"D:\TTCK\translated\vie_train_formatted.csv"
    output_path = r"D:\TTCK\results\STR_baseline_vie_train_formatted_output.csv"

    # Read CSV
    df = pd.read_csv(input_path)

    # Tạo cặp câu (Create sentence pairs (only for computing, not for saving))
    sentence_pairs = df[["sentence_1_vie", "sentence_2_viet"]].values.tolist()

    # Tính  baseline score (2 chữ số sau thập phân)
    df["baseline_score"] = (
        pd.Series(sentence_pairs)
        .apply(dice_similarity)
        .round(2)
    )

    #Đánh giá mô hình
    score_col = None
    if "score" in df.columns:
        score_col = "score"
    elif "Score" in df.columns:
        score_col = "Score"

    if score_col:
        pearson = pearsonr(df[score_col], df["baseline_score"])[0]
        

        print("Evaluation results:")
        #Pearson correlation đo mức độ tương quan tuyến tính giữa score và baseline_score
        #1.0: giống y hệt, 0.0: kh liên quan, 0.7: theo khá sát score đã có
        print(f"Pearson correlation : {pearson:.4f}")
        

    # Columns to keep when exporting
    output_columns = [
        "ID",
        "sentence_1_vie",
        "sentence_2_viet",
        "score",
        "baseline_score"
    ]

    # Keep only existing columns (safe for files without score): Xuất file kết quả
    output_columns = [c for c in output_columns if c in df.columns]

    df_output = df[output_columns]

    # Save output
    df_output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
