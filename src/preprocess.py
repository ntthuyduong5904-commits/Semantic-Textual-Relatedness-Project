import pandas as pd

# Đọc file
df = pd.read_csv(
    "D:\TTCK\Semantic_Relatedness_SemEval2024-main\Semantic_Relatedness_SemEval2024-main\Track A\eng\eng_train.csv"
)

def split_text(text):
    if "\t" in text:
        parts = text.split("\t")
    else:
        parts = text.splitlines()

    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    else:
        return parts[0].strip(), ""

# Tách câu
df[["cau1", "cau2"]] = df["Text"].apply(
    lambda x: pd.Series(split_text(x))
)

# 🔹 Kiểm tra có score hay không
if "Score" in df.columns:
    print("File có cột Score → xuất kèm score")
    out_df = pd.DataFrame({
        "ID": df["PairID"],
        "cau1": df["cau1"],
        "cau2": df["cau2"],
        "score": df["Score"]
    })
else:
    print("File KHÔNG có cột Score → bỏ qua score")
    out_df = pd.DataFrame({
        "ID": df["PairID"],
        "cau1": df["cau1"],
        "cau2": df["cau2"]
    })

# Lưu file
out_df.to_csv(
    "D:/TTCK/preprocessing/eng_train_formatted.csv",
    index=False,
    encoding="utf-8"
)

print("DONE")
print(out_df.head())