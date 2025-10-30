import pandas as pd
import re

def clean_text(text):
    """Clean unwanted characters, URLs, and symbols from text."""
    if pd.isnull(text):
        return ""
    text = str(text)
    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # remove non-alphanumeric characters (keep basic punctuation)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\- ]", " ", text)
    # replace multiple spaces with one
    text = re.sub(r"\s+", " ", text).strip()
    # lowercase
    return text.lower()

def clean_dataset(file_path, save_path="cleaned_IFND.csv"):
    """Load, clean, and save dataset with cleaned Statement column."""
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp1252")

    print("Before cleaning:")
    print(df.info())

    # Use your text column explicitly
    text_col = "Statement"
    label_col = "Label"

    # Clean the text column
    df[text_col] = df[text_col].apply(clean_text)

    # Drop duplicates and empty statements
    df = df.drop_duplicates(subset=[text_col])
    df = df.dropna(subset=[text_col])

    # Standardize label names (optional)
    df[label_col] = df[label_col].str.upper().replace({
        "TRUE": "REAL",
        "FALSE": "FAKE",
        "REAL NEWS": "REAL",
        "FAKE NEWS": "FAKE"
    })

    print(f"\nAfter cleaning: {len(df)} rows remain")
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"✅ Cleaned dataset saved to {save_path}")

if __name__ == "__main__":
    clean_dataset("IFND.csv")
