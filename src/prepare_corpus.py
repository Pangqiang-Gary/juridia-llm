import pandas as pd
from pathlib import Path

def build_corpus(train_csv: str, out_csv: str):
    """
    Combine multiple columns into a single text corpus for fine-tuning.
    Example fields: long_title, doc_type, date, file_name.
    """
    df = pd.read_csv(train_csv)
    df.fillna("", inplace=True)
    texts = (
        df["long_title"] + " [TYPE] " + df["doc_type"] +
        " [DATE] " + df["date"] + " [FILE] " + df["file_name"]
    )
    corpus = pd.DataFrame({"id": df["Id"], "text": texts})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(out_csv, index=False)

if __name__ == "__main__":
    build_corpus("data/train.csv", "outputs/corpus.csv")
