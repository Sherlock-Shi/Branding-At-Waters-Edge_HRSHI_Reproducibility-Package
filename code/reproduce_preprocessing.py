from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.config import ADMIN_OFFICIALS, DEMOCRAT_TERMS, REPUBLICAN_TERMS
from src.politician_data import PoliticianDataManager
from src.preprocessing import UkraineContentExtractor, explode_and_clean

RENAME = {
    "排序号": "index", "文档id": "doc_id", "出版社": "publisher",
    "标题": "title", "日期": "date", "时间": "time", "作者": "author",
    "字数": "word_count", "语言": "language", "公司": "company",
    "行业": "industry", "主题": "subject", "地区": "region",
    "所在版面": "page", "ART": "ART", "摘要": "abstract",
    **{f"全文_{i}": f"body_{i}" for i in range(1, 14)},
}
DROP = ["index", "doc_id", "time", "author", "language", "company",
        "industry", "subject", "region", "page", "ART", "abstract"]
BODY = [f"body_{i}" for i in range(1, 14)]


def load_export(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path)
    elif suffix == ".xlsx":
        raw = pd.read_excel(path)
    else:
        raise ValueError("--input must be a CSV or XLSX file")
    missing = [name for name in RENAME if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing Factiva columns: {missing}")
    df = raw.rename(columns=RENAME).drop(columns=DROP)
    df["body"] = df[BODY].apply(
        lambda row: " ".join(row.dropna().astype(str)), axis=1
    )
    df = df.drop(columns=BODY)
    df["primary_body"] = df["body"].str[:50]
    df = df.drop_duplicates(subset=["title", "primary_body"]).copy()
    df["word_count"] = (
        df["word_count"].astype(str).str.replace(" words", "").astype(int)
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    return raw, df.reset_index(drop=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw, articles = load_export(args.input)
    data_root = Path(__file__).resolve().parents[1] / "data"
    manager = PoliticianDataManager(cache_dir=str(data_root / "politicians"))
    rep_names, dem_names = manager.get_party_lists()
    republican_terms = REPUBLICAN_TERMS + rep_names
    democrat_terms = DEMOCRAT_TERMS + dem_names
    extractor = UkraineContentExtractor(republican_terms, democrat_terms)
    extracted = extractor.process_dataframe(
        articles, text_column="body", salience_threshold=0.0,
        apply_threshold=False, weight_transform="log_volume",
    )
    result = explode_and_clean(
        extracted, republican_terms, democrat_terms, ADMIN_OFFICIALS,
        use_entity_mask=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "processed_masked.csv", index=False)
    entity_rows = result["cleaned_sentence"].str.contains(r"\[ENTITY\]", na=False).sum()
    print(f"Input rows: {len(raw)}")
    print(f"Output rows: {len(result)}")
    print(f"Rows containing [ENTITY]: {int(entity_rows)}")


if __name__ == "__main__":
    main()
