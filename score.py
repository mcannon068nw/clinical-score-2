from pathlib import Path
import zipfile
import pandas as pd
import os, shutil

def load_pmid_assessments(zip_file, search_method, out_dir="."):
    """
    Extracts <zip_file> into <out_dir>/<zip_stem>/ and concatenates all CSVs.
    Robust to empty files, mixed encodings, and various delimiters.
    """
    zip_path = Path(zip_file)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_path)

    # The extracted directory is typically the ZIP name without ".zip"
    extract_dir = out_path / zip_path.stem

    # Only CSV files (skip hidden/system files)
    csv_files = [p for p in (extract_dir.glob("*.csv")) if p.is_file()]

    # If your zip extracted files directly into out_dir (no folder),
    # fall back to scanning that directory too:
    if not csv_files:
        csv_files = [p for p in out_path.glob("*.csv") if p.is_file()]

    dfs = []
    failures = []

    # Try a few common encodings
    encodings_to_try = ["utf-8", "utf-8-sig", "utf-16", "latin1"]

    for fpath in sorted(csv_files):
        # Skip truly empty files quickly
        if fpath.stat().st_size == 0:
            failures.append((str(fpath), "empty file"))
            continue

        last_err = None
        tdf = None

        for enc in encodings_to_try:
            try:
                # sep=None + engine="python" asks pandas to infer delimiter
                tdf = pd.read_csv(
                    fpath,
                    encoding=enc,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip",  # skip malformed rows instead of erroring
                )
                # If it loaded but has no columns (e.g., whitespace-only), treat as failure
                if tdf.shape[1] == 0:
                    raise pd.errors.EmptyDataError("No columns after parse")
                break
            except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                last_err = e
                tdf = None

        if tdf is None:
            failures.append((str(fpath), repr(last_err)))
            continue

        # Derive gene/drug from filename "GENE_DRUG.csv"
        name = fpath.name.rsplit(".", 1)[0]
        parts = name.split("_", 1)
        if len(parts) == 2:
            gene, drug = parts
        else:
            # Fallback if underscore missing
            gene, drug = parts[0], ""

        tdf["gene"] = gene
        tdf["drug"] = drug
        tdf["method"] = search_method
        dfs.append(tdf)

    if not dfs:
        # Nothing loaded — surface failures to help debugging
        raise RuntimeError(
            "No CSVs could be loaded. Examples of failures: "
            + "; ".join([f"{p} -> {err}" for p, err in failures[:5]])
        )

    df = pd.concat(dfs, ignore_index=True)

    # Optional: log any skipped files so you know what was ignored
    if failures:
        print("Skipped files:")
        for p, err in failures:
            print(f" - {p}: {err}")

    shutil.rmtree(extract_dir, ignore_errors=True)

    return df
