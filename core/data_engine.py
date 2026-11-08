from io import BytesIO
import pandas as pd

class DataEngine:
    def __init__(self, datasets):
        self.datasets = datasets

    @classmethod
    def from_uploads(cls, uploads):
        datasets = {}
        for file in uploads:
            raw = file.read()
            if len(raw) > 25 * 1024 * 1024:
                raise ValueError(f"{file.name} is larger than 25 MB.")
            df = pd.read_csv(BytesIO(raw))
            if df.empty:
                raise ValueError(f"{file.name} is empty.")
            datasets[file.name] = df
        return cls(datasets)

    @property
    def primary(self):
        return next(iter(self.datasets.values()))

    def profile(self):
        rows = []
        for name, df in self.datasets.items():
            rows.append({
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            })
        return pd.DataFrame(rows)

    def quality_report(self):
        df = self.primary
        return pd.DataFrame([{
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "missing_%": round(float(df[col].isna().mean() * 100), 2),
            "unique": int(df[col].nunique(dropna=True)),
        } for col in df.columns])

    def schema_text(self):
        parts = []
        for name, df in self.datasets.items():
            parts.append(
                f"DATASET: {name}\nCOLUMNS: " +
                ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
            )
            parts.append("SAMPLE:\n" + df.head(5).to_csv(index=False))
        return "\n\n".join(parts)
