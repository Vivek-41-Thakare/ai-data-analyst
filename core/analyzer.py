from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px


# ============================================================
# RESULT
# ============================================================

@dataclass
class AnalysisResult:
    answer: str
    evidence: list[str]
    code: str | None = None
    sql: str | None = None
    chart: Any | None = None


# ============================================================
# ANALYST
# ============================================================

class Analyst:
    def __init__(
        self,
        engine,
        llm=None,
        history: list[dict] | None = None,
    ):
        self.engine = engine
        self.llm = llm
        self.history = history or []

    # ========================================================
    # DATAFRAME
    # ========================================================

    def _get_dataframe(self) -> pd.DataFrame | None:
        """
        Resolve the primary dataframe from DataEngine.

        Supported storage patterns:
        - engine.primary
        - engine.df
        - engine.data
        - engine.get_dataframe()
        - engine.get_data()
        """

        try:
            if hasattr(self.engine, "primary"):
                df = self.engine.primary

                if isinstance(df, pd.DataFrame):
                    return df.copy()
        except Exception:
            pass

        try:
            if hasattr(self.engine, "df"):
                df = self.engine.df

                if isinstance(df, pd.DataFrame):
                    return df.copy()
        except Exception:
            pass

        try:
            if hasattr(self.engine, "data"):
                df = self.engine.data

                if isinstance(df, pd.DataFrame):
                    return df.copy()
        except Exception:
            pass

        try:
            if hasattr(self.engine, "get_dataframe"):
                df = self.engine.get_dataframe()

                if isinstance(df, pd.DataFrame):
                    return df.copy()
        except Exception:
            pass

        try:
            if hasattr(self.engine, "get_data"):
                df = self.engine.get_data()

                if isinstance(df, pd.DataFrame):
                    return df.copy()
        except Exception:
            pass

        return None

    # ========================================================
    # MAIN ANSWER ROUTER
    # ========================================================

    def answer(self, question: str) -> dict:
        q = (question or "").strip()

        if not q:
            return self._to_dict(
                AnalysisResult(
                    answer="Please enter a question.",
                    evidence=[],
                )
            )

        df = self._get_dataframe()

        if df is None or df.empty:
            return self._to_dict(
                AnalysisResult(
                    answer="No dataset is currently available.",
                    evidence=[
                        "No usable dataframe was found in the DataEngine."
                    ],
                )
            )

        result = None

        # ----------------------------------------------------
        # 1. SQL
        # ----------------------------------------------------

        if self._is_sql_request(q):
            result = self._sql_result(q, df)

        # ----------------------------------------------------
        # 2. Pandas / Python code
        # ----------------------------------------------------

        if result is None and self._is_code_request(q):
            result = self._code_result(q, df)

        # ----------------------------------------------------
        # 3. Dataset summary
        # ----------------------------------------------------

        if result is None and self._is_summary_request(q):
            result = self._business_summary(q, df)

        # ----------------------------------------------------
        # 4. Data quality
        # ----------------------------------------------------

        if result is None and self._is_quality_request(q):
            result = self._data_quality_result(q, df)

        # ----------------------------------------------------
        # 5. Anomaly detection
        # ----------------------------------------------------

        if result is None and self._is_anomaly_request(q):
            result = self._anomaly_result(q, df)

        # ----------------------------------------------------
        # 6. Trend
        # ----------------------------------------------------

        if result is None and self._is_trend_request(q):
            result = self._monthly_trend_result(q, df)

        # ----------------------------------------------------
        # 7. Average
        # ----------------------------------------------------

        if result is None and self._is_average_request(q):
            result = self._average_result(q, df)

        # ----------------------------------------------------
        # 8. Total
        # ----------------------------------------------------

        if result is None and self._is_total_request(q):
            result = self._total_result(q, df)

        # ----------------------------------------------------
        # 9. Specific entity
        #
        # Examples:
        #   What is the revenue generated by West?
        #   Revenue generated by C6?
        #   What is the ROI for Product A?
        # ----------------------------------------------------

        if result is None:
            result = self._specific_group_revenue_result(q, df)

        # ----------------------------------------------------
        # 10. Ranking
        #
        # Examples:
        #   Top 5 customers by revenue
        #   Highest revenue by region
        #   Bottom 5 products by revenue
        # ----------------------------------------------------

        if result is None and self._is_ranking_request(q):
            result = self._ranking_result(q, df)

        # ----------------------------------------------------
        # 11. Underperforming
        # ----------------------------------------------------

        if result is None and self._is_underperforming_request(q):
            result = self._underperforming_result(q, df)

        # ----------------------------------------------------
        # 12. Breakdown
        # ----------------------------------------------------

        if result is None and self._is_breakdown_request(q):
            result = self._breakdown_result(q, df)

        # ----------------------------------------------------
        # 13. Single metric
        # ----------------------------------------------------

        if result is None and self._is_single_metric_request(q):
            result = self._single_metric_result(q, df)

        # ----------------------------------------------------
        # 14. Context
        #
        # Examples:
        #   How much revenue did that region generate?
        #   What about East?
        #   What about C6?
        #   And South?
        # ----------------------------------------------------

        if result is None:
            result = self._context_result(q, df)

        # ----------------------------------------------------
        # 15. LLM fallback
        # ----------------------------------------------------

        if result is None:
            result = self._llm_fallback(q, df)

        return self._to_dict(result)

    # ========================================================
    # DICT CONVERSION
    # ========================================================

    def _to_dict(self, result: AnalysisResult) -> dict:
        return {
            "answer": result.answer,
            "evidence": result.evidence,
            "code": result.code,
            "sql": result.sql,
            "chart": result.chart,
        }

    # ========================================================
    # COLUMN HELPERS
    # ========================================================

    def _numeric_columns(self, df: pd.DataFrame) -> list[str]:
        return list(
            df.select_dtypes(include="number").columns
        )

    def _categorical_columns(self, df: pd.DataFrame) -> list[str]:
        return list(
            df.select_dtypes(
                include=["object", "string", "category"]
            ).columns
        )

    def _find_column(
        self,
        df: pd.DataFrame,
        candidates: list[str],
    ) -> str | None:

        exact = {
            str(col).lower(): col
            for col in df.columns
        }

        # Exact match first.
        for candidate in candidates:
            if candidate.lower() in exact:
                return exact[candidate.lower()]

        # Partial match second.
        for col in df.columns:
            col_lower = str(col).lower()

            for candidate in candidates:
                if candidate.lower() in col_lower:
                    return col

        return None

    # ========================================================
    # REVENUE / METRIC COLUMN
    # ========================================================

    def _revenue_column(
        self,
        df: pd.DataFrame,
    ) -> str | None:

        # Common business metric aliases.
        #
        # Revenue is intentionally first because the existing
        # dataset uses revenue.
        #
        # Additional aliases:
        # ARR, ROI, Gross, LTV, etc.
        candidates = [
            "revenue",
            "sales",
            "amount",
            "total",
            "value",
            "income",
            "profit",
            "arr",
            "annual recurring revenue",
            "roi",
            "return on investment",
            "gross",
            "gross profit",
            "gross revenue",
            "ltv",
            "lifetime value",
            "customer lifetime value",
        ]

        return self._find_column(
            df,
            candidates,
        )

    # ========================================================
    # METRIC COLUMN
    # ========================================================

    def _metric_column(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> str | None:

        q_lower = q.lower()

        metric_aliases = {
            "revenue": [
                "revenue",
                "sales",
                "amount",
                "total",
                "value",
                "income",
            ],
            "arr": [
                "arr",
                "annual recurring revenue",
            ],
            "roi": [
                "roi",
                "return on investment",
            ],
            "gross": [
                "gross",
                "gross profit",
                "gross revenue",
            ],
            "ltv": [
                "ltv",
                "lifetime value",
                "customer lifetime value",
            ],
            "profit": [
                "profit",
                "net profit",
            ],
        }

        # First identify the metric explicitly asked by user.
        for _, aliases in metric_aliases.items():

            for alias in aliases:

                if re.search(
                    rf"\b{re.escape(alias)}\b",
                    q_lower,
                ):

                    col = self._find_column(
                        df,
                        [alias],
                    )

                    if col is not None:
                        return col

        # Fallback to revenue-like metric.
        return self._revenue_column(df)

    # ========================================================
    # GROUP COLUMN
    # ========================================================

    def _group_column(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> str | None:

        q_lower = q.lower()

        preferred = [
            "region",
            "customer",
            "product",
            "category",
            "segment",
            "city",
            "country",
            "department",
            "salesperson",
            "employee",
        ]

        # Explicit dimension mentioned in question.
        for candidate in preferred:

            pattern = rf"\b{re.escape(candidate)}s?\b"

            if re.search(pattern, q_lower):

                col = self._find_column(
                    df,
                    [candidate],
                )

                if col is not None:
                    return col

        # Fallback to categorical column.
        categorical = self._categorical_columns(df)

        for col in categorical:

            col_lower = str(col).lower()

            if (
                "date" in col_lower
                or "time" in col_lower
                or "timestamp" in col_lower
            ):
                continue

            return col

        return None

    # ========================================================
    # REQUEST DETECTORS
    # ========================================================

    def _is_sql_request(self, q: str) -> bool:
        q = q.lower()

        return (
            "sql" in q
            or "write sql" in q
            or "show sql" in q
            or "sql query" in q
        )

    def _is_code_request(self, q: str) -> bool:
        q = q.lower()

        return (
            "pandas code" in q
            or "python code" in q
            or "show code" in q
            or "generated code" in q
            or "write code" in q
        )

    def _is_summary_request(self, q: str) -> bool:
        q = q.lower()

        patterns = [
            "summary",
            "summarize",
            "summarise",
            "business summary",
            "dataset overview",
            "overview of the dataset",
            "describe the dataset",
            "complete summary",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    def _is_quality_request(self, q: str) -> bool:
        q = q.lower()

        patterns = [
            "data quality",
            "quality report",
            "missing values",
            "missing data",
            "null values",
            "duplicate rows",
            "duplicates",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    def _is_anomaly_request(self, q: str) -> bool:
        q = q.lower()

        patterns = [
            "anomal",
            "outlier",
            "unusual value",
            "unusual values",
            "detect unusual",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    def _is_trend_request(self, q: str) -> bool:
        q = q.lower()

        patterns = [
            "trend",
            "monthly trend",
            "over time",
            "month by month",
            "monthly revenue",
            "monthly sales",
            "monthly arr",
            "monthly roi",
            "monthly gross",
            "monthly ltv",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    def _is_average_request(self, q: str) -> bool:
        q = q.lower()

        return (
            "average" in q
            or "mean" in q
        )

    def _is_total_request(self, q: str) -> bool:
        q = q.lower()

        patterns = [
            "total revenue",
            "total sales",
            "total amount",
            "sum of revenue",
            "sum revenue",
            "overall revenue",
            "overall sales",
        ]

        if self._is_contextual_question(q):
            return False

        return any(
            pattern in q
            for pattern in patterns
        )

    def _is_ranking_request(self, q: str) -> bool:
        q = q.lower()

        ranking_words = [
            "top",
            "highest",
            "largest",
            "best",
            "bottom",
            "lowest",
            "smallest",
            "worst",
        ]

        return any(
            word in q
            for word in ranking_words
        )

    def _is_underperforming_request(
        self,
        q: str,
    ) -> bool:

        q = q.lower()

        patterns = [
            "underperforming",
            "under performing",
            "poor performing",
            "poorly performing",
            "weak performing",
            "weak products",
            "weak customers",
            "weak regions",
            "worst performing",
            "least performing",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    def _is_breakdown_request(self, q: str) -> bool:
        q = q.lower()

        patterns = [
            "by region",
            "by customer",
            "by product",
            "by category",
            "by segment",
            "by city",
            "by country",
            "breakdown",
            "grouped by",
            "per region",
            "per customer",
            "per product",
            "per category",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    def _is_single_metric_request(
        self,
        q: str,
    ) -> bool:

        q_lower = q.lower()

        if self._is_contextual_question(q_lower):
            return False

        metric_words = [
            "revenue",
            "sales",
            "amount",
            "profit",
            "income",
            "value",
            "arr",
            "roi",
            "gross",
            "ltv",
        ]

        question_words = [
            "what is",
            "how much",
            "show me",
            "give me",
        ]

        return (
            any(
                word in q_lower
                for word in metric_words
            )
            and
            any(
                word in q_lower
                for word in question_words
            )
        )

    # ========================================================
    # CONTEXT QUESTION DETECTOR
    # ========================================================

    def _is_contextual_question(
        self,
        q: str,
    ) -> bool:

        q_lower = q.lower()

        patterns = [
            r"\bwhat\s+about\b",
            r"\band\s+what\s+about\b",
            r"\band\s+(?:east|west|north|south)\b",
            r"\bthat\s+region\b",
            r"\bthat\s+customer\b",
            r"\bthat\s+product\b",
            r"\bthat\s+category\b",
            r"\bthat\s+segment\b",
            r"\bthat\s+city\b",
            r"\bthat\s+country\b",
            r"\bthat\s+one\b",
            r"\bthis\s+region\b",
            r"\bthis\s+customer\b",
            r"\bthis\s+product\b",
            r"\bthis\s+category\b",
            r"\bthis\s+segment\b",
            r"\bthis\s+city\b",
            r"\bthis\s+country\b",
            r"\bthis\s+one\b",
        ]

        return any(
            re.search(
                pattern,
                q_lower,
            )
            for pattern in patterns
        )

    # ========================================================
    # NUMBER EXTRACTION
    # ========================================================

    def _extract_n(
        self,
        q: str,
        default: int = 5,
    ) -> int:

        match = re.search(
            r"\b(?:top|bottom)\s+(\d+)\b",
            q.lower(),
        )

        if match:
            return max(
                1,
                int(match.group(1)),
            )

        return default

    # ========================================================
    # BUSINESS SUMMARY
    # ========================================================

    def _business_summary(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        rows = len(df)
        cols = len(df.columns)

        numeric_cols = self._numeric_columns(
            df
        )

        revenue_col = self._revenue_column(
            df
        )

        total_revenue = None
        average_revenue = None

        if revenue_col:

            numeric_revenue = pd.to_numeric(
                df[revenue_col],
                errors="coerce",
            )

            total_revenue = float(
                numeric_revenue.sum()
            )

            average_revenue = float(
                numeric_revenue.mean()
            )

        missing_cells = int(
            df.isna().sum().sum()
        )

        if total_revenue is not None:

            answer = (
                f"The dataset contains {rows} rows across "
                f"{cols} columns. "
                f"Total {revenue_col} is "
                f"{total_revenue:,.2f}. "
                f"Average {revenue_col} per transaction is "
                f"{average_revenue:,.2f}."
            )

        else:

            answer = (
                f"The dataset contains {rows} rows across "
                f"{cols} columns and "
                f"{len(numeric_cols)} numeric column(s). "
                f"It contains {missing_cells} missing cell(s)."
            )

        evidence = [
            f"Dataset contains {rows} rows and {cols} columns.",
            f"Found {len(numeric_cols)} numeric column(s).",
            f"Dataset contains {missing_cells} missing cell(s).",
        ]

        if revenue_col:

            evidence.append(
                f"Calculated SUM({revenue_col}) and "
                f"MEAN({revenue_col})."
            )

        return AnalysisResult(
            answer=answer,
            evidence=evidence,
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    def _data_quality_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        missing = int(
            df.isna().sum().sum()
        )

        duplicates = int(
            df.duplicated().sum()
        )

        if (
            missing == 0
            and duplicates == 0
        ):

            answer = (
                "The dataset has no missing values "
                "and no duplicate rows."
            )

        else:

            answer = (
                f"The dataset contains {missing} missing cell(s) "
                f"and {duplicates} duplicate row(s)."
            )

        evidence = [
            f"Checked {len(df.columns)} columns for missing values.",
            f"Found {missing} missing cell(s).",
            "Checked duplicate rows.",
            f"Found {duplicates} duplicate row(s).",
        ]

        return AnalysisResult(
            answer=answer,
            evidence=evidence,
        )

    # ========================================================
    # ANOMALY
    # ========================================================

    def _anomaly_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        revenue_col = self._revenue_column(
            df
        )

        if revenue_col is None:

            numeric = self._numeric_columns(
                df
            )

            if not numeric:

                return AnalysisResult(
                    answer=(
                        "No numeric column is available "
                        "for anomaly detection."
                    ),
                    evidence=[],
                )

            revenue_col = numeric[0]

        values = pd.to_numeric(
            df[revenue_col],
            errors="coerce",
        ).dropna()

        if values.empty:

            return AnalysisResult(
                answer=(
                    f"No valid numeric values were found "
                    f"in `{revenue_col}`."
                ),
                evidence=[],
            )

        q1 = float(
            values.quantile(0.25)
        )

        q3 = float(
            values.quantile(0.75)
        )

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        numeric_series = pd.to_numeric(
            df[revenue_col],
            errors="coerce",
        )

        mask = (
            (numeric_series < lower)
            |
            (numeric_series > upper)
        )

        anomalies = df.loc[
            mask
        ]

        if anomalies.empty:

            answer = (
                f"No potential anomalies were detected in "
                f"{revenue_col} using the IQR method."
            )

        else:

            values_text = ", ".join(
                f"{float(v):,.2f}"
                for v in anomalies[
                    revenue_col
                ]
            )

            count = len(
                anomalies
            )

            noun = (
                "anomaly"
                if count == 1
                else "anomalies"
            )

            answer = (
                f"Detected {count} potential "
                f"{noun} in {revenue_col}: "
                f"{values_text}."
            )

        evidence = [
            f"Checked `{revenue_col}` using the IQR method.",
            f"Q1 = {q1:,.2f}.",
            f"Q3 = {q3:,.2f}.",
            f"Lower bound = {lower:,.2f}.",
            f"Upper bound = {upper:,.2f}.",
        ]

        return AnalysisResult(
            answer=answer,
            evidence=evidence,
        )

    # ========================================================
    # TOTAL
    # ========================================================

    def _total_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        revenue_col = self._metric_column(
            q,
            df,
        )

        if revenue_col is None:

            return AnalysisResult(
                answer=(
                    "I could not find a suitable "
                    "numeric metric column."
                ),
                evidence=[],
            )

        total = float(
            pd.to_numeric(
                df[revenue_col],
                errors="coerce",
            ).sum()
        )

        return AnalysisResult(
            answer=(
                f"The total {revenue_col} is "
                f"{total:,.2f}."
            ),
            evidence=[
                f"Calculated SUM({revenue_col}) across all rows."
            ],
            code=(
                f"result = df['{revenue_col}'].sum()"
            ),
            sql=(
                f"SELECT SUM({revenue_col}) "
                f"AS total_{revenue_col} "
                f"FROM data;"
            ),
        )

    # ========================================================
    # AVERAGE
    # ========================================================

    def _average_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        revenue_col = self._metric_column(
            q,
            df,
        )

        if revenue_col is None:

            return AnalysisResult(
                answer=(
                    "I could not find a suitable "
                    "numeric metric column."
                ),
                evidence=[],
            )

        average = float(
            pd.to_numeric(
                df[revenue_col],
                errors="coerce",
            ).mean()
        )

        return AnalysisResult(
            answer=(
                f"The average {revenue_col} is "
                f"{average:,.2f}."
            ),
            evidence=[
                f"Calculated MEAN({revenue_col}) across all rows."
            ],
            code=(
                f"result = df['{revenue_col}'].mean()"
            ),
            sql=(
                f"SELECT AVG({revenue_col}) "
                f"AS average_{revenue_col} "
                f"FROM data;"
            ),
        )

    # ========================================================
    # FIND ENTITY COLUMN
    # ========================================================

    def _find_entity_column(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> str | None:

        q_lower = q.lower()

        preferred = [
            "region",
            "customer",
            "product",
            "category",
            "segment",
            "city",
            "country",
            "department",
            "salesperson",
            "employee",
        ]

        # ----------------------------------------------------
        # Preferred business columns
        # ----------------------------------------------------

        for candidate in preferred:

            col = self._find_column(
                df,
                [candidate],
            )

            if col is None:
                continue

            values = (
                df[col]
                .dropna()
                .astype(str)
                .unique()
            )

            for value in values:

                value_lower = (
                    value.strip().lower()
                )

                if not value_lower:
                    continue

                if re.search(
                    rf"\b{re.escape(value_lower)}\b",
                    q_lower,
                ):
                    return col

        # ----------------------------------------------------
        # All categorical columns
        # ----------------------------------------------------

        for col in self._categorical_columns(
            df
        ):

            col_lower = str(
                col
            ).lower()

            if (
                "date" in col_lower
                or "time" in col_lower
                or "timestamp" in col_lower
            ):
                continue

            values = (
                df[col]
                .dropna()
                .astype(str)
                .unique()
            )

            for value in values:

                value_lower = (
                    value.strip().lower()
                )

                if not value_lower:
                    continue

                if re.search(
                    rf"\b{re.escape(value_lower)}\b",
                    q_lower,
                ):
                    return col

        return None

    # ========================================================
    # SPECIFIC ENTITY METRIC
    # ========================================================

    def _specific_group_revenue_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult | None:

        metric_col = self._metric_column(
            q,
            df,
        )

        if metric_col is None:
            return None

        # Ranking should be handled by ranking_result.
        if self._is_ranking_request(q):
            return None

        # Breakdown should be handled by breakdown_result.
        if self._is_breakdown_request(q):
            return None

        entity_col = self._find_entity_column(
            q,
            df,
        )

        if entity_col is None:
            return None

        q_lower = q.lower()

        relevant_words = [
            "revenue",
            "sales",
            "amount",
            "generated",
            "generate",
            "made",
            "make",
            "earned",
            "earn",
            "profit",
            "income",
            "value",
            "arr",
            "roi",
            "gross",
            "ltv",
            "how much",
            "what is",
        ]

        if not any(
            word in q_lower
            for word in relevant_words
        ):
            return None

        entity_value = None

        for value in (
            df[entity_col]
            .dropna()
            .unique()
        ):

            value_str = str(
                value
            ).strip()

            if re.search(
                rf"\b{re.escape(value_str.lower())}\b",
                q_lower,
            ):
                entity_value = value
                break

        if entity_value is None:
            return None

        mask = (
            df[entity_col]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            str(entity_value)
            .strip()
            .lower()
        )

        total = float(
            pd.to_numeric(
                df.loc[
                    mask,
                    metric_col,
                ],
                errors="coerce",
            ).sum()
        )

        return AnalysisResult(
            answer=(
                f"{entity_value} generated "
                f"{metric_col} of {total:,.2f}."
            ),
            evidence=[
                f"Filtered `{entity_col}` for `{entity_value}`.",
                f"Calculated SUM({metric_col}) "
                f"for the matching rows.",
            ],
            code=(
                "result = (\n"
                f"    df[df['{entity_col}'].astype(str).str.lower() "
                f"== '{str(entity_value).lower()}']"
                f"['{metric_col}']\n"
                "    .sum()\n"
                ")"
            ),
            sql=(
                f"SELECT SUM({metric_col}) AS revenue "
                f"FROM data "
                f"WHERE LOWER({entity_col}) = "
                f"'{str(entity_value).lower()}';"
            ),
            chart=self._bar_chart(
                df.loc[
                    mask
                ],
                entity_col,
                metric_col,
                title=(
                    f"{metric_col} for "
                    f"{entity_value}"
                ),
            ),
        )

    # ========================================================
    # RANKING
    # ========================================================

    def _ranking_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult | None:

        group_col = self._group_column(
            q,
            df,
        )

        metric_col = self._metric_column(
            q,
            df,
        )

        if (
            group_col is None
            or metric_col is None
        ):
            return None

        n = self._extract_n(
            q,
            default=5,
        )

        q_lower = q.lower()

        ascending = (
            "bottom" in q_lower
            or "lowest" in q_lower
            or "smallest" in q_lower
            or "worst" in q_lower
        )

        grouped = (
            df.groupby(
                group_col,
                dropna=False,
            )[metric_col]
            .sum()
            .reset_index()
            .sort_values(
                metric_col,
                ascending=ascending,
            )
            .head(n)
        )

        direction = (
            "bottom"
            if ascending
            else "top"
        )

        parts = []

        for _, row in grouped.iterrows():

            parts.append(
                f"{row[group_col]}: "
                f"{float(row[metric_col]):,.2f}"
            )

        answer = (
            f"{direction.capitalize()} {n} "
            f"{group_col}s by {metric_col}: "
            + " | ".join(parts)
            + "."
        )

        evidence = [
            f"Grouped `{metric_col}` by `{group_col}`.",
            f"Calculated SUM({metric_col}) "
            f"for each group.",
            (
                "Sorted groups in ascending order."
                if ascending
                else
                "Sorted groups in descending order."
            ),
            f"Selected the {direction} {n} result(s).",
        ]

        code = (
            "result = (\n"
            f"    df.groupby('{group_col}', as_index=False)"
            f"['{metric_col}']\n"
            "      .sum()\n"
            f"      .sort_values('{metric_col}', "
            f"ascending={ascending})\n"
            f"      .head({n})\n"
            ")"
        )

        sql = (
            f"SELECT {group_col}, "
            f"SUM({metric_col}) AS revenue\n"
            f"FROM data\n"
            f"GROUP BY {group_col}\n"
            f"ORDER BY revenue "
            f"{'ASC' if ascending else 'DESC'}\n"
            f"LIMIT {n};"
        )

        chart = self._bar_chart(
            grouped,
            group_col,
            metric_col,
            title=(
                f"{direction.capitalize()} "
                f"{n} {group_col}s by {metric_col}"
            ),
        )

        return AnalysisResult(
            answer=answer,
            evidence=evidence,
            code=code,
            sql=sql,
            chart=chart,
        )

    # ========================================================
    # UNDERPERFORMING
    # ========================================================

    def _underperforming_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult | None:

        group_col = self._group_column(
            q,
            df,
        )

        metric_col = self._metric_column(
            q,
            df,
        )

        if (
            group_col is None
            or metric_col is None
        ):
            return None

        grouped = (
            df.groupby(
                group_col,
                dropna=False,
            )[metric_col]
            .sum()
            .reset_index()
            .sort_values(
                metric_col,
                ascending=True,
            )
            .head(5)
        )

        parts = [
            f"{row[group_col]}: "
            f"{float(row[metric_col]):,.2f}"
            for _, row in grouped.iterrows()
        ]

        answer = (
            f"Underperforming {group_col}s by "
            f"{metric_col}: "
            + " | ".join(parts)
            + "."
        )

        evidence = [
            f"Grouped `{metric_col}` by `{group_col}`.",
            f"Calculated SUM({metric_col}) for each group.",
            "Sorted groups from lowest to highest performance.",
            "Selected the 5 lowest-performing groups.",
        ]

        code = (
            "result = (\n"
            f"    df.groupby('{group_col}', as_index=False)"
            f"['{metric_col}']\n"
            "      .sum()\n"
            f"      .sort_values('{metric_col}', "
            "ascending=True)\n"
            "      .head(5)\n"
            ")"
        )

        sql = (
            f"SELECT {group_col}, "
            f"SUM({metric_col}) AS revenue\n"
            f"FROM data\n"
            f"GROUP BY {group_col}\n"
            f"ORDER BY revenue ASC\n"
            f"LIMIT 5;"
        )

        chart = self._bar_chart(
            grouped,
            group_col,
            metric_col,
            title=(
                f"Underperforming {group_col}s"
            ),
        )

        return AnalysisResult(
            answer=answer,
            evidence=evidence,
            code=code,
            sql=sql,
            chart=chart,
        )

    # ========================================================
    # BREAKDOWN
    # ========================================================

    def _breakdown_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult | None:

        group_col = self._group_column(
            q,
            df,
        )

        metric_col = self._metric_column(
            q,
            df,
        )

        if (
            group_col is None
            or metric_col is None
        ):
            return None

        grouped = (
            df.groupby(
                group_col,
                dropna=False,
            )[metric_col]
            .sum()
            .reset_index()
            .sort_values(
                metric_col,
                ascending=False,
            )
        )

        parts = []

        for _, row in grouped.iterrows():

            parts.append(
                f"{row[group_col]}: "
                f"{float(row[metric_col]):,.2f}"
            )

        answer = (
            f"{metric_col.capitalize()} by "
            f"{group_col}: "
            + " | ".join(parts)
            + "."
        )

        evidence = [
            f"Grouped `{metric_col}` by `{group_col}`.",
            f"Calculated SUM({metric_col}) "
            f"for each group.",
            f"Sorted the groups by {metric_col} "
            f"in descending order.",
        ]

        code = (
            "result = (\n"
            f"    df.groupby('{group_col}', as_index=False)"
            f"['{metric_col}']\n"
            "      .sum()\n"
            f"      .sort_values('{metric_col}', "
            "ascending=False)\n"
            ")"
        )

        sql = (
            f"SELECT {group_col}, "
            f"SUM({metric_col}) AS revenue\n"
            f"FROM data\n"
            f"GROUP BY {group_col}\n"
            f"ORDER BY revenue DESC;"
        )

        chart = self._bar_chart(
            grouped,
            group_col,
            metric_col,
            title=(
                f"{metric_col.capitalize()} "
                f"by {group_col}"
            ),
        )

        return AnalysisResult(
            answer=answer,
            evidence=evidence,
            code=code,
            sql=sql,
            chart=chart,
        )

    # ========================================================
    # SINGLE METRIC
    # ========================================================

    def _single_metric_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult | None:

        if self._is_contextual_question(q):
            return None

        metric_col = self._metric_column(
            q,
            df,
        )

        if metric_col is None:
            return None

        q_lower = q.lower()

        # If an entity is explicitly mentioned,
        # let specific entity handler process it.
        if self._find_entity_column(
            q,
            df,
        ):
            return None

        metric_words = [
            "revenue",
            "sales",
            "amount",
            "profit",
            "income",
            "value",
            "arr",
            "roi",
            "gross",
            "ltv",
        ]

        if any(
            word in q_lower
            for word in metric_words
        ):

            total = float(
                pd.to_numeric(
                    df[metric_col],
                    errors="coerce",
                ).sum()
            )

            return AnalysisResult(
                answer=(
                    f"The total {metric_col} is "
                    f"{total:,.2f}."
                ),
                evidence=[
                    f"Calculated SUM({metric_col}) "
                    f"across all rows."
                ],
                code=(
                    f"result = "
                    f"df['{metric_col}'].sum()"
                ),
                sql=(
                    f"SELECT SUM({metric_col}) "
                    f"AS total "
                    f"FROM data;"
                ),
            )

        return None

    # ========================================================
    # MONTHLY TREND
    # ========================================================

    def _monthly_trend_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult | None:

        metric_col = self._metric_column(
            q,
            df,
        )

        if metric_col is None:
            return None

        date_col = None

        for col in df.columns:

            if (
                pd.api.types.is_datetime64_any_dtype(
                    df[col]
                )
                or "date" in str(col).lower()
                or "time" in str(col).lower()
            ):
                date_col = col
                break

        if date_col is None:

            return AnalysisResult(
                answer=(
                    "No date column was found "
                    "for a monthly trend."
                ),
                evidence=[
                    "Checked the dataset for a date "
                    "or time column."
                ],
            )

        dates = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

        if dates.notna().sum() == 0:

            return AnalysisResult(
                answer=(
                    "The date column could not be parsed."
                ),
                evidence=[
                    f"Attempted to parse `{date_col}` as dates."
                ],
            )

        temp = df.copy()

        temp["_parsed_date"] = dates

        temp["_month"] = (
            temp["_parsed_date"]
            .dt.to_period("M")
            .astype(str)
        )

        grouped = (
            temp.groupby(
                "_month"
            )[metric_col]
            .sum()
            .reset_index()
            .sort_values(
                "_month"
            )
        )

        if grouped.empty:
            return None

        parts = [
            f"{row['_month']}: "
            f"{float(row[metric_col]):,.2f}"
            for _, row in grouped.iterrows()
        ]

        # SQL is intentionally written as a readable
        # analytical query. The app displays it as generated
        # SQL; it does not execute this SQL directly.
        sql = (
            f"SELECT "
            f"DATE_TRUNC('month', {date_col}) AS month, "
            f"SUM({metric_col}) AS revenue\n"
            f"FROM data\n"
            f"GROUP BY DATE_TRUNC('month', {date_col})\n"
            f"ORDER BY month;"
        )

        return AnalysisResult(
            answer=(
                f"Monthly {metric_col} trend: "
                + " | ".join(parts)
                + "."
            ),
            evidence=[
                f"Parsed `{date_col}` as dates.",
                f"Grouped `{metric_col}` by month.",
                f"Calculated SUM({metric_col}) "
                f"for each month.",
            ],
            code=(
                "result = (\n"
                f"    df.assign(month="
                f"pd.to_datetime(df['{date_col}'], "
                f"errors='coerce')"
                ".dt.to_period('M').astype(str))\n"
                f"      .groupby('month')['{metric_col}']\n"
                "      .sum()\n"
                "      .reset_index()\n"
                "      .sort_values('month')\n"
                ")"
            ),
            sql=sql,
            chart=self._bar_chart(
                grouped,
                "_month",
                metric_col,
                title=(
                    f"Monthly "
                    f"{metric_col.capitalize()} Trend"
                ),
            ),
        )

    # ========================================================
    # PREVIOUS ASSISTANT ANSWER
    # ========================================================

    def _get_previous_assistant_answer(
        self,
    ) -> str | None:

        for message in reversed(
            self.history
        ):

            if not isinstance(
                message,
                dict,
            ):
                continue

            if message.get(
                "role"
            ) != "assistant":
                continue

            content = message.get(
                "content"
            )

            if isinstance(
                content,
                str,
            ):
                return content

            if isinstance(
                content,
                dict,
            ):

                answer = content.get(
                    "answer"
                )

                if isinstance(
                    answer,
                    str,
                ):
                    return answer

        return None

    # ========================================================
    # ENTITY FROM PREVIOUS ANSWER
    # ========================================================

    def _extract_entity_from_previous_answer(
        self,
        previous_answer: str,
        df: pd.DataFrame,
    ) -> tuple[Any | None, str | None]:

        answer_lower = (
            previous_answer.lower()
        )

        preferred = [
            "region",
            "customer",
            "product",
            "category",
            "segment",
            "city",
            "country",
            "department",
            "salesperson",
            "employee",
        ]

        # Preferred business columns first.
        for candidate in preferred:

            col = self._find_column(
                df,
                [candidate],
            )

            if col is None:
                continue

            for value in (
                df[col]
                .dropna()
                .unique()
            ):

                value_str = (
                    str(value)
                    .strip()
                )

                if not value_str:
                    continue

                if re.search(
                    rf"\b{re.escape(value_str.lower())}\b",
                    answer_lower,
                ):
                    return value, col

        # Other categorical columns.
        for col in self._categorical_columns(
            df
        ):

            col_lower = str(
                col
            ).lower()

            if (
                "date" in col_lower
                or "time" in col_lower
                or "timestamp" in col_lower
            ):
                continue

            for value in (
                df[col]
                .dropna()
                .unique()
            ):

                value_str = (
                    str(value)
                    .strip()
                )

                if not value_str:
                    continue

                if re.search(
                    rf"\b{re.escape(value_str.lower())}\b",
                    answer_lower,
                ):
                    return value, col

        return None, None

    # ========================================================
    # CONTEXT PARSER
    # ========================================================

    def _parse_context_query(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> dict[str, Any]:

        q_clean = re.sub(
            r"\s+",
            " ",
            q.strip(),
        )

        q_lower = q_clean.lower()

        # ----------------------------------------------------
        # Explicit entity forms
        #
        # What about East?
        # What about C6?
        # And South?
        # And what about North?
        # ----------------------------------------------------

        explicit_patterns = [
            r"^what\s+about\s+(?P<entity>.+?)\??$",
            r"^and\s+what\s+about\s+(?P<entity>.+?)\??$",
            r"^and\s+(?P<entity>.+?)\??$",
        ]

        for pattern in explicit_patterns:

            match = re.match(
                pattern,
                q_lower,
            )

            if not match:
                continue

            entity_text = (
                match.group(
                    "entity"
                )
                .strip()
                .rstrip(" ?!.")
            )

            entity_text = re.sub(
                r"^(that|this|the)\s+",
                "",
                entity_text,
            ).strip()

            if not entity_text:

                return {
                    "type": "none"
                }

            entity_col = (
                self._find_entity_column(
                    entity_text,
                    df,
                )
            )

            if entity_col is not None:

                entity_value = None

                for value in (
                    df[entity_col]
                    .dropna()
                    .unique()
                ):

                    value_str = (
                        str(value)
                        .strip()
                    )

                    if (
                        value_str.lower()
                        ==
                        entity_text.lower()
                    ):
                        entity_value = value
                        break

                if entity_value is None:

                    for value in (
                        df[entity_col]
                        .dropna()
                        .unique()
                    ):

                        value_str = (
                            str(value)
                            .strip()
                        )

                        if re.search(
                            rf"\b{re.escape(value_str.lower())}\b",
                            entity_text.lower(),
                        ):
                            entity_value = value
                            break

                if entity_value is not None:

                    return {
                        "type": "explicit_entity",
                        "entity": entity_value,
                        "column": entity_col,
                    }

            return {
                "type": "unknown_entity",
                "entity_text": entity_text,
            }

        # ----------------------------------------------------
        # Pronoun forms
        # ----------------------------------------------------

        pronoun_patterns = [
            r"\bthat\s+(?P<dimension>region|customer|product|category|segment|city|country)\b",
            r"\bthis\s+(?P<dimension>region|customer|product|category|segment|city|country)\b",
            r"\bthe\s+(?P<dimension>region|customer|product|category|segment|city|country)\b",
            r"\bthat\s+one\b",
            r"\bthis\s+one\b",
        ]

        for pattern in pronoun_patterns:

            match = re.search(
                pattern,
                q_lower,
            )

            if match:

                dimension = (
                    match.groupdict()
                    .get("dimension")
                    if match.groupdict()
                    else None
                )

                return {
                    "type": "pronoun_reference",
                    "dimension": dimension,
                }

        return {
            "type": "none"
        }

    # ========================================================
    # CONTEXT RESULT
    # ========================================================

    def _context_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult | None:

        revenue_col = self._metric_column(
            q,
            df,
        )

        if revenue_col is None:
            return None

        parsed = self._parse_context_query(
            q,
            df,
        )

        parse_type = parsed.get(
            "type"
        )

        # ====================================================
        # EXPLICIT ENTITY
        #
        # What about East?
        # What about C6?
        # ====================================================

        if parse_type == "explicit_entity":

            entity_value = parsed[
                "entity"
            ]

            entity_col = parsed[
                "column"
            ]

            mask = (
                df[entity_col]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(entity_value)
                .strip()
                .lower()
            )

            total = float(
                pd.to_numeric(
                    df.loc[
                        mask,
                        revenue_col,
                    ],
                    errors="coerce",
                ).sum()
            )

            return AnalysisResult(
                answer=(
                    f"{entity_value} generated "
                    f"{revenue_col} of "
                    f"{total:,.2f}."
                ),
                evidence=[
                    f"Identified `{entity_col}` value "
                    f"`{entity_value}` from the contextual question.",
                    f"Filtered `{entity_col}` for `{entity_value}`.",
                    f"Calculated SUM({revenue_col}) "
                    f"for the matching rows.",
                ],
                code=(
                    "result = (\n"
                    f"    df[df['{entity_col}'].astype(str).str.lower() "
                    f"== '{str(entity_value).lower()}']"
                    f"['{revenue_col}']\n"
                    "    .sum()\n"
                    ")"
                ),
                sql=(
                    f"SELECT SUM({revenue_col}) AS revenue "
                    f"FROM data "
                    f"WHERE LOWER({entity_col}) = "
                    f"'{str(entity_value).lower()}';"
                ),
                chart=self._bar_chart(
                    df.loc[
                        mask
                    ],
                    entity_col,
                    revenue_col,
                    title=(
                        f"Revenue for "
                        f"{entity_value}"
                    ),
                ),
            )

        # ====================================================
        # UNKNOWN ENTITY
        # ====================================================

        if parse_type == "unknown_entity":

            entity_text = parsed.get(
                "entity_text",
                "",
            )

            return AnalysisResult(
                answer=(
                    f"I could not find `{entity_text}` "
                    f"in the available dataset."
                ),
                evidence=[
                    f"Searched categorical columns "
                    f"for `{entity_text}`."
                ],
            )

        # ====================================================
        # PRONOUN REFERENCE
        #
        # How much revenue did that region generate?
        # ====================================================

        if parse_type == "pronoun_reference":

            previous_answer = (
                self._get_previous_assistant_answer()
            )

            if not previous_answer:
                return None

            (
                entity_value,
                entity_col,
            ) = (
                self._extract_entity_from_previous_answer(
                    previous_answer,
                    df,
                )
            )

            if (
                entity_value is None
                or entity_col is None
            ):
                return None

            dimension = parsed.get(
                "dimension"
            )

            if dimension:

                expected_col = (
                    self._find_column(
                        df,
                        [dimension],
                    )
                )

                if (
                    expected_col is not None
                    and expected_col != entity_col
                ):

                    found = False

                    for value in (
                        df[expected_col]
                        .dropna()
                        .unique()
                    ):

                        if (
                            str(value)
                            .strip()
                            .lower()
                            ==
                            str(entity_value)
                            .strip()
                            .lower()
                        ):

                            entity_col = (
                                expected_col
                            )

                            entity_value = (
                                value
                            )

                            found = True
                            break

                    if not found:
                        return None

            mask = (
                df[entity_col]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                str(entity_value)
                .strip()
                .lower()
            )

            total = float(
                pd.to_numeric(
                    df.loc[
                        mask,
                        revenue_col,
                    ],
                    errors="coerce",
                ).sum()
            )

            return AnalysisResult(
                answer=(
                    f"{entity_value} generated "
                    f"{revenue_col} of "
                    f"{total:,.2f}."
                ),
                evidence=[
                    f"Used the previous answer "
                    f"to identify `{entity_value}`.",
                    f"Filtered `{entity_col}` "
                    f"for `{entity_value}`.",
                    f"Calculated SUM({revenue_col}) "
                    f"for the matching rows.",
                ],
                code=(
                    "result = (\n"
                    f"    df[df['{entity_col}'].astype(str).str.lower() "
                    f"== '{str(entity_value).lower()}']"
                    f"['{revenue_col}']\n"
                    "    .sum()\n"
                    ")"
                ),
                sql=(
                    f"SELECT SUM({revenue_col}) AS revenue "
                    f"FROM data "
                    f"WHERE LOWER({entity_col}) = "
                    f"'{str(entity_value).lower()}';"
                ),
                chart=self._bar_chart(
                    df.loc[
                        mask
                    ],
                    entity_col,
                    revenue_col,
                    title=(
                        f"Revenue for "
                        f"{entity_value}"
                    ),
                ),
            )

        return None

    # ========================================================
    # BAR CHART
    # ========================================================

    def _bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
    ):

        if (
            df is None
            or df.empty
        ):
            return None

        try:

            fig = px.bar(
                df,
                x=x,
                y=y,
                title=title,
                text_auto=".2f",
            )

            fig.update_layout(
                template="simple_white",
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
            )

            return fig

        except Exception:
            return None

    # ========================================================
    # SQL
    # ========================================================

    def _sql_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        metric_col = self._metric_column(
            q,
            df,
        )

        if metric_col is None:

            return AnalysisResult(
                answer=(
                    "No suitable numeric metric "
                    "column was found."
                ),
                evidence=[],
            )

        group_col = self._group_column(
            q,
            df,
        )

        q_lower = q.lower()

        is_top = (
            "top" in q_lower
            or "highest" in q_lower
            or "largest" in q_lower
            or "best" in q_lower
        )

        is_bottom = (
            "bottom" in q_lower
            or "lowest" in q_lower
            or "smallest" in q_lower
            or "worst" in q_lower
        )

        if is_top or is_bottom:

            n = self._extract_n(
                q,
                default=5,
            )

            if group_col:

                order = (
                    "ASC"
                    if is_bottom
                    else "DESC"
                )

                sql = (
                    f"SELECT {group_col}, "
                    f"SUM({metric_col}) AS revenue\n"
                    f"FROM data\n"
                    f"GROUP BY {group_col}\n"
                    f"ORDER BY revenue {order}\n"
                    f"LIMIT {n};"
                )

            else:

                sql = (
                    f"SELECT SUM({metric_col}) "
                    f"AS revenue\n"
                    f"FROM data;"
                )

        else:

            sql = (
                f"SELECT SUM({metric_col}) "
                f"AS revenue\n"
                f"FROM data;"
            )

        return AnalysisResult(
            answer=(
                "Generated SQL for the "
                "requested analysis."
            ),
            evidence=[
                "Mapped the natural-language "
                "request to SQL.",
                "Used the detected dataset columns.",
            ],
            sql=sql,
        )

    # ========================================================
    # PANDAS CODE
    # ========================================================

    def _code_result(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        metric_col = self._metric_column(
            q,
            df,
        )

        group_col = self._group_column(
            q,
            df,
        )

        q_lower = q.lower()

        if metric_col is None:

            return AnalysisResult(
                answer=(
                    "No suitable numeric metric "
                    "column was found."
                ),
                evidence=[],
            )

        if (
            (
                "top" in q_lower
                or "highest" in q_lower
            )
            and group_col
        ):

            n = self._extract_n(
                q,
                default=5,
            )

            code = (
                "result = (\n"
                f"    df.groupby('{group_col}', "
                f"as_index=False)"
                f"['{metric_col}']\n"
                "      .sum()\n"
                f"      .sort_values('{metric_col}', "
                "ascending=False)\n"
                f"      .head({n})\n"
                ")"
            )

        elif (
            (
                "bottom" in q_lower
                or "lowest" in q_lower
                or "smallest" in q_lower
                or "worst" in q_lower
            )
            and group_col
        ):

            n = self._extract_n(
                q,
                default=5,
            )

            code = (
                "result = (\n"
                f"    df.groupby('{group_col}', "
                f"as_index=False)"
                f"['{metric_col}']\n"
                "      .sum()\n"
                f"      .sort_values('{metric_col}', "
                "ascending=True)\n"
                f"      .head({n})\n"
                ")"
            )

        elif (
            "average" in q_lower
            or "mean" in q_lower
        ):

            code = (
                f"result = "
                f"df['{metric_col}'].mean()"
            )

        else:

            code = (
                f"result = "
                f"df['{metric_col}'].sum()"
            )

        return AnalysisResult(
            answer=(
                "Generated Pandas code "
                "for the requested analysis."
            ),
            evidence=[
                "Mapped the natural-language "
                "request to Pandas.",
                "Used the detected dataset columns.",
            ],
            code=code,
        )

    # ========================================================
    # LLM FALLBACK
    # ========================================================

    def _llm_fallback(
        self,
        q: str,
        df: pd.DataFrame,
    ) -> AnalysisResult:

        if self.llm is None:

            return AnalysisResult(
                answer=(
                    "I could not determine the requested "
                    "analysis from the available dataset."
                ),
                evidence=[
                    "The question did not match a supported "
                    "deterministic analysis pattern."
                ],
            )

        try:

            schema = self._schema_for_llm(
                df
            )

            prompt = f"""
You are a data analyst.

Answer the user's question using ONLY
the provided dataset schema.

Dataset schema:
{schema}

User question:
{q}

If the question cannot be answered
from the schema, say that clearly.

Do not invent numbers.
"""

            response = self.llm.generate(
                prompt
            )

            if response:

                return AnalysisResult(
                    answer=str(
                        response
                    ),
                    evidence=[
                        "Used the language model as "
                        "a fallback for an unsupported "
                        "deterministic pattern."
                    ],
                )

        except Exception:
            pass

        return AnalysisResult(
            answer=(
                "I could not determine the requested "
                "analysis from the available dataset."
            ),
            evidence=[
                "The question did not match a supported "
                "deterministic analysis pattern."
            ],
        )

    # ========================================================
    # LLM SCHEMA
    # ========================================================

    def _schema_for_llm(
        self,
        df: pd.DataFrame,
    ) -> str:

        lines = []

        for col in df.columns:

            dtype = str(
                df[col].dtype
            )

            lines.append(
                f"- {col}: {dtype}"
            )

        return "\n".join(
            lines
        )