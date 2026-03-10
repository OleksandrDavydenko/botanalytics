from __future__ import annotations

import io
import zipfile
import warnings
import os
from textwrap import fill
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 130

REQUIRED_COLS = {"id", "user_id", "username", "action", "timestamp", "message_id"}

MONTHS_UA = {
    "січень", "лютий", "березень", "квітень", "травень", "червень",
    "липень", "серпень", "вересень", "жовтень", "листопад", "грудень",
}
MONTHS_EN = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
NAVIGATION_ACTIONS_DEFAULT = {
    "Головне меню",
    "Назад",
    "/menu",
    "/start",
}
INTERNAL_EXCLUDED_USERS = {
    "Давиденко Олександр",
    "Ступа Олександр",
}


def resolve_report_password() -> str:
    try:
        secret_password = st.secrets.get("PASSWORD")
    except Exception:
        # Streamlit throws if secrets.toml is missing in local runs.
        secret_password = None

    if secret_password and str(secret_password).strip():
        return str(secret_password).strip()

    return os.getenv("REPORT_PASSWORD", "").strip()


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def wrap_labels(labels, width=24):
    return [fill(str(x), width=width) for x in labels]


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.getvalue()


def create_zip_bytes(
    excel_bytes: bytes,
    report_text: str,
    chart_pngs: dict[str, bytes],
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analysis_tables.xlsx", excel_bytes)
        zf.writestr("insights_report.txt", report_text.encode("utf-8"))
        for filename, png in chart_pngs.items():
            zf.writestr(filename, png)
    buf.seek(0)
    return buf.getvalue()


def cramers_v_from_contingency(contingency: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = max(min((kcorr - 1), (rcorr - 1)), 1e-9)
    return float(np.sqrt(phi2corr / denom))


def parse_multiline_set(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


def get_excel_sheet_names(file_bytes: bytes) -> list[str]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return list(xls.sheet_names)


def read_excel_bytes_from_url(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Підтримуються лише http/https посилання на файл.")
    with urlopen(url, timeout=60) as response:
        return response.read()


def resolve_data_source_url() -> str:
    params = st.query_params
    if "data" in params and str(params["data"]).strip():
        return str(params["data"]).strip()

    secrets_url = None
    try:
        secrets_url = st.secrets.get("DATASET_URL")
    except Exception:
        # Streamlit throws if secrets.toml is missing; that's valid in local runs.
        secrets_url = None

    if secrets_url and str(secrets_url).strip():
        return str(secrets_url).strip()

    env_url = os.getenv("DATASET_URL", "").strip()
    return env_url


def load_default_local_excel_bytes() -> bytes | None:
    default_path = "data.xlsx"
    if not os.path.exists(default_path):
        return None
    with open(default_path, "rb") as f:
        return f.read()


def load_data(file_bytes: bytes, sheet_name):
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"У файлі немає обов'язкових колонок: {sorted(missing)}")
    return df


def prepare_data(
    df_raw: pd.DataFrame,
    excluded_users: set[str],
    exclude_month_year: bool,
    exclude_navigation_actions: bool,
    navigation_actions: set[str],
    exclude_phone_events: bool,
    session_gap_minutes: int,
):
    df = df_raw.copy()
    raw = df.copy()

    df["username"] = normalize_text(df["username"])
    df["action"] = normalize_text(df["action"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp", "action", "username", "user_id"]).copy()

    if excluded_users:
        df = df[~df["username"].isin(excluded_users)].copy()

    if exclude_month_year:
        action_lower = df["action"].str.lower()
        is_month = action_lower.isin(MONTHS_UA | MONTHS_EN)
        is_year = df["action"].str.fullmatch(r"20\d{2}", na=False)
        df = df[~(is_month | is_year)].copy()

    if exclude_phone_events:
        phone_mask = df["action"].str.startswith("Надано номер телефону:", na=False)
        df = df[~phone_mask].copy()

    if exclude_navigation_actions and navigation_actions:
        df = df[~df["action"].isin(navigation_actions)].copy()

    df = df.sort_values(["user_id", "timestamp", "id"]).reset_index(drop=True)
    df["date"] = df["timestamp"].dt.floor("D")
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["day_of_month"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["weekday_num"] = df["timestamp"].dt.weekday
    weekday_map = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Нд"}
    df["weekday"] = df["weekday_num"].map(weekday_map)
    df["is_weekend"] = df["weekday_num"].isin([5, 6])

    sessions = build_sessions(df, session_gap_minutes)
    return raw, df, sessions


def build_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    x = df.sort_values(["user_id", "timestamp", "id"]).copy()
    prev_user = x["user_id"].shift(1)
    prev_ts = x["timestamp"].shift(1)

    is_new_user = x["user_id"] != prev_user
    gap = (x["timestamp"] - prev_ts).dt.total_seconds().div(60)
    is_new_session = is_new_user | gap.gt(gap_minutes) | gap.isna()

    x["session_num"] = is_new_session.cumsum()
    sessions = (
        x.groupby(["user_id", "session_num"], as_index=False)
        .agg(
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            events=("id", "count"),
            unique_actions=("action", "nunique"),
            first_action=("action", "first"),
            last_action=("action", "last"),
        )
    )
    sessions["duration_min"] = (
        (sessions["session_end"] - sessions["session_start"]).dt.total_seconds() / 60
    ).clip(lower=0)
    sessions["date"] = sessions["session_start"].dt.floor("D")
    return sessions


def get_calendar_day_occurrences(df: pd.DataFrame) -> pd.Series:
    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return all_days.to_series().dt.day.value_counts().sort_index()


def get_calendar_weekday_occurrences(df: pd.DataFrame) -> pd.Series:
    all_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    weekday_map = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Нд"}
    return all_days.to_series().dt.weekday.map(weekday_map).value_counts().reindex(
        ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    )


def create_summary_tables(df: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary = pd.DataFrame(
        {
            "metric": [
                "Період від",
                "Період до",
                "Кількість подій",
                "Унікальні користувачі",
                "Унікальні функції",
                "Середня кількість подій на день",
                "Медіанна кількість подій на день",
                "Середня кількість активних користувачів на день",
                "Кількість сесій",
                "Середня тривалість сесії, хв",
                "Медіанна тривалість сесії, хв",
                "Середня кількість подій у сесії",
            ],
            "value": [
                df["timestamp"].min(),
                df["timestamp"].max(),
                len(df),
                df["user_id"].nunique(),
                df["action"].nunique(),
                round(df.groupby("date").size().mean(), 2),
                round(df.groupby("date").size().median(), 2),
                round(df.groupby("date")["user_id"].nunique().mean(), 2),
                len(sessions),
                round(sessions["duration_min"].mean(), 2),
                round(sessions["duration_min"].median(), 2),
                round(sessions["events"].mean(), 2),
            ],
        }
    )

    action_popularity = (
        df.groupby("action")
        .agg(
            events=("id", "count"),
            users=("user_id", "nunique"),
            active_days=("date", "nunique"),
        )
        .sort_values(["events", "users"], ascending=False)
        .reset_index()
    )
    action_popularity["events_share_%"] = (action_popularity["events"] / len(df) * 100).round(2)
    action_popularity["users_share_%"] = (
        action_popularity["users"] / max(df["user_id"].nunique(), 1) * 100
    ).round(2)

    users_activity = (
        df.groupby(["user_id", "username"])
        .agg(
            events=("id", "count"),
            unique_actions=("action", "nunique"),
            active_days=("date", "nunique"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .sort_values("events", ascending=False)
        .reset_index()
    )

    daily_usage = (
        df.groupby("date")
        .agg(
            events=("id", "count"),
            users=("user_id", "nunique"),
            actions=("action", "nunique"),
        )
        .reset_index()
    )
    daily_usage["year_month"] = daily_usage["date"].dt.to_period("M").astype(str)

    monthly_usage = (
        df.groupby("year_month")
        .agg(
            events=("id", "count"),
            users=("user_id", "nunique"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )
    monthly_usage["avg_events_per_active_day"] = (
        monthly_usage["events"] / monthly_usage["active_days"]
    ).round(2)

    session_summary = (
        sessions.groupby("date")
        .agg(
            sessions=("session_num", "count"),
            avg_duration_min=("duration_min", "mean"),
            avg_events_per_session=("events", "mean"),
        )
        .reset_index()
    )

    entry_actions = (
        sessions["first_action"].value_counts().rename_axis("action").reset_index(name="sessions_started")
    )

    return {
        "summary": summary,
        "action_popularity": action_popularity,
        "users_activity": users_activity,
        "daily_usage": daily_usage,
        "monthly_usage": monthly_usage,
        "session_summary": session_summary,
        "entry_actions": entry_actions,
    }


def analyze_day_of_month(df: pd.DataFrame):
    day_occ = get_calendar_day_occurrences(df)

    daily_events = df.groupby("date").size().rename("events").reset_index()
    daily_events["day_of_month"] = daily_events["date"].dt.day
    dom_events = daily_events.groupby("day_of_month")["events"].agg(["sum", "mean", "median"]).reset_index()
    dom_events["calendar_occurrences"] = dom_events["day_of_month"].map(day_occ)
    dom_events["avg_events_per_calendar_day"] = (
        dom_events["sum"] / dom_events["calendar_occurrences"]
    ).round(2)

    daily_users = df.groupby("date")["user_id"].nunique().rename("users").reset_index()
    daily_users["day_of_month"] = daily_users["date"].dt.day
    dom_users = daily_users.groupby("day_of_month")["users"].agg(["sum", "mean", "median"]).reset_index()
    dom_users["calendar_occurrences"] = dom_users["day_of_month"].map(day_occ)
    dom_users["avg_users_per_calendar_day"] = (
        dom_users["sum"] / dom_users["calendar_occurrences"]
    ).round(2)

    dom = dom_events.merge(dom_users[["day_of_month", "avg_users_per_calendar_day"]], on="day_of_month", how="left")
    dom = dom.sort_values("day_of_month")
    return dom, day_occ.rename_axis("day_of_month").reset_index(name="calendar_occurrences")


def analyze_weekday(df: pd.DataFrame):
    weekday_order = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    weekday_occ = get_calendar_weekday_occurrences(df)

    daily_events = df.groupby("date").size().rename("events").reset_index()
    daily_events["weekday"] = daily_events["date"].dt.weekday.map(
        {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Нд"}
    )
    wk = daily_events.groupby("weekday")["events"].agg(["sum", "mean", "median"]).reindex(weekday_order)
    wk["calendar_occurrences"] = weekday_occ
    wk["avg_events_per_weekday"] = (wk["sum"] / wk["calendar_occurrences"]).round(2)
    return wk.reset_index()


def build_weekday_hour_heatmap(df: pd.DataFrame):
    base = df.groupby(["date", "weekday_num", "hour"]).size().rename("events").reset_index()
    heat = (
        base.groupby(["weekday_num", "hour"])["events"]
        .mean()
        .unstack(fill_value=0)
        .reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=range(24), fill_value=0)
    )
    heat.index = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    return heat


def analyze_action_day_relationship(df: pd.DataFrame, min_action_count: int = 25, top_n_actions: int = 12):
    action_counts = df["action"].value_counts()
    selected_actions = action_counts[action_counts >= min_action_count].head(top_n_actions).index.tolist()
    sub = df[df["action"].isin(selected_actions)].copy()

    if sub.empty:
        lift = pd.DataFrame(index=[], columns=range(1, 32))
        chi_table = pd.DataFrame({"metric": ["chi2", "p_value", "cramers_v", "n_actions", "n_rows"], "value": [np.nan] * 5})
        peaks = pd.DataFrame(columns=["action", "day_of_month", "lift"])
        return lift, chi_table, peaks

    daily_action = sub.groupby(["date", "action"]).size().rename("events").reset_index()
    daily_action["day_of_month"] = daily_action["date"].dt.day

    avg_by_dom_action = (
        daily_action.groupby(["action", "day_of_month"])["events"].mean().unstack(fill_value=0)
    )
    overall_avg_by_action = daily_action.groupby("action")["events"].mean()

    lift = avg_by_dom_action.div(overall_avg_by_action, axis=0)
    lift = lift.replace([np.inf, -np.inf], np.nan).fillna(0)
    lift = lift.reindex(index=selected_actions, columns=range(1, 32), fill_value=0)

    bins = [0, 5, 10, 15, 20, 25, 31]
    labels = ["01-05", "06-10", "11-15", "16-20", "21-25", "26-31"]
    sub["dom_bucket"] = pd.cut(sub["day_of_month"], bins=bins, labels=labels)
    contingency = pd.crosstab(sub["action"], sub["dom_bucket"])

    chi2, p_value, _, _ = chi2_contingency(contingency)
    cramers_v = cramers_v_from_contingency(contingency)

    chi_table = pd.DataFrame(
        {
            "metric": ["chi2", "p_value", "cramers_v", "n_actions", "n_rows"],
            "value": [chi2, p_value, cramers_v, contingency.shape[0], int(contingency.values.sum())],
        }
    )

    peaks = (
        lift.stack()
        .rename("lift")
        .reset_index()
        .rename(columns={"level_1": "day_of_month"})
        .sort_values("lift", ascending=False)
    )
    peaks = peaks[peaks["lift"] > 1.15].copy()
    return lift, chi_table, peaks


def analyze_transitions(df: pd.DataFrame, session_gap_minutes: int = 30):
    x = df.sort_values(["user_id", "timestamp", "id"]).copy()
    x["next_action"] = x.groupby("user_id")["action"].shift(-1)
    next_gap = (
        x.groupby("user_id")["timestamp"].shift(-1) - x["timestamp"]
    ).dt.total_seconds().div(60)
    x["same_session_forward"] = next_gap.le(session_gap_minutes)

    transitions = x.loc[x["same_session_forward"] & x["next_action"].notna(), ["action", "next_action"]].copy()
    trans = (
        transitions.groupby(["action", "next_action"]).size().rename("count").reset_index()
        .sort_values("count", ascending=False)
    )

    if trans.empty:
        return trans

    next_base = transitions["next_action"].value_counts()
    total_trans = len(transitions)

    trans["p_transition"] = trans["count"] / total_trans
    trans["p_next_action_base"] = trans["next_action"].map(next_base / total_trans)
    trans["lift"] = (trans["p_transition"] / trans["p_next_action_base"]).round(3)
    return trans


def build_text_insights(df, tables, dom, weekday_df, chi_table, peaks, transitions):
    action_popularity = tables["action_popularity"]
    users_activity = tables["users_activity"]

    if action_popularity.empty:
        return "Після фільтрації не залишилось даних для побудови інсайтів."

    top_action = action_popularity.iloc[0]
    peak_dom = dom.sort_values("avg_events_per_calendar_day", ascending=False).iloc[0]
    peak_weekday = weekday_df.sort_values("avg_events_per_weekday", ascending=False).iloc[0]
    hourly = df.groupby("hour").size().sort_values(ascending=False)
    peak_hour = int(hourly.index[0])
    peak_hour_events = int(hourly.iloc[0])

    top5_event_share = action_popularity.head(5)["events"].sum() / max(len(df), 1) * 100
    top10_users_share = users_activity.head(10)["events"].sum() / max(len(df), 1) * 100

    cramers_v = float(chi_table.loc[chi_table["metric"] == "cramers_v", "value"].iloc[0])
    p_value = float(chi_table.loc[chi_table["metric"] == "p_value", "value"].iloc[0])

    lines = []
    lines.append("Продуктовий аналітичний бриф")
    lines.append(f"Період: {df['timestamp'].min()} — {df['timestamp'].max()}")
    lines.append(f"Подій: {len(df):,}".replace(",", " "))
    lines.append(f"Активних користувачів: {df['user_id'].nunique()}")
    lines.append(f"Кількість функцій у використанні: {df['action'].nunique()}")
    lines.append("")
    lines.append("1) Драйвери використання")
    lines.append(
        f"   Ключова функція: '{top_action['action']}' — {int(top_action['events'])} подій, "
        f"охоплення {top_action['users_share_%']}% аудиторії."
    )
    lines.append(f"   ТОП-5 функцій формують {top5_event_share:.1f}% взаємодій, тобто ядро цінності зосереджене у кількох сценаріях.")
    lines.append("")
    lines.append("2) Часові патерни попиту")
    lines.append(
        f"   Пік за числом місяця: {int(peak_dom['day_of_month'])} — "
        f"{peak_dom['avg_events_per_calendar_day']:.2f} подій у середньому на календарний день."
    )
    lines.append(
        f"   Пік за днем тижня: {peak_weekday['weekday']} — "
        f"{peak_weekday['avg_events_per_weekday']:.2f} подій у середньому."
    )
    lines.append(f"   Найактивніше впродовж доби: {peak_hour:02d}:00 ({peak_hour_events} подій).")
    lines.append("")
    lines.append("3) Сегментація попиту за календарем")
    if pd.notna(p_value):
        lines.append(
            f"   Статистично: p-value = {p_value:.6f}; Cramér's V = {cramers_v:.3f}."
        )
    else:
        lines.append("   Недостатньо даних для статистичної оцінки.")
    if not peaks.empty:
        lines.append("   Найсильніші календарні сплески:")
        for _, row in peaks.head(8).iterrows():
            lines.append(
                f"   - {row['action']} → {int(row['day_of_month'])} число, lift={row['lift']:.2f}x"
            )
    lines.append("")
    lines.append("4) Концентрація активності")
    lines.append(f"   ТОП-10 користувачів формують {top10_users_share:.1f}% усіх подій: це ознака високої залежності від вузького ядра.")
    if not transitions.empty:
        top_transition = transitions.iloc[0]
        lines.append("")
        lines.append("5) Поведінковий сценарій у сесії")
        lines.append(
            f"   Найтиповіший перехід: {top_transition['action']} → {top_transition['next_action']} "
            f"({int(top_transition['count'])} разів, lift={top_transition['lift']:.2f})."
        )
    return "\n".join(lines)


def get_terms_explanation_text() -> str:
    lines = []
    lines.append("Пояснення ключових термінів")
    lines.append("1) p-value")
    lines.append("   Показує, наскільки ймовірно отримати такий результат випадково.")
    lines.append("   Просте правило: чим менше значення, тим менше шанс, що це випадковість.")
    lines.append("   Орієнтир: p-value < 0.05 зазвичай вважають статистично значущим результатом.")
    lines.append("")
    lines.append("2) Cramér's V")
    lines.append("   Показує силу зв'язку між двома категоріями (у нас: функція і період календаря).")
    lines.append("   Значення від 0 до 1: 0 = зв'язку майже немає, 1 = дуже сильний зв'язок.")
    lines.append("   Практичний орієнтир: ~0.1 слабкий, ~0.3 середній, ~0.5+ сильний.")
    lines.append("")
    lines.append("3) lift")
    lines.append("   Коефіцієнт відхилення від середнього рівня.")
    lines.append("   lift = 1.0: як зазвичай; >1.0: вище норми; <1.0: нижче норми.")
    lines.append("   Приклад: lift=3.0 означає, що подія трапляється приблизно в 3 рази частіше за свій середній рівень.")
    lines.append("")
    lines.append("4) Share %")
    lines.append("   Частка подій конкретного користувача від усіх подій у вибірці.")
    lines.append("   Приклад: Share %=12.5 означає, що користувач сформував 12.5% усіх подій.")
    lines.append("")
    lines.append("5) HHI користувачів")
    lines.append("   Індекс концентрації активності між користувачами.")
    lines.append("   Формула: HHI = sum((share_i)^2) * 10 000, де share_i - частка подій i-го користувача (від 0 до 1).")
    lines.append("   Як читати: чим більше HHI, тим сильніше події зосереджені у невеликої групи користувачів.")
    lines.append("   Приклад: якщо частки 50%, 30%, 20%, тоді HHI = (0.5^2 + 0.3^2 + 0.2^2) * 10 000 = 3 800.")
    lines.append("")
    lines.append("6) Пік/сплеск")
    lines.append("   День або період, де метрика суттєво вища за звичний рівень.")
    lines.append("   Це сигнал, коли користувачам найбільше потрібен конкретний сценарій.")
    lines.append("")
    lines.append("7) Важливе обмеження")
    lines.append("   Статистичний зв'язок не доводить причину.")
    lines.append("   Тобто ми бачимо 'коли' зростає попит, але не завжди точно знаємо 'чому'.")
    lines.append("   Для підтвердження причин корисні експерименти або аналіз кількох періодів окремо.")
    return "\n".join(lines)


def plot_top_actions(action_popularity: pd.DataFrame, top_n: int = 15):
    plot_df = action_popularity.head(top_n).sort_values("events")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["action"], plot_df["events"])
    ax.set_title("ТОП функцій за кількістю використань")
    ax.set_xlabel("Кількість подій")
    ax.set_ylabel("")
    for i, v in enumerate(plot_df["events"]):
        ax.text(v + max(plot_df["events"]) * 0.01, i, str(v), va="center")
    return fig


def plot_monthly_usage(monthly_usage: pd.DataFrame):
    x = np.arange(len(monthly_usage))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, monthly_usage["events"], marker="o", label="Події")
    ax.plot(x, monthly_usage["users"], marker="o", label="Унікальні користувачі")
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_usage["year_month"], rotation=45, ha="right")
    ax.set_title("Динаміка використання по місяцях")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_day_of_month(dom: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dom["day_of_month"], dom["avg_events_per_calendar_day"], marker="o", label="Середня кількість подій")
    ax.plot(dom["day_of_month"], dom["avg_users_per_calendar_day"], marker="o", label="Середня кількість користувачів")
    ax.set_xticks(range(1, 32))
    ax.set_title("Використання за числами місяця")
    ax.set_xlabel("Число місяця")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_weekday_usage(weekday_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(weekday_df["weekday"], weekday_df["avg_events_per_weekday"])
    ax.set_title("Середня кількість подій за днем тижня")
    ax.set_ylabel("Середні події на календарний день")
    for i, v in enumerate(weekday_df["avg_events_per_weekday"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    return fig


def plot_weekday_hour_heatmap(heatmap_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap_df.values, aspect="auto")
    ax.set_title("Heatmap: день тижня × година")
    ax.set_xlabel("Година")
    ax.set_xticks(range(24))
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Середня кількість подій")
    return fig


def plot_entry_actions(entry_actions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df = entry_actions.head(12).sort_values("sessions_started")
    ax.barh(plot_df["action"], plot_df["sessions_started"])
    ax.set_title("Які функції найчастіше відкривають сесію")
    ax.set_xlabel("Кількість сесій")
    return fig


def plot_action_day_lift(lift: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 7))
    if lift.empty:
        ax.text(0.5, 0.5, "Недостатньо даних", ha="center", va="center")
        ax.axis("off")
        return fig
    im = ax.imshow(lift.values, aspect="auto")
    ax.set_title("Heatmap: функції по числах місяця (lift)")
    ax.set_xlabel("Число місяця")
    ax.set_xticks(range(31))
    ax.set_xticklabels(range(1, 32))
    ax.set_yticks(range(len(lift.index)))
    ax.set_yticklabels(wrap_labels(lift.index, width=24))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Lift (>1 = вище середнього)")
    return fig


def plot_top_transitions(transitions: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    if transitions.empty:
        ax.text(0.5, 0.5, "Недостатньо переходів", ha="center", va="center")
        ax.axis("off")
        return fig
    plot_df = transitions.head(12).iloc[::-1].copy()
    plot_df["pair"] = plot_df["action"] + " → " + plot_df["next_action"]
    ax.barh(plot_df["pair"], plot_df["count"])
    ax.set_title("Найпоширеніші переходи між функціями")
    ax.set_xlabel("Кількість переходів")
    return fig


def plot_user_concentration(users_activity: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = users_activity.sort_values("events", ascending=False).reset_index(drop=True).copy()
    x["user_rank"] = np.arange(1, len(x) + 1)
    x["cum_share_events_%"] = x["events"].cumsum() / max(x["events"].sum(), 1) * 100
    x["cum_share_users_%"] = x["user_rank"] / max(len(x), 1) * 100
    ax.plot(x["cum_share_users_%"], x["cum_share_events_%"], marker="o", markersize=2)
    ax.plot([0, 100], [0, 100], linestyle="--")
    ax.set_title("Концентрація активності користувачів (Pareto)")
    ax.set_xlabel("% користувачів")
    ax.set_ylabel("% усіх подій")
    ax.grid(alpha=0.25)
    return fig


def inject_modern_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: light;
            --ui-text-main: #0f172a;
            --ui-text-muted: #334155;
            --ui-surface: #ffffff;
            --ui-surface-soft: #f8fafc;
            --ui-border: #dbe7ff;
            --ui-accent: #1d4ed8;
        }
        .stApp {
            background: radial-gradient(circle at 10% 10%, #f4f8ff 0%, #f8fafc 45%, #ffffff 100%);
            color: var(--ui-text-main);
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp li, .stApp label, .stApp span,
        .stMarkdown, [data-testid="stMarkdownContainer"] {
            color: var(--ui-text-main);
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--ui-text-muted) !important;
        }
        .hero-wrap {
            padding: 1rem 1.2rem;
            border: 1px solid #dbe7ff;
            border-radius: 14px;
            background: linear-gradient(120deg, #edf4ff 0%, #f8fbff 100%);
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: 1.7rem;
            font-weight: 700;
            color: #113a7c;
            line-height: 1.25;
        }
        .hero-sub {
            margin-top: 0.35rem;
            color: #334155;
            font-size: 0.95rem;
        }
        .kpi-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.8rem;
            background: #ffffff;
            box-shadow: 0 6px 14px rgba(15, 23, 42, 0.04);
            min-height: 96px;
        }
        .kpi-label {
            color: #475569;
            font-size: 0.85rem;
        }
        .kpi-value {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }
        .login-wrap {
            margin: 0.4rem 0 0.8rem;
            padding: 0.9rem 1rem;
            border: 1px solid #dbe7ff;
            border-radius: 12px;
            background: #ffffff;
        }
        .login-title {
            color: #0f172a;
            font-size: 1.5rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .login-sub {
            margin-top: 0.25rem;
            color: #334155;
            font-size: 0.95rem;
        }
        [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--ui-border);
            border-radius: 12px;
            padding: 0.2rem;
            gap: 0.3rem;
        }
        [data-baseweb="tab"] {
            color: var(--ui-text-main) !important;
            background: var(--ui-surface-soft);
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            font-weight: 600;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            color: #ffffff !important;
            background: var(--ui-accent);
            border-color: var(--ui-accent);
        }
        [data-testid="stTextInput"] input {
            color: var(--ui-text-main) !important;
            background: var(--ui-surface) !important;
        }
        [data-testid="stForm"] button,
        [data-testid="stFormSubmitButton"] button {
            color: #ffffff !important;
            background: var(--ui-accent) !important;
            border: 1px solid var(--ui-accent) !important;
        }
        @media (max-width: 640px) {
            .login-wrap {
                padding: 0.8rem 0.85rem;
            }
            .login-title {
                font-size: 1.2rem;
            }
            .login-sub {
                font-size: 0.9rem;
            }
            [data-baseweb="tab"] {
                font-size: 0.92rem;
                min-height: 40px;
                padding: 0.3rem 0.55rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class=\"kpi-card\">
            <div class=\"kpi-label\">{label}</div>
            <div class=\"kpi-value\">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_top_users_metrics(users_activity: pd.DataFrame) -> dict[str, float]:
    total_events = float(users_activity["events"].sum())
    if total_events <= 0:
        return {"top3_share": 0.0, "top10_share": 0.0, "hhi": 0.0}

    shares = users_activity["events"] / total_events
    top3_share = float(users_activity.head(3)["events"].sum() / total_events * 100)
    top10_share = float(users_activity.head(10)["events"].sum() / total_events * 100)
    hhi = float((shares.pow(2).sum()) * 10_000)
    return {"top3_share": top3_share, "top10_share": top10_share, "hhi": hhi}


def require_report_access() -> bool:
    if st.session_state.get("report_authenticated", False):
        return True

    report_password = resolve_report_password()
    if not report_password:
        st.error("Пароль для доступу не налаштовано. Додайте PASSWORD у Secrets.")
        return False

    st.markdown(
        """
        <div class="login-wrap">
            <div class="login-title">Вхід до звіту</div>
            <div class="login-sub">Для перегляду аналітики введіть пароль.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("report_login_form", clear_on_submit=False):
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Увійти")

    if submitted:
        if password == report_password:
            st.session_state["report_authenticated"] = True
            st.rerun()
        else:
            st.error("Невірний пароль")
    return False


def main():
    st.set_page_config(page_title="Аналітика використання бота", layout="wide")
    inject_modern_styles()

    if not require_report_access():
        return

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">Аналітика використання бота</div>
            <div class="hero-sub">Огляд ключових метрик, часових патернів та інсайтів для оптимізації продукту</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_url = resolve_data_source_url()
    local_excel_bytes = load_default_local_excel_bytes()

    if not source_url and local_excel_bytes is None:
        st.error("Сервіс тимчасово недоступний: джерело даних не знайдено.")
        return

    # Fixed product defaults for end users (no interactive setup needed).
    excluded_users_text = ""
    exclude_month_year = True
    exclude_navigation_actions = True
    navigation_actions_text = "\n".join(sorted(NAVIGATION_ACTIONS_DEFAULT))
    exclude_phone_events = True
    session_gap_minutes = 30
    min_action_count = 25
    top_n_actions = 15

    if source_url:
        try:
            file_bytes = read_excel_bytes_from_url(source_url)
        except Exception as e:
            st.error(f"Не вдалося завантажити датасет за посиланням: {e}")
            return
    else:
        file_bytes = local_excel_bytes

    try:
        sheet_names = get_excel_sheet_names(file_bytes)
    except Exception as e:
        st.error(f"Не вдалося прочитати Excel: {e}")
        return

    selected_sheet = sheet_names[0]

    try:
        df_raw = load_data(file_bytes, selected_sheet)
    except Exception as e:
        st.error(str(e))
        return

    excluded_users = parse_multiline_set(excluded_users_text) | INTERNAL_EXCLUDED_USERS
    navigation_actions = parse_multiline_set(navigation_actions_text)

    try:
        _, df, sessions = prepare_data(
            df_raw=df_raw,
            excluded_users=excluded_users,
            exclude_month_year=exclude_month_year,
            exclude_navigation_actions=exclude_navigation_actions,
            navigation_actions=navigation_actions,
            exclude_phone_events=exclude_phone_events,
            session_gap_minutes=session_gap_minutes,
        )
    except Exception as e:
        st.error(f"Помилка підготовки даних: {e}")
        return

    if df.empty:
        st.warning("Після фільтрації не залишилось рядків. Послаб деякі фільтри.")
        return

    tables = create_summary_tables(df, sessions)
    dom, _ = analyze_day_of_month(df)
    weekday_df = analyze_weekday(df)
    heatmap_df = build_weekday_hour_heatmap(df)
    lift, chi_table, peaks = analyze_action_day_relationship(
        df,
        min_action_count=min_action_count,
        top_n_actions=min(top_n_actions, 12),
    )
    transitions = analyze_transitions(df, session_gap_minutes=session_gap_minutes)
    insights_text = build_text_insights(df, tables, dom, weekday_df, chi_table, peaks, transitions)
    terms_explanation_text = get_terms_explanation_text()
    top_users_metrics = get_top_users_metrics(tables["users_activity"])

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        render_kpi_card("Подій", f"{len(df):,}".replace(",", " "))
    with k2:
        render_kpi_card("Користувачів", str(df["user_id"].nunique()))
    with k3:
        render_kpi_card("Функцій", str(df["action"].nunique()))
    with k4:
        render_kpi_card("Сесій", str(len(sessions)))
    with k5:
        render_kpi_card("Частка ТОП-10", f"{top_users_metrics['top10_share']:.1f}%")
    with k6:
        render_kpi_card("HHI користувачів", f"{top_users_metrics['hhi']:.0f}")

    fig1 = plot_top_actions(tables["action_popularity"], top_n=top_n_actions)
    fig2 = plot_monthly_usage(tables["monthly_usage"])
    fig3 = plot_day_of_month(dom)
    fig4 = plot_weekday_usage(weekday_df)
    fig5 = plot_weekday_hour_heatmap(heatmap_df)
    fig6 = plot_entry_actions(tables["entry_actions"])
    fig7 = plot_action_day_lift(lift)
    fig8 = plot_top_transitions(transitions)
    fig9 = plot_user_concentration(tables["users_activity"])

    chart_figs = {
        "01_top_actions.png": fig1,
        "02_monthly_usage.png": fig2,
        "03_day_of_month_usage.png": fig3,
        "04_weekday_usage.png": fig4,
        "05_weekday_hour_heatmap.png": fig5,
        "06_session_entry_actions.png": fig6,
        "07_action_day_lift_heatmap.png": fig7,
        "08_top_transitions.png": fig8,
        "09_user_concentration_pareto.png": fig9,
    }

    tabs = st.tabs(["Головна", "Графіки", "Таблиці"])

    with tabs[0]:
        st.subheader("Ключове зведення")
        st.text(insights_text)
        st.subheader("Міра концентрації ТОП-користувачів")
        c1, c2, c3 = st.columns(3)
        c1.metric("Частка ТОП-3", f"{top_users_metrics['top3_share']:.1f}%")
        c2.metric("Частка ТОП-10", f"{top_users_metrics['top10_share']:.1f}%")
        c3.metric("HHI", f"{top_users_metrics['hhi']:.0f}")

        with st.expander("Пояснення термінів"):
            st.text(terms_explanation_text)

    with tabs[1]:
        st.pyplot(fig1, clear_figure=False)
        st.pyplot(fig2, clear_figure=False)
        st.pyplot(fig3, clear_figure=False)
        st.pyplot(fig4, clear_figure=False)
        st.pyplot(fig5, clear_figure=False)
        st.pyplot(fig6, clear_figure=False)
        st.pyplot(fig7, clear_figure=False)
        st.pyplot(fig8, clear_figure=False)
        st.pyplot(fig9, clear_figure=False)
        st.info(
            "Що показує графік 'Концентрація активності користувачів (Pareto)': "
            "по осі X - накопичена частка користувачів (%), по осі Y - накопичена частка всіх подій (%). "
            "Чим вище крива над пунктирною діагоналлю, тим сильніше активність зосереджена у невеликої групи користувачів."
        )

    with tabs[2]:
        st.subheader("Зведені таблиці")
        st.write("Summary")
        st.dataframe(tables["summary"], use_container_width=True)
        st.write("Популярність функцій")
        st.dataframe(tables["action_popularity"], use_container_width=True)
        st.write("Використання по числах місяця")
        st.dataframe(dom, use_container_width=True)
        st.write("Використання по днях тижня")
        st.dataframe(weekday_df, use_container_width=True)
        st.write("Піки функцій по числах місяця")
        st.dataframe(peaks.head(50), use_container_width=True)
        st.write("Переходи між функціями")
        st.dataframe(transitions.head(50), use_container_width=True)

    for fig in chart_figs.values():
        plt.close(fig)


if __name__ == "__main__":
    main()