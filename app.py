import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


ACTUAL_DATA_MAX_YEAR = 2023
EXCLUDED_DASHBOARD_YEARS = {2024}
DATA_DIR = Path(".")


st.set_page_config(
    page_title="BharatScope",
    page_icon="ÃƒÂ°Ã…Â¸Ã¢â‚¬Â¡Ã‚Â®ÃƒÂ°Ã…Â¸Ã¢â‚¬Â¡Ã‚Â³",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_csv_dataset(path_str, *, encoding="utf-8", **kwargs):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.name}")
    try:
        frame = pd.read_csv(path, encoding=encoding, **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path.name} is empty.") from exc
    except (UnicodeDecodeError, ValueError, OSError) as exc:
        raise ValueError(f"{path.name} could not be read as a valid CSV file.") from exc
    if frame.empty:
        raise ValueError(f"{path.name} is empty.")
    return frame


@st.cache_data(show_spinner=False)
def load_excel_dataset(path_str, *, sheet_name=0, **kwargs):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.name}")
    try:
        frame = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    except ValueError as exc:
        raise ValueError(f"{path.name} could not be read as a valid Excel file.") from exc
    except OSError as exc:
        raise ValueError(f"{path.name} could not be opened.") from exc
    if isinstance(frame, pd.DataFrame) and frame.empty:
        raise ValueError(f"{path.name} is empty.")
    return frame


DATASET_SPECS = {
    "main": ("csv", "CS661 Dataset - Sheet1.csv", {}),
    "gdp": ("excel", "GDP.xlsx", {}),
    "employment": ("csv", "employment data.csv", {}),
    "policy": ("excel", "Indian Policy Timeline.xlsx", {}),
    "infant": ("excel", "Infant Mortality.xlsx", {}),
    "literacy": ("excel", "Literacy rate.xlsx", {"header": None}),
    "population": ("excel", "population.xlsx", {}),
    "poverty": ("csv", "all_states_interpolated_poverty.csv", {}),
    "poverty_reference": ("excel", "Poverty.xlsx", {}),
    "sector": ("excel", "agriculture_industrial.xlsx", {}),
}


@st.cache_data(show_spinner=False)
def load_dataset(name):
    kind, filename, kwargs = DATASET_SPECS[name]
    loader = load_csv_dataset if kind == "csv" else load_excel_dataset
    return loader(str(DATA_DIR / filename), **kwargs)


def load_required_datasets(names):
    loaded = {}
    for name in names:
        try:
            loaded[name] = load_dataset(name)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    return loaded


def to_number(series):
    return pd.to_numeric(series, errors="coerce")


def normalize_state(value):
    if pd.isna(value):
        return value
    value = str(value).strip()
    value = value.replace("State / Union Territory", "").replace('"', "").strip()
    value = value.replace("&", "and")
    value = re.sub(r"\s+", " ", value)
    canonical = {
        "Andaman and Nicobar": "Andaman and Nicobar Islands",
        "Andaman and Nicobar Islands": "Andaman and Nicobar Islands",
        "Andaman and Nicobar Islands.": "Andaman and Nicobar Islands",
        "Andaman and Nicobar Island": "Andaman and Nicobar Islands",
    }
    value = canonical.get(value, value)
    return value


def exclude_incomplete_year_rows(df, year_col="Year"):
    if year_col not in df.columns:
        return df
    years = pd.to_numeric(df[year_col], errors="coerce")
    return df[~years.isin(EXCLUDED_DASHBOARD_YEARS)].copy()


def drop_incomplete_year_columns(df):
    drop_cols = []
    for col in df.columns:
        try:
            year = int(float(col))
        except (TypeError, ValueError):
            continue
        if year in EXCLUDED_DASHBOARD_YEARS:
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def clean_frames(raw):
    empty = pd.DataFrame()
    main = raw.get("main", empty).copy()
    gdp = raw.get("gdp", empty).copy()
    emp = raw.get("employment", empty).copy()
    policy = raw.get("policy", empty).copy()
    infant = raw.get("infant", empty).copy()
    literacy = raw.get("literacy", empty).copy()
    population = raw.get("population", empty).copy()
    poverty = raw.get("poverty", empty).copy()
    poverty_reference = raw.get("poverty_reference", empty).copy()
    sector = raw.get("sector", empty).copy()

    if not main.empty and "State/UT" in main.columns:
        main["State/UT"] = main["State/UT"].map(normalize_state)
    if not gdp.empty and "State/UT" in gdp.columns:
        gdp["State/UT"] = gdp["State/UT"].map(normalize_state)
    if not emp.empty and "State" in emp.columns:
        emp["State"] = emp["State"].map(normalize_state)
    if not population.empty and "Region" in population.columns:
        population["Region"] = population["Region"].map(normalize_state)

    for col in [
        "Year",
        "Total Population",
        "Male Population",
        "Female Population",
        "Literate Population",
        "Literacy Rate (%)",
        "Male Literacy Rate (%)",
        "Female Literacy Rate (%)",
        "Working Population",
        "Non-working Population",
        "Headcount Ratio (%)",
        "Intensity (%)",
        "MPI",
        "gsdp_rs_crore",
        "gsdp_growth_pct",
        "gsdp_growth_3yr_avg",
        "gsdp_rank_in_year",
        "per_capita_income_rs",
        "pci_growth_pct",
        "pci_growth_3yr_avg",
        "pci_rank_in_year",
        "unemployment_rate_pct",
        "unemployment_change_pp",
        "unemployment_rank_in_year",
        "COVID Deaths",
        "HDI",
        "Life Expectancy",
        "Infant Mortality Rate",
    ]:
        if not main.empty and col in main.columns:
            main[col] = to_number(main[col])
    if main.empty or "COVID Deaths" not in main.columns:
        st.error("The main dataset 'CS661 Dataset - Sheet1.csv' must include the 'COVID Deaths' column.")
        st.stop()

    if not gdp.empty:
        gdp["Year"] = to_number(gdp["Year"])
        gdp["gsdp_rs_crore"] = to_number(gdp["gsdp_rs_crore"])
        gdp["gsdp_growth_pct"] = to_number(gdp["gsdp_growth_pct"])
    for frame in [main, gdp]:
        for col in ["gsdp_rs_crore", "per_capita_income_rs"]:
            if col in frame.columns:
                frame[col] = frame[col].replace(0, np.nan)
    if not emp.empty:
        emp["Year"] = to_number(emp["Year"])
        emp["WPR (%)"] = to_number(emp["WPR (%)"])
        emp["UR (%)"] = to_number(emp["UR (%)"])
    if not policy.empty:
        policy["Launch_Year"] = to_number(policy["Launch_Year"])
        policy["Start_Year"] = to_number(policy["Start_Year"])
        policy["End_Year"] = to_number(policy["End_Year"].where(policy["End_Year"] != "Present", 2026))
    if not poverty.empty:
        poverty.columns = [normalize_state(c) if isinstance(c, str) else c for c in poverty.columns]
        for col in poverty.columns:
            if col != "State":
                poverty[col] = to_number(poverty[col])
    if not sector.empty:
        for col in sector.columns:
            if col != "Financial Year":
                sector[col] = to_number(sector[col])
        sector["Year"] = sector["Financial Year"].astype(str).str.extract(r"(\d{4})").astype(float)
        sector["Year"] = to_number(sector["Year"])

    main = exclude_incomplete_year_rows(main)
    gdp = exclude_incomplete_year_rows(gdp)
    emp = exclude_incomplete_year_rows(emp)
    poverty = exclude_incomplete_year_rows(poverty)
    sector = exclude_incomplete_year_rows(sector)
    literacy = drop_incomplete_year_columns(literacy)
    infant = drop_incomplete_year_columns(infant)
    population = drop_incomplete_year_columns(population)
    poverty_reference = drop_incomplete_year_columns(poverty_reference)

    return {
        "main": main,
        "gdp": gdp,
        "employment": emp,
        "policy": policy,
        "infant": infant,
        "literacy": literacy,
        "population": population,
        "poverty": poverty,
        "poverty_reference": poverty_reference,
        "sector": sector,
    }


def metric_card(label, value, delta=None):
    delta_part = f"<div class='delta'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_part}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_caption(text):
    st.caption(text)


def parse_literacy_sheet(df):
    year_labels = []
    for value in df.iloc[2, 1:].tolist():
        try:
            year_labels.append(int(float(value)))
        except (TypeError, ValueError):
            year_labels.append(str(value))
    data = df.iloc[4:, : len(year_labels) + 1].copy()
    data.columns = ["State"] + year_labels
    data["State"] = data["State"].map(normalize_state)
    # Convert all year columns in one pass to avoid pandas slice-assignment issues
    # that can surface on newer runtimes such as Streamlit Cloud.
    year_cols = [col for col in data.columns if col != "State"]
    data[year_cols] = (
        data[year_cols]
        .replace(["na", "NA", "n.a.", "-", "--"], pd.NA)
        .apply(pd.to_numeric, errors="coerce")
    )
    return data


def parse_infant_sheet(df):
    out = df.copy()
    out = out.rename(columns={out.columns[0]: "State"})
    out["State"] = out["State"].map(normalize_state)
    out = out.rename(columns={c: int(c) if str(c).isdigit() else c for c in out.columns})
    for col in out.columns:
        if col != "State":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_population_sheet(df):
    out = df.copy()
    out = out.rename(columns={"Region": "State"})
    out["State"] = out["State"].map(normalize_state)
    out = out.rename(columns={c: int(c) if str(c).isdigit() else c for c in out.columns})
    for col in out.columns:
        if col != "State":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def parse_poverty_sheet(df):
    if "Year" in df.columns:
        out = df.melt(id_vars="Year", var_name="State", value_name="Poverty")
        out["State"] = out["State"].map(normalize_state)
        out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
        out["Poverty"] = pd.to_numeric(out["Poverty"], errors="coerce")
        return out.pivot(index="State", columns="Year", values="Poverty").reset_index()
    out = df.copy()
    out = out.rename(columns={out.columns[0]: "State"})
    out["State"] = out["State"].map(normalize_state)
    out = out.rename(columns={c: int(c) if str(c).isdigit() else c for c in out.columns})
    for col in out.columns:
        if col != "State":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def poverty_value_from_csv(df, year, state=None):
    if "Year" not in df.columns:
        return pd.NA
    year_row = df[df["Year"] == year]
    if year_row.empty:
        return pd.NA
    row = year_row.iloc[0]
    if state is None or state == "All India":
        for candidate in ["National Average", "All India", "India"]:
            if candidate in df.columns:
                return pd.to_numeric(row[candidate], errors="coerce")
        return pd.NA
    for candidate in [state, normalize_state(state)]:
        if candidate in df.columns:
            return pd.to_numeric(row[candidate], errors="coerce")
    return pd.NA


def get_long_value(df, state, year, state_col="State", value_col="Value"):
    if year not in df.columns:
        return pd.NA
    row = df[df[state_col] == state]
    if row.empty:
        return pd.NA
    return row.iloc[0][year]


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.mean() if len(series.dropna()) else pd.NA


def fmt_value(value, fmt):
    if pd.isna(value):
        return "N/A"
    return fmt.format(value)


def prepare_size_column(df, source_col, target_col="Bubble Size", min_size=8):
    out = df.copy()
    if source_col not in out.columns:
        out[target_col] = min_size
        return out
    sizes = pd.to_numeric(out[source_col], errors="coerce").abs()
    if sizes.notna().any():
        sizes = sizes.fillna(sizes.median())
        if pd.isna(sizes.median()) or float(sizes.max()) == 0:
            out[target_col] = min_size
        else:
            out[target_col] = sizes.fillna(min_size)
    else:
        out[target_col] = min_size
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").fillna(min_size).clip(lower=min_size)
    return out


def page_filter_key(page, name):
    return f"{page}_{name}"


def sidebar_page_filters(page, states, years):
    filters = {"states": list(states), "year_range": (years[0], years[-1]), "category": None, "value_range": None}
    with st.sidebar.expander("Filters", expanded=True):
        if page in {"development_map", "economic_development", "policy_impact", "human_development", "state_compare", "covid_impact", "regional_patterns", "forecasting", "classification", "sector_dashboard"}:
            filters["states"] = st.multiselect(
                "State",
                options=states,
                default=list(states),
                key=page_filter_key(page, "states"),
            )
        if page in {"development_map", "economic_development", "policy_impact", "human_development", "covid_impact", "regional_patterns", "forecasting", "classification", "sector_dashboard"}:
            filters["year_range"] = st.slider(
                "Year range",
                min_value=int(years[0]),
                max_value=int(years[-1]),
                value=(int(years[0]), int(years[-1])),
                key=page_filter_key(page, "year_range"),
            )
        if page == "policy_impact":
            policy_frame = load_dataset("policy")
            policy_categories = sorted([c for c in policy_frame["Category"].dropna().astype(str).unique().tolist() if str(c).strip()]) if "Category" in policy_frame.columns else []
            if policy_categories:
                filters["category"] = st.selectbox(
                    "Category",
                    options=["All"] + policy_categories,
                    index=0,
                    key=page_filter_key(page, "category"),
                )
        if page == "economic_development":
            gdp_min = float(np.nanmin(pd.to_numeric(gdp["gsdp_rs_crore"], errors="coerce")))
            gdp_max = float(np.nanmax(pd.to_numeric(gdp["gsdp_rs_crore"], errors="coerce")))
            filters["value_range"] = st.slider(
                "GSDP range",
                min_value=0.0,
                max_value=float(max(gdp_max, 1.0)),
                value=(0.0, float(max(gdp_max, 1.0))),
                key=page_filter_key(page, "gsdp_range"),
            )
        if page == "human_development":
            filters["value_range"] = st.slider(
                "Literacy range",
                min_value=0.0,
                max_value=100.0,
                value=(0.0, 100.0),
                key=page_filter_key(page, "literacy_range"),
            )
    return filters


def show_no_data_message(message="No data matches the selected filters."):
    st.info(message)


STATE_CENTROIDS = {
    "Andaman and Nicobar Islands": (11.7, 92.7),
    "Andhra Pradesh": (15.9, 79.7),
    "Arunachal Pradesh": (28.2, 94.7),
    "Assam": (26.2, 92.9),
    "Bihar": (25.1, 85.3),
    "Chandigarh": (30.7, 76.8),
    "Chhattisgarh": (21.3, 81.9),
    "Dadra and Nagar Haveli": (20.2, 73.0),
    "Dadra and Nagar Haveli and Daman and Diu": (20.4, 72.9),
    "Daman and Diu": (20.4, 72.9),
    "Delhi": (28.6, 77.2),
    "Goa": (15.3, 74.1),
    "Gujarat": (22.3, 71.2),
    "Haryana": (29.1, 76.1),
    "Himachal Pradesh": (31.1, 77.2),
    "Jammu and Kashmir": (33.8, 76.6),
    "Jharkhand": (23.6, 85.3),
    "Karnataka": (15.3, 75.7),
    "Kerala": (10.9, 76.3),
    "Ladakh": (34.2, 77.6),
    "Lakshadweep": (10.6, 72.6),
    "Madhya Pradesh": (22.9, 78.6),
    "Maharashtra": (19.8, 75.7),
    "Manipur": (24.7, 93.9),
    "Meghalaya": (25.5, 91.3),
    "Mizoram": (23.2, 92.9),
    "Nagaland": (26.2, 94.6),
    "Odisha": (20.9, 85.1),
    "Puducherry": (11.9, 79.8),
    "Punjab": (31.1, 75.3),
    "Rajasthan": (27.0, 74.2),
    "Sikkim": (27.5, 88.5),
    "Tamil Nadu": (11.1, 78.7),
    "Telangana": (18.1, 79.0),
    "Tripura": (23.9, 91.3),
    "Uttar Pradesh": (26.8, 80.9),
    "Uttarakhand": (30.1, 79.0),
    "West Bengal": (22.9, 87.9),
}


def poverty_long_from_csv(df):
    if "Year" not in df.columns:
        return pd.DataFrame(columns=["State", "Year", "Poverty Rate"])
    long_df = df.melt(id_vars="Year", var_name="State", value_name="Poverty Rate")
    long_df["State"] = long_df["State"].map(normalize_state)
    long_df["Year"] = to_number(long_df["Year"])
    long_df["Poverty Rate"] = to_number(long_df["Poverty Rate"])
    return long_df.dropna(subset=["Year"])


@st.cache_data
def build_analysis_panel(main_df, emp_df, poverty_df):
    panel = main_df.copy()
    panel = panel.rename(
        columns={
            "State/UT": "State",
            "gsdp_rs_crore": "GSDP",
            "gsdp_growth_pct": "GSDP Growth",
            "per_capita_income_rs": "Per Capita Income",
            "pci_growth_pct": "PCI Growth",
            "unemployment_rate_pct": "Unemployment Rate",
            "Infant Mortality Rate": "Infant Mortality",
        }
    )
    panel["State"] = panel["State"].map(normalize_state)

    emp_small = emp_df.rename(
        columns={"State": "State", "WPR (%)": "Worker Participation Rate", "UR (%)": "Employment UR"}
    ).copy()
    panel = panel.merge(emp_small, on=["State", "Year"], how="left")
    panel["Unemployment Rate"] = panel["Unemployment Rate"].fillna(panel["Employment UR"])

    poverty_long = poverty_long_from_csv(poverty_df)
    panel = panel.merge(poverty_long, on=["State", "Year"], how="left")
    panel["Poverty Rate"] = panel["Poverty Rate"].fillna(to_number(panel.get("Headcount Ratio (%)", pd.Series(index=panel.index))))

    for col in [
        "GSDP",
        "GSDP Growth",
        "Per Capita Income",
        "PCI Growth",
        "Literacy Rate (%)",
        "Unemployment Rate",
        "Worker Participation Rate",
        "Poverty Rate",
        "MPI",
        "HDI",
        "Life Expectancy",
        "Infant Mortality",
        "COVID Deaths",
        "Total Population",
        "Working Population",
    ]:
        if col in panel.columns:
            panel[col] = to_number(panel[col])
    return panel


def latest_available_year(df, preferred_year, value_col):
    candidates = df.dropna(subset=[value_col])
    if candidates.empty:
        return preferred_year
    available = sorted(candidates["Year"].dropna().astype(int).unique())
    if preferred_year in available:
        return preferred_year
    before = [year for year in available if year <= preferred_year]
    return before[-1] if before else available[-1]


def state_year_snapshot(panel, year, value_col):
    year = latest_available_year(panel, year, value_col)
    snap = panel[panel["Year"] == year].copy().dropna(subset=[value_col])
    coords = snap["State"].map(STATE_CENTROIDS)
    snap["Lat"] = coords.map(lambda item: item[0] if isinstance(item, tuple) else np.nan)
    snap["Lon"] = coords.map(lambda item: item[1] if isinstance(item, tuple) else np.nan)
    return snap, year


def add_development_labels(df):
    out = df.copy()
    feature_weights = {
        "HDI": 1.0,
        "Literacy Rate (%)": 0.8,
        "Life Expectancy": 0.7,
        "Per Capita Income": 0.8,
        "GSDP Growth": 0.4,
        "Poverty Rate": -0.7,
        "Unemployment Rate": -0.5,
        "Infant Mortality": -0.6,
        "MPI": -0.5,
    }
    score_parts = []
    for col, weight in feature_weights.items():
        if col not in out.columns:
            continue
        series = to_number(out[col])
        spread = series.max() - series.min()
        if pd.isna(spread) or spread == 0:
            continue
        score_parts.append(((series - series.min()) / spread) * weight)
    if not score_parts:
        out["Development Score"] = np.nan
        out["Development Category"] = "Unknown"
        return out
    out["Development Score"] = pd.concat(score_parts, axis=1).mean(axis=1)
    valid = out["Development Score"].dropna()
    out["Development Category"] = "Unknown"
    if valid.nunique() >= 3:
        out.loc[valid.index, "Development Category"] = pd.qcut(
            valid,
            q=3,
            labels=["Low-development", "Emerging", "High-development"],
            duplicates="drop",
        ).astype(str)
    return out


def feature_matrix(df, feature_cols):
    available = [col for col in feature_cols if col in df.columns]
    matrix = df[["State", "Year"] + available].copy().dropna(subset=["State", "Year"])
    keep = [col for col in available if matrix[col].notna().sum() >= 5]
    return matrix[["State", "Year"] + keep], keep


def temporal_forecast(series_df, target_col, horizon=5):
    clean = series_df[["Year", target_col]].dropna().sort_values("Year")
    clean = clean.groupby("Year", as_index=False)[target_col].mean()
    clean[target_col] = clean[target_col].replace(0, np.nan)
    actual_lookup = clean.copy()
    train_pool = clean[(clean["Year"] <= 2023) & clean[target_col].notna()].copy()
    if len(train_pool) < 8 or not SKLEARN_AVAILABLE:
        return None

    first_year = int(train_pool["Year"].min())

    def features_for_year(year, value_history):
        past = value_history[value_history["Year"] < year].sort_values("Year")
        if len(past) < 3:
            return None
        lag1 = float(past.iloc[-1][target_col])
        lag2 = float(past.iloc[-2][target_col])
        lag3 = float(past.iloc[-3][target_col])
        rolling3 = float(past.tail(3)[target_col].mean())
        rolling5 = float(past.tail(min(5, len(past)))[target_col].mean())
        change1 = lag1 - lag2
        change2 = lag2 - lag3
        growth1 = change1 / abs(lag2) if lag2 else 0.0
        return {
            "Year": year,
            "YearIndex": year - first_year,
            "Lag1": lag1,
            "Lag2": lag2,
            "Lag3": lag3,
            "Rolling3": rolling3,
            "Rolling5": rolling5,
            "Momentum1": change1,
            "Momentum2": change2,
            "Growth1": growth1,
        }

    feature_rows = []
    targets = []
    for _, row in train_pool.iterrows():
        row_features = features_for_year(int(row["Year"]), train_pool)
        if row_features is None:
            continue
        feature_rows.append(row_features)
        targets.append(float(row[target_col]))

    feature_df = pd.DataFrame(feature_rows)
    target = pd.Series(targets)
    if len(feature_df) < 8:
        return None

    split_idx = max(3, int(len(feature_df) * 0.8))
    if split_idx >= len(feature_df):
        split_idx = len(feature_df) - 1
    X_train = feature_df.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = feature_df.iloc[split_idx:]
    y_test = target.iloc[split_idx:]

    candidates = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=350, learning_rate=0.035, max_depth=2),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=350, max_depth=5, min_samples_leaf=2),
        "Polynomial Ridge": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=2.0)),
    }
    model_scores = []
    best_model = None
    best_name = None
    best_pred = None
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        pred = candidate.predict(X_test)
        candidate_mae = float(mean_absolute_error(y_test, pred))
        candidate_rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        model_scores.append({"Model": name, "MAE": candidate_mae, "RMSE": candidate_rmse})
        if best_model is None or candidate_rmse < min(score["RMSE"] for score in model_scores[:-1]):
            best_model = candidate
            best_name = name
            best_pred = pred

    mae = float(mean_absolute_error(y_test, best_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, best_pred)))

    final_model = candidates[best_name]
    final_model.fit(feature_df, target)
    forecast_start = 2024
    forecast_end = max(2026, int(train_pool["Year"].max()) + horizon)
    future_years = pd.DataFrame({"Year": range(forecast_start, forecast_end + 1)})
    recursive_history = train_pool[["Year", target_col]].copy()
    future_predictions = []
    for year in future_years["Year"].astype(int).tolist():
        row_features = features_for_year(year, recursive_history)
        if row_features is None:
            break
        pred_value = float(final_model.predict(pd.DataFrame([row_features]))[0])
        pred_value = max(0.0, pred_value)
        future_predictions.append({"Year": year, "Prediction": pred_value})
        recursive_history = pd.concat(
            [recursive_history, pd.DataFrame({"Year": [year], target_col: [pred_value]})],
            ignore_index=True,
        )

    history = train_pool.rename(columns={target_col: "Value"})
    test_out = X_test[["Year"]].copy()
    test_out[target_col] = y_test.values
    test_out["Prediction"] = best_pred
    forecast = pd.DataFrame(future_predictions)
    actual_future = actual_lookup[actual_lookup["Year"].between(forecast_start, forecast_end)][["Year", target_col]].copy()
    actual_future = actual_future.rename(columns={target_col: "Actual"})
    actual_future["Actual"] = actual_future["Actual"].replace(0, np.nan)
    forecast = forecast.merge(actual_future, on="Year", how="left")
    forecast["Validation Status"] = "Missing actual"
    level_indicator = not any(token in target_col.lower() for token in ["growth", "rate", "ratio", "hdi", "%"])
    for idx, row in forecast.iterrows():
        if pd.isna(row["Actual"]):
            continue
        previous_values = pd.concat(
            [
                train_pool[["Year", target_col]].rename(columns={target_col: "Value"}),
                forecast.loc[: idx - 1, ["Year", "Prediction"]].rename(columns={"Prediction": "Value"}),
            ],
            ignore_index=True,
        )
        previous_values = previous_values[previous_values["Year"] < row["Year"]].dropna(subset=["Value"])
        previous_value = previous_values.sort_values("Year").iloc[-1]["Value"] if not previous_values.empty else np.nan
        if level_indicator and pd.notna(previous_value) and previous_value > 0 and row["Actual"] < previous_value * 0.7:
            forecast.at[idx, "Validation Status"] = "Flagged partial/outlier actual"
        else:
            forecast.at[idx, "Validation Status"] = "Used for validation"
    forecast["Error"] = forecast["Actual"] - forecast["Prediction"]
    forecast["Absolute Error"] = forecast["Error"].abs()
    forecast["Lower"] = forecast["Prediction"] - 1.96 * rmse
    forecast["Upper"] = forecast["Prediction"] + 1.96 * rmse
    used_validation = forecast[forecast["Validation Status"] == "Used for validation"]
    validation_rows = int(len(used_validation))
    flagged_rows = int((forecast["Validation Status"] == "Flagged partial/outlier actual").sum())
    validation_mae = used_validation["Absolute Error"].dropna().mean()
    return history, test_out, forecast, {
        "MAE": mae,
        "RMSE": rmse,
        "Train Rows": len(X_train),
        "Test Rows": len(X_test),
        "Validation Rows": validation_rows,
        "Flagged Rows": flagged_rows,
        "Validation MAE": validation_mae,
        "Model": best_name,
        "Model Scores": pd.DataFrame(model_scores),
    }


STATE_ONE_COLOR = "#2563eb"
STATE_TWO_COLOR = "#f97316"
INDIA_COLOR = "#15803d"


def render_plotly(fig, width="stretch"):
    ink = "#111827"
    muted_ink = "#1f2937"
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=ink, size=13),
        title=dict(font=dict(color=ink, size=18)),
        legend=dict(
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(31,41,55,0.16)",
            borderwidth=1,
            font=dict(color=ink),
            title=dict(font=dict(color=ink)),
        ),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(31,41,55,0.22)", font=dict(color=ink)),
        coloraxis_colorbar=dict(tickfont=dict(color=ink), title=dict(font=dict(color=ink))),
    )
    fig.update_xaxes(
        gridcolor="rgba(31,41,55,0.14)",
        zerolinecolor="rgba(31,41,55,0.18)",
        color=ink,
        title_font=dict(color=ink),
        tickfont=dict(color=muted_ink),
    )
    fig.update_yaxes(
        gridcolor="rgba(31,41,55,0.14)",
        zerolinecolor="rgba(31,41,55,0.18)",
        color=ink,
        title_font=dict(color=ink),
        tickfont=dict(color=muted_ink),
    )
    fig.update_polars(
        bgcolor="#ffffff",
        angularaxis=dict(color=ink, tickfont=dict(color=muted_ink)),
        radialaxis=dict(color=ink, tickfont=dict(color=muted_ink), title_font=dict(color=ink)),
    )
    fig.update_annotations(font=dict(color=ink))
    fig.update_traces(textfont=dict(color=ink), selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=(width == "stretch"))


def downsample_frame(df, max_points=5000):
    if len(df) <= max_points:
        return df
    step = max(len(df) // max_points, 1)
    return df.iloc[::step].copy()


def comparison_ready(values):
    return all(pd.notna(value) for value in values)


def comparison_insight(label, state_one, state_two, india, state_one_name, state_two_name, fmt, lower_is_better=False):
    if not comparison_ready([state_one, state_two, india]):
        return "Data not available for this comparison."
    difference = abs(state_one - state_two)
    leader = state_one_name if (state_one < state_two if lower_is_better else state_one > state_two) else state_two_name
    other = state_two_name if leader == state_one_name else state_one_name
    relation = "lower" if lower_is_better else "higher"
    national_relation = "below" if ((state_one + state_two) / 2 < india) else "above"
    return (
        f"{leader} is {fmt.format(difference)} {relation} than {other}; "
        f"the two-state average is {national_relation} the India average of {fmt.format(india)} for {label}."
    )


def comparison_chart_or_message(title, values, chart_builder, insight):
    if not comparison_ready(values):
        st.info(f"{title}: Data not available for the selected states and year.")
        return
    render_plotly(chart_builder(), width="stretch")
    st.caption(insight)


def apply_comparison_layout(fig, title, xaxis_title=None, yaxis_title=None):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=45, r=25, t=60, b=45),
        legend_title_text="",
        height=360,
    )
    if xaxis_title:
        fig.update_xaxes(title=xaxis_title)
    if yaxis_title:
        fig.update_yaxes(title=yaxis_title)
    return fig


def dumbbell_chart(title, metric, state_one_name, state_two_name, state_one, state_two, india, axis_title, fmt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[state_one, state_two], y=[metric, metric], mode="lines", line=dict(color="#9ca3af", width=3), showlegend=False, hoverinfo="skip"))
    for name, value, color in [(state_one_name, state_one, STATE_ONE_COLOR), (state_two_name, state_two, STATE_TWO_COLOR), ("India Average", india, INDIA_COLOR)]:
        fig.add_trace(go.Scatter(x=[value], y=[metric], mode="markers", name=name, marker=dict(size=13, color=color, symbol="diamond" if name == "India Average" else "circle"), hovertemplate=f"{name}: {fmt.format(value)}<extra></extra>"))
    return apply_comparison_layout(fig, title, axis_title, "Comparison")


def lollipop_chart(title, state_one_name, state_two_name, state_one, state_two, india, axis_title, fmt):
    fig = go.Figure()
    for name, value, color in [(state_one_name, state_one, STATE_ONE_COLOR), (state_two_name, state_two, STATE_TWO_COLOR), ("India Average", india, INDIA_COLOR)]:
        fig.add_trace(go.Scatter(x=[name, name], y=[0, value], mode="lines+markers", name=name, line=dict(color=color, width=4), marker=dict(size=[1, 15], color=color), hovertemplate=f"{name}: {fmt.format(value)}<extra></extra>"))
    return apply_comparison_layout(fig, title, "Geography", axis_title)


def bullet_chart(title, state_one_name, state_two_name, state_one, state_two, india, axis_title, fmt):
    fig = go.Figure()
    for name, value, color in [(state_one_name, state_one, STATE_ONE_COLOR), (state_two_name, state_two, STATE_TWO_COLOR)]:
        fig.add_trace(go.Bar(y=[name], x=[value], orientation="h", name=name, marker_color=color, hovertemplate=f"{name}: {fmt.format(value)}<extra></extra>"))
    fig.add_vline(x=india, line_dash="dash", line_color=INDIA_COLOR, line_width=3, annotation_text=f"India avg: {fmt.format(india)}", annotation_position="top")
    return apply_comparison_layout(fig, title, axis_title, "Geography")


def butterfly_chart(title, state_one_name, state_two_name, state_one, state_two, india, axis_title, fmt):
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[state_one_name], x=[-state_one], orientation="h", name=state_one_name, marker_color=STATE_ONE_COLOR, customdata=[state_one], hovertemplate=f"{state_one_name}: %{{customdata:.1f}}<extra></extra>"))
    fig.add_trace(go.Bar(y=[state_two_name], x=[state_two], orientation="h", name=state_two_name, marker_color=STATE_TWO_COLOR, customdata=[state_two], hovertemplate=f"{state_two_name}: %{{customdata:.1f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[-india, india], y=[state_one_name, state_two_name], mode="markers", name="India Average", marker=dict(size=12, color=INDIA_COLOR, symbol="diamond"), hovertemplate=f"India Average: {fmt.format(india)}<extra></extra>"))
    max_value = max(state_one, state_two, india) * 1.2
    fig.update_xaxes(range=[-max_value, max_value], tickvals=[-max_value, -max_value / 2, 0, max_value / 2, max_value], ticktext=[fmt.format(max_value), fmt.format(max_value / 2), "0", fmt.format(max_value / 2), fmt.format(max_value)])
    return apply_comparison_layout(fig, title, axis_title, "Geography")


def bubble_chart(title, state_one_name, state_two_name, state_one, state_two, india, fmt):
    values = [state_one, state_two, india]
    names = [state_one_name, state_two_name, "India Average"]
    colors = [STATE_ONE_COLOR, STATE_TWO_COLOR, INDIA_COLOR]
    size_ref = max(values) / 1800 if max(values) else 1
    fig = go.Figure()
    for name, value, color in zip(names, values, colors):
        fig.add_trace(go.Scatter(x=[name], y=[1], mode="markers", name=name, marker=dict(size=[value], sizemode="area", sizeref=size_ref, color=color, opacity=0.72), hovertemplate=f"{name}: {fmt.format(value)}<extra></extra>"))
    fig.update_yaxes(visible=False)
    return apply_comparison_layout(fig, title, "Geography", None)


def dot_reference_chart(title, metric, state_one_name, state_two_name, state_one, state_two, india, axis_title, fmt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[state_one, state_two], y=[metric, metric], mode="lines", line=dict(color="#d1d5db", width=2), showlegend=False, hoverinfo="skip"))
    for name, value, color, symbol in [(state_one_name, state_one, STATE_ONE_COLOR, "circle"), (state_two_name, state_two, STATE_TWO_COLOR, "circle"), ("India Average", india, INDIA_COLOR, "diamond")]:
        fig.add_trace(go.Scatter(x=[value], y=[metric], mode="markers", name=name, marker=dict(size=14, color=color, symbol=symbol), hovertemplate=f"{name}: {fmt.format(value)}<extra></extra>"))
    return apply_comparison_layout(fig, title, axis_title, "Comparison")


def radial_chart(title, state_one_name, state_two_name, state_one, state_two, india, fmt):
    fig = go.Figure()
    for name, value, color in [(state_one_name, state_one, STATE_ONE_COLOR), (state_two_name, state_two, STATE_TWO_COLOR), ("India Average", india, INDIA_COLOR)]:
        fig.add_trace(go.Barpolar(r=[value], theta=[name], name=name, marker_color=color, hovertemplate=f"{name}: {fmt.format(value)}<extra></extra>"))
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=45, r=25, t=60, b=45), legend_title_text="", height=360, polar=dict(radialaxis=dict(title="HDI", visible=True)))
    return fig


def main_state_list(main, gdp, emp, poverty):
    states = set(main["State/UT"].dropna().unique())
    states |= set(gdp["State/UT"].dropna().unique())
    states |= set(emp["State"].dropna().unique())
    if "State" in poverty.columns:
        states |= set(poverty["State"].dropna().astype(str).map(normalize_state).unique())
    else:
        for col in poverty.columns:
            if col == "Year":
                continue
            states.add(normalize_state(col))
    return sorted(s for s in states if pd.notna(s) and str(s).strip())


core_raw = load_required_datasets(["main", "gdp", "employment", "poverty"])
core_data = clean_frames(core_raw)
main = core_data["main"]
gdp = core_data["gdp"]
emp = core_data["employment"]
poverty = core_data["poverty"]
analysis_panel = build_analysis_panel(main, emp, poverty)

states = main_state_list(main, gdp, emp, poverty)
years = sorted(
    year
    for year in main["Year"].dropna().astype(int).unique().tolist()
    if year <= ACTUAL_DATA_MAX_YEAR and year not in EXCLUDED_DASHBOARD_YEARS
)
if "focus" not in st.session_state:
    st.session_state.focus = "home"


st.markdown(
    """
    <style>
        :root {
            --bg1: #f3efe4;
            --bg2: #e8efe8;
            --ink: #1f2937;
            --ink-strong: #111827;
            --muted: #5b6472;
            --border: rgba(31, 41, 55, 0.12);
            --gold: #b45309;
            --teal: #0f766e;
            --blue: #1d4ed8;
            --card: rgba(255,255,255,0.82);
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(180,83,9,0.15), transparent 22%),
                radial-gradient(circle at 88% 8%, rgba(15,118,110,0.14), transparent 20%),
                linear-gradient(180deg, var(--bg1), var(--bg2));
            color: var(--ink-strong);
            line-height: 1.55;
            font-size: 16px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(17,24,39,0.97), rgba(55,65,81,0.93));
            color: #fff;
            border-radius: 30px;
            padding: 2.8rem 2.4rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 20px 50px rgba(15,23,42,0.20);
            min-height: 260px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.9rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 3rem;
            line-height: 1;
        }
        .hero p {
            margin: 0;
            line-height: 1.65;
            color: rgba(255,255,255,0.8);
        }
        .section {
            margin-top: 1.75rem;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            font-weight: 800;
        }
        .metric-card, .tile {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.1rem 1.1rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        }
        .metric-label { color: var(--muted); font-size: 0.87rem; margin-bottom: 0.3rem; }
        .metric-value { font-size: 1.65rem; font-weight: 800; line-height: 1.05; }
        .delta { margin-top: 0.25rem; font-size: 0.82rem; color: var(--teal); }
        .landing-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            min-height: 172px;
            padding: 1.35rem 1.3rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        }
        .landing-card h3 { margin: 0 0 0.45rem; }
        .landing-card p { color: var(--muted); margin: 0; }
        div[data-testid="stRadio"] label,
        div[data-testid="stRadio"] p,
        div[data-testid="stRadio"] span {
            color: var(--ink) !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.55rem 0.8rem;
            flex-wrap: wrap;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.76) !important;
            border: 1px solid var(--border) !important;
            border-radius: 999px !important;
            padding: 0.45rem 0.95rem !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: #ffffff !important;
            border-color: rgba(37, 99, 235, 0.35) !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            background: #ffffff !important;
            border: 2px solid #1f2937 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 0.15rem;
        }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: rgba(255, 255, 255, 0.92) !important;
        }
        section[data-testid="stSidebar"] .stCaptionContainer,
        section[data-testid="stSidebar"] .stCaptionContainer p {
            color: rgba(255, 255, 255, 0.72) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: transparent !important;
            border: 0 !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            padding: 0.28rem 0.2rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.12) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            background: rgba(255, 255, 255, 0.96) !important;
            border: 2px solid rgba(255, 255, 255, 0.96) !important;
        }
        .stButton button, .stButton > button {
            background: #ffffff !important;
            color: var(--ink-strong) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            min-height: 44px !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }
        .stButton button span,
        .stButton > button span {
            color: var(--ink-strong) !important;
        }
        .stButton button:hover,
        .stButton > button:hover {
            color: #0f172a !important;
            border-color: rgba(37, 99, 235, 0.45) !important;
            background: rgba(255, 255, 255, 0.92) !important;
        }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stMultiselect label,
        section[data-testid="stSidebar"] .stText,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] .stMarkdown {
            color: #f8fafc !important;
        }
        label, .stSelectbox label, .stSlider label {
            color: var(--ink-strong) !important;
            font-weight: 600 !important;
        }
        .stMarkdown, .stCaption, .stText, p, li {
            color: var(--ink-strong);
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stDownloadButton"] {
            margin-top: 0.8rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stPlotlyChart"] {
            margin: 0.8rem 0 1rem 0;
        }
        .stButton > button {
            padding: 0.6rem 1rem !important;
        }
        :focus-visible {
            outline: 3px solid rgba(29, 78, 216, 0.55) !important;
            outline-offset: 3px !important;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = {
    "ÃƒÂ°Ã…Â¸Ã‚ÂÃ‚Â  Home": "home",
    "ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã‹â€  Economic Development": "economic_development",
    "ÃƒÂ°Ã…Â¸Ã¢â‚¬Å“Ã…â€œ Policy Impact": "policy_impact",
    "ÃƒÂ°Ã…Â¸Ã‚ÂÃ‚Â¥ Human Development": "human_development",
    "ÃƒÂ¢Ã…Â¡Ã¢â‚¬â€œ State Comparison": "state_compare",
    "ÃƒÂ°Ã…Â¸Ã…â€™Ã‚Â¾ Sector Dashboard": "sector_dashboard",
}
pages.update(
    {
        "Development Explorer": "development_map",
        "COVID Impact": "covid_impact",
        "Regional Patterns": "regional_patterns",
        "Forecasting": "forecasting",
        "Classification": "classification",
    }
)
if st.session_state.focus not in pages.values():
    st.session_state.focus = "home"

pages = {
    "Home": "home",
    "Development Explorer": "development_map",
    "Economic Development": "economic_development",
    "Policy Impact": "policy_impact",
    "Human Development": "human_development",
    "State Comparison": "state_compare",
    "Sector Dashboard": "sector_dashboard",
    "COVID Impact": "covid_impact",
    "Regional Patterns": "regional_patterns",
    "Forecasting": "forecasting",
    "Classification": "classification",
}

with st.sidebar:
    st.title("BharatScope")
    st.caption("Indian Economic Development Analytics Platform")
    st.divider()
    current_label = next(label for label, page in pages.items() if page == st.session_state.focus)
    selected_label = st.radio(
        "Dashboard section",
        list(pages),
        index=list(pages).index(current_label),
        horizontal=False,
    )
    st.session_state.focus = pages[selected_label]
    st.divider()
    if st.session_state.focus in {"development_map", "economic_development", "policy_impact", "human_development", "state_compare", "covid_impact"}:
        selected_state = st.selectbox("State / UT", options=["All India"] + states, index=0, help="Filter the page to a single state or union territory.")
    else:
        selected_state = "All India"

selected_year = years[-1]


def local_year_control(label="Year", key="local_year", value=None):
    return st.select_slider(label, options=years, value=value or years[-1], key=key)

page_filters = sidebar_page_filters(st.session_state.focus, states, years)

selected_rows_gdp = gdp[gdp["Year"] == selected_year].copy()
if selected_state != "All India":
    selected_rows_gdp = selected_rows_gdp[selected_rows_gdp["State/UT"] == selected_state]


@st.cache_data(show_spinner=False)
def load_page_datasets(page_name):
    names_by_page = {
        "development_map": [],
        "economic_development": [],
        "policy_impact": ["policy"],
        "human_development": ["infant", "literacy", "population", "poverty_reference"],
        "state_compare": ["infant", "literacy", "population", "poverty_reference"],
        "sector_dashboard": ["sector"],
        "covid_impact": [],
        "regional_patterns": [],
        "forecasting": [],
        "classification": [],
        "home": [],
    }
    return load_required_datasets(names_by_page.get(page_name, []))

if st.session_state.focus == "development_map":
    st.subheader("State-wise Development Explorer")
    selected_year = local_year_control("Development snapshot year", key="development_year")
    explorer_indicators = [
        "GSDP",
        "Per Capita Income",
        "GSDP Growth",
        "Literacy Rate (%)",
        "HDI",
        "Poverty Rate",
        "Unemployment Rate",
        "Life Expectancy",
        "Infant Mortality",
        "COVID Deaths",
    ]
    explorer_indicators = [col for col in explorer_indicators if col in analysis_panel.columns]
    control_one, control_two, control_three = st.columns([1.15, 1.15, 1.15])
    with control_one:
        explorer_indicator = st.selectbox("Primary indicator", explorer_indicators, index=0, help="Pick the main metric used for size, color, and ranking.")
    with control_two:
        x_indicator = st.selectbox("X-axis", [col for col in explorer_indicators if col != explorer_indicator], index=0, help="Select the metric shown on the horizontal axis.")
    with control_three:
        y_indicator = st.selectbox("Y-axis", [col for col in explorer_indicators if col != explorer_indicator], index=1, help="Select the metric shown on the vertical axis.")

    map_df, map_year = state_year_snapshot(analysis_panel, selected_year, explorer_indicator)
    if page_filters["states"]:
        map_df = map_df[map_df["State"].isin(page_filters["states"])]
    map_df = map_df.copy()
    map_df = prepare_size_column(map_df, explorer_indicator)

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Displayed Year", str(map_year), "Nearest available data")
    with m2:
        metric_card("States/UTs", f"{map_df['State'].nunique():,.0f}" if not map_df.empty else "0")
    with m3:
        metric_card("Average", fmt_value(safe_mean(map_df[explorer_indicator]), "{:,.2f}") if not map_df.empty else "N/A")

    if map_df.empty:
        show_no_data_message("No data matches the selected filters for this page.")
    else:
        hover_cols = [
            col
            for col in ["GSDP", "Per Capita Income", "Literacy Rate (%)", "HDI", "Poverty Rate", "Unemployment Rate", "COVID Deaths"]
            if col in map_df.columns and col not in {explorer_indicator, x_indicator, y_indicator}
        ]
        left, right = st.columns(2)
        with left:
            bubble_df = map_df.dropna(subset=[x_indicator, y_indicator, explorer_indicator])
            fig = px.scatter(
                bubble_df,
                x=x_indicator,
                y=y_indicator,
                color=explorer_indicator,
                size="Bubble Size",
                hover_name="State",
                hover_data=hover_cols,
                color_continuous_scale="Viridis",
                title=f"Bubble Plot: {y_indicator} vs {x_indicator} ({map_year})",
            )
            fig.update_layout(template="plotly_white")
            render_plotly(fig, width="stretch")
            chart_caption(f"{explorer_indicator} compared against {x_indicator} and {y_indicator} for {map_year}.")
        with right:
            top_df = map_df.dropna(subset=[explorer_indicator]).sort_values(explorer_indicator, ascending=False).head(15)
            fig = px.bar(
                top_df,
                x=explorer_indicator,
                y="State",
                orientation="h",
                color=explorer_indicator,
                color_continuous_scale="Teal",
                title=f"Top States by {explorer_indicator}",
            )
            fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"})
            render_plotly(fig, width="stretch")
            chart_caption(f"Top 15 states ranked by {explorer_indicator} in {map_year}.")

        state_position = map_df.dropna(subset=["Lat", "Lon", explorer_indicator]).copy()
        if not state_position.empty:
            fig = px.scatter(
                state_position,
                x="Lon",
                y="Lat",
                color=explorer_indicator,
                size="Bubble Size",
                hover_name="State",
                hover_data=hover_cols,
                color_continuous_scale="Viridis",
                title="State Position View (geographic layout without distracting basemap)",
            )
            fig.update_layout(template="plotly_white", xaxis_title="Longitude", yaxis_title="Latitude")
            render_plotly(fig, width="stretch")
            chart_caption("Geographic view of the same indicator using longitude and latitude anchors.")

    heat_df = analysis_panel[["State", "Year", explorer_indicator]].dropna()
    if selected_state != "All India":
        heat_df = heat_df[heat_df["State"] == selected_state]
    else:
        top_states = (
            heat_df[heat_df["Year"] == map_year]
            .sort_values(explorer_indicator, ascending=False)
            .head(15)["State"]
            .tolist()
        )
        heat_df = heat_df[heat_df["State"].isin(top_states)]
    if not heat_df.empty:
        fig = px.density_heatmap(
            heat_df,
            x="Year",
            y="State",
            z=explorer_indicator,
            histfunc="avg",
            title=f"{explorer_indicator} Heatmap Across Time",
            color_continuous_scale="Cividis",
        )
        render_plotly(fig, width="stretch")
        chart_caption(f"Heatmap of {explorer_indicator} across the selected states and years.")
        st.dataframe(
            map_df[["State", "Year", explorer_indicator, x_indicator, y_indicator]].sort_values(explorer_indicator, ascending=False),
            width="stretch",
            hide_index=True,
        )

if st.session_state.focus == "home":
    st.markdown(
        """
        <div class="hero">
            <h1>BharatScope</h1>
            <p>Interactive Visual Analytics of Indian Economic Development</p>
            <p>Explore India's economic progress, policy outcomes, human development, state-level performance, and sectoral transformation through focused analytical dashboards.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section'>Dashboards</div>", unsafe_allow_html=True)
    landing_cards = [
        ("Development Explorer", "Compare states with heatmaps, bubble plots, rankings, and geographic state-position views.", "development_map"),
        ("Economic Development", "Analyze GSDP, growth, and per-capita income across states.", "economic_development"),
        ("Policy Impact", "Examine policy eras alongside employment, poverty, and growth indicators.", "policy_impact"),
        ("Human Development", "Review literacy, population, poverty, and infant mortality outcomes.", "human_development"),
        ("State Comparison", "Benchmark two states against each other and the national average.", "state_compare"),
        ("Sector Dashboard", "Track agriculture, industry, and services in India's economy.", "sector_dashboard"),
        ("COVID Impact", "Compare pre/post pandemic shifts in growth, employment, and recovery.", "covid_impact"),
        ("Regional Patterns", "Use PCA and K-Means to find similar state development profiles.", "regional_patterns"),
        ("Forecasting", "Train/test time-series forecasts for economic and development indicators.", "forecasting"),
        ("Classification", "Classify states and inspect Random Forest feature importance.", "classification"),
    ]
    for row in (landing_cards[:2], landing_cards[2:4], landing_cards[4:6], landing_cards[6:8], landing_cards[8:]):
        columns = st.columns(len(row))
        for column, (title, description, page) in zip(columns, row):
            with column:
                st.markdown(
                    f"<div class='landing-card'><h3>{title}</h3><p>{description}</p></div>",
                    unsafe_allow_html=True,
                )
                if st.button("Open", key=f"explore_{page}", width="stretch", help=f"Open the {title} section."):
                    st.session_state.focus = page
                    st.rerun()

if st.session_state.focus == "economic_development":
    st.subheader("Economic Development")
    selected_year = local_year_control("Economic snapshot year", key="economic_year")
    econ_gdp = gdp.copy()
    if page_filters["states"]:
        econ_gdp = econ_gdp[econ_gdp["State/UT"].isin(page_filters["states"])]
    econ_gdp = econ_gdp[econ_gdp["Year"].between(*page_filters["year_range"])]
    selected_gdp_rows = econ_gdp[econ_gdp["Year"] == selected_year].copy()
    selected_gsdp = safe_mean(selected_gdp_rows["gsdp_rs_crore"]) if "gsdp_rs_crore" in selected_gdp_rows.columns else pd.NA
    selected_growth = safe_mean(selected_gdp_rows["gsdp_growth_pct"]) if "gsdp_growth_pct" in selected_gdp_rows.columns else pd.NA
    selected_pci = safe_mean(selected_gdp_rows["per_capita_income_rs"]) if "per_capita_income_rs" in selected_gdp_rows.columns else pd.NA
    selected_pci_growth = safe_mean(selected_gdp_rows["pci_growth_pct"]) if "pci_growth_pct" in selected_gdp_rows.columns else pd.NA
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("GSDP", f"Ã¢â€šÂ¹ {fmt_value(selected_gsdp, '{:,.0f}')} Cr" if pd.notna(selected_gsdp) else "N/A", "Selected year/state")
    with c2:
        metric_card("GSDP Growth", f"{fmt_value(selected_growth, '{:.2f}')}%" if pd.notna(selected_growth) else "N/A")
    with c3:
        metric_card("Per Capita Income", f"Ã¢â€šÂ¹ {fmt_value(selected_pci, '{:,.0f}')}" if pd.notna(selected_pci) else "N/A")
    with c4:
        metric_card("PCI Growth", f"{fmt_value(selected_pci_growth, '{:.2f}')}%" if pd.notna(selected_pci_growth) else "N/A")

    left, right = st.columns(2)
    with left:
        gdp_trend = econ_gdp.dropna(subset=["gsdp_rs_crore"]).copy()
        gdp_trend = downsample_frame(gdp_trend, 4000)
        if gdp_trend.empty:
            show_no_data_message("No GSDP data matches the selected filters.")
        else:
            fig = px.line(
                gdp_trend,
                x="Year",
                y="gsdp_rs_crore",
                color="State/UT",
                title="GSDP Trends by State",
            )
            render_plotly(fig, width="stretch")
        chart_caption("GSDP trend lines for the selected states and year range.")
    with right:
        top_gdp = selected_gdp_rows.dropna(subset=["gsdp_rs_crore"]).sort_values("gsdp_rs_crore", ascending=False).head(10)
        if top_gdp.empty:
            show_no_data_message("No state-level GSDP values match the selected filters.")
        else:
            fig = px.bar(top_gdp, x="State/UT", y="gsdp_rs_crore", color="gsdp_growth_pct", title=f"Top GSDP States in {selected_year}")
            render_plotly(fig, width="stretch")
            chart_caption(f"Top GSDP states for {selected_year}.")

if st.session_state.focus == "sector_dashboard":
    st.subheader("Sector Dashboard")
    selected_year = local_year_control("Sector snapshot year", key="sector_year")
    page_data = load_page_datasets("sector_dashboard")
    sector = page_data["sector"].copy()
    if "Year" not in sector.columns and "Financial Year" in sector.columns:
        sector["Year"] = sector["Financial Year"].astype(str).str.extract(r"(\d{4})").astype(float)
    if "Year" in sector.columns:
        sector["Year"] = to_number(sector["Year"])
    sector_year = sector[sector["Year"] == (selected_year - 1)].copy()
    sector_year = sector_year.dropna(subset=["Year"])
    s1, s2, s3 = st.columns(3)
    with s1:
        metric_card("Agriculture Share", f"{sector_year['Agriculture - Share to Total GDP'].mean():.2f}%" if not sector_year.empty else "N/A")
    with s2:
        metric_card("Industry Share", f"{sector_year['Industry - Share to Total GDP'].mean():.2f}%" if not sector_year.empty else "N/A")
    with s3:
        metric_card("Services Share", f"{sector_year['Services - Share to Total GDP'].mean():.2f}%" if not sector_year.empty else "N/A")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        sector_trend = sector.dropna(subset=["Year"])[
            ["Year", "Agriculture - Share to Total GDP", "Industry - Share to Total GDP", "Services - Share to Total GDP"]
        ].copy()
        sector_trend = sector_trend[sector_trend["Year"].between(*page_filters["year_range"])]
        sector_trend = sector_trend.melt(id_vars="Year", var_name="Sector", value_name="Share")
        if sector_trend.empty:
            show_no_data_message("No sector data matches the selected filters.")
        else:
            fig = px.line(sector_trend, x="Year", y="Share", color="Sector", title="Sector Share in GDP")
            render_plotly(fig, width="stretch")
            chart_caption("Sector share trend over time.")
    with lower_right:
        sector_latest = sector_year if not sector_year.empty else sector.dropna(subset=["Year"]).sort_values("Year").tail(1)
        if not sector_latest.empty:
            sector_cards = pd.DataFrame(
                {
                    "Sector": ["Agriculture", "Industry", "Services"],
                    "Share": [
                        sector_latest["Agriculture - Share to Total GDP"].mean(),
                        sector_latest["Industry - Share to Total GDP"].mean(),
                        sector_latest["Services - Share to Total GDP"].mean(),
                    ],
                }
            )
            sector_cards = sector_cards.dropna(subset=["Share"])
            if sector_cards.empty:
                st.info("No data available")
            else:
                fig = px.pie(
                    sector_cards,
                    values="Share",
                    names="Sector",
                    title=f"Sector Share Snapshot ({int(sector_latest['Year'].mean())})",
                    hole=0.35,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig.update_traces(
                    textposition="inside",
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value:.2f}%<br>Share: %{percent}<extra></extra>",
                )
                render_plotly(fig, width="stretch")
                chart_caption(f"Sector share snapshot for {int(sector_latest['Year'].mean())}.")
        else:
            st.info("No data available")

if st.session_state.focus == "policy_impact":
    st.subheader("Policy Impact")
    selected_year = local_year_control("Policy context year", key="policy_year")
    page_data = load_page_datasets("policy_impact")
    policy = page_data["policy"].copy()
    policy_years = sorted(set(emp["Year"].dropna().astype(int).unique()) & set(poverty["Year"].dropna().astype(int).unique()))
    policy_year = selected_year if selected_year in policy_years else (policy_years[-1] if policy_years else int(emp["Year"].dropna().max()))
    policy_emp = emp[emp["Year"] == policy_year].copy()
    if page_filters["states"]:
        policy_emp = policy_emp[policy_emp["State"].isin(page_filters["states"])]
    poverty_scope = selected_state if selected_state != "All India" else "All India"
    policy_poverty = poverty_value_from_csv(poverty, policy_year, poverty_scope)
    policy_gdp = gdp[gdp["Year"] == policy_year].copy()
    if page_filters["states"]:
        policy_gdp = policy_gdp[policy_gdp["State/UT"].isin(page_filters["states"])]
    if page_filters["category"] and page_filters["category"] != "All":
        policy = policy[policy["Category"] == page_filters["category"]]

    p1, p2, p3 = st.columns(3)
    with p1:
        metric_card("Poverty Ratio", f"{policy_poverty:.2f}%" if pd.notna(policy_poverty) else "N/A")
    with p2:
        metric_card(
            "Unemployment",
            f"{policy_emp['UR (%)'].mean():.2f}%" if not policy_emp.empty else "N/A",
        )
    with p3:
        metric_card(
            "Employment Rate",
            f"{policy_emp['WPR (%)'].mean():.2f}%" if not policy_emp.empty else "N/A",
        )

    metric_card("GSDP", f"Ã¢â€šÂ¹ {policy_gdp['gsdp_rs_crore'].mean():,.0f} Cr" if not policy_gdp.empty else "N/A")

    left, right = st.columns(2)
    with left:
        policy_years = sorted(set(emp["Year"].dropna().astype(int).unique()) & set(poverty["Year"].dropna().astype(int).unique()))
        policy_trend = pd.DataFrame(
            [
                {"Year": y, "Indicator": "Poverty Ratio", "Value": poverty_value_from_csv(poverty, y, poverty_scope)}
                for y in policy_years
            ]
            + [
                {"Year": y, "Indicator": "Unemployment", "Value": emp.loc[emp["Year"] == y, "UR (%)"].mean()}
                for y in policy_years
            ]
            + [
                {"Year": y, "Indicator": "Employment Rate", "Value": emp.loc[emp["Year"] == y, "WPR (%)"].mean()}
                for y in policy_years
            ]
            + [
                {"Year": y, "Indicator": "GSDP", "Value": gdp.loc[gdp["Year"] == y, "gsdp_rs_crore"].mean()}
                for y in policy_years
            ]
        )
        policy_trend["Year"] = to_number(policy_trend["Year"])
        policy_trend = policy_trend[policy_trend["Year"].between(*page_filters["year_range"])]
        if policy_trend.empty:
            show_no_data_message("No policy-linked data matches the selected filters.")
        else:
            center_left, center_center, center_right = st.columns([1, 6, 1])
            with center_center:
                fig = px.line(
                    policy_trend,
                    x="Year",
                    y="Value",
                    color="Indicator",
                    title="Policy-Linked Indicators",
                )
                fig.update_layout(margin=dict(l=30, r=30, t=60, b=35))
                render_plotly(fig, width="stretch")
            chart_caption("Policy-linked indicator trends over the selected time window.")

    policy_table = policy[["Policy_Name", "Category", "Launch_Year", "Description"]]
    st.dataframe(policy_table, width="stretch", hide_index=True)

if st.session_state.focus == "human_development":
    st.subheader("Human Development")
    page_data = load_page_datasets("human_development")
    infant = page_data["infant"].copy()
    literacy = page_data["literacy"].copy()
    population = page_data["population"].copy()
    poverty_reference = page_data["poverty_reference"].copy()
    literacy_long = parse_literacy_sheet(literacy)
    poverty_state = poverty.melt(id_vars="Year", var_name="State", value_name="Poverty")
    poverty_state["State"] = poverty_state["State"].map(normalize_state)
    poverty_state["Year"] = to_number(poverty_state["Year"])
    poverty_state["Poverty"] = to_number(poverty_state["Poverty"])
    human_year = st.slider("Select year", min_value=1991, max_value=2011, value=2011, step=10)
    human_rows = main[main["Year"] == human_year].copy()
    if page_filters["states"]:
        human_rows = human_rows[human_rows["State/UT"].isin(page_filters["states"])]
    lit_year = literacy_long[["State", human_year]].rename(columns={human_year: "Literacy"})
    lit_year = lit_year.dropna(subset=["Literacy"])
    if page_filters["states"]:
        lit_year = lit_year[lit_year["State"].isin(page_filters["states"])]
    pov_year = poverty_state[poverty_state["Year"] == human_year]
    if pov_year.empty:
        poverty_fallback = poverty_reference.rename(columns={poverty_reference.columns[0]: "State"}).melt(
            id_vars="State", var_name="Year", value_name="Poverty"
        )
        poverty_fallback["State"] = poverty_fallback["State"].map(normalize_state)
        poverty_fallback["Year"] = to_number(poverty_fallback["Year"])
        poverty_fallback["Poverty"] = to_number(poverty_fallback["Poverty"])
        pov_year = poverty_fallback[poverty_fallback["Year"] == human_year]
    if page_filters["states"]:
        pov_year = pov_year[pov_year["State"].isin(page_filters["states"])]
    elif "National Average" in pov_year["State"].values:
        pov_year = pov_year[pov_year["State"] == "National Average"]
    pop_year = population[["Region", human_year]] if human_year in population.columns else population[[population.columns[0], 2011]]
    pop_year = pop_year.rename(columns={pop_year.columns[0]: "State", human_year if human_year in pop_year.columns else 2011: "Population"})
    pop_year["State"] = pop_year["State"].map(normalize_state)
    if page_filters["states"]:
        pop_year = pop_year[pop_year["State"].isin(page_filters["states"])]
    imr_state = infant.rename(columns={infant.columns[0]: "State"}).copy()
    imr_state["State"] = imr_state["State"].map(normalize_state)
    imr_state = imr_state.rename(columns={c: int(c) if str(c).isdigit() else c for c in imr_state.columns})
    imr_years = [c for c in imr_state.columns if c != "State" and c in {1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2011}]
    imr_long = imr_state.melt(id_vars="State", value_vars=imr_years, var_name="Year", value_name="IMR")
    imr_long["Year"] = to_number(imr_long["Year"])
    imr_long["IMR"] = to_number(imr_long["IMR"])
    if page_filters["states"]:
        imr_long = imr_long[imr_long["State"].isin(page_filters["states"])]

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        metric_card("Literacy", f"{lit_year['Literacy'].mean():.2f}%" if not lit_year.empty else "N/A")
    with h2:
        metric_card("Population", f"{pop_year['Population'].mean():,.0f}" if not pop_year.empty else "N/A")
    with h3:
        metric_card("Poverty", f"{pov_year['Poverty'].mean():.2f}%" if not pov_year.empty else "N/A")
    with h4:
        imr_year = imr_long[imr_long["Year"] == human_year]
        metric_card("Infant Mortality", f"{imr_year['IMR'].mean():.1f}" if not imr_year.empty else "N/A")

    left, right = st.columns(2)
    with left:
        lit_plot = literacy_long.melt(id_vars="State", var_name="Year", value_name="Literacy")
        lit_plot["Year"] = to_number(lit_plot["Year"])
        lit_plot = lit_plot.dropna(subset=["Literacy"])
        if page_filters["states"]:
            lit_plot = lit_plot[lit_plot["State"].isin(page_filters["states"])]
        lit_plot = lit_plot[lit_plot["Year"].between(page_filters["year_range"][0], page_filters["year_range"][1])]
        lit_plot = downsample_frame(lit_plot, 3000)
        if lit_plot.empty:
            show_no_data_message("No literacy data matches the selected filters.")
        else:
            fig = px.line(lit_plot, x="Year", y="Literacy", color="State", title="Literacy Across States")
            render_plotly(fig, width="stretch")
            chart_caption("Literacy trend lines for the selected states and year range.")
    with right:
        if imr_long.empty:
            show_no_data_message("No infant mortality data matches the selected filters.")
        else:
            fig = px.line(imr_long, x="Year", y="IMR", color="State", title="Infant Mortality Trend")
            render_plotly(fig, width="stretch")
            chart_caption("Infant mortality trend lines for the selected states.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        pop_long = population.melt(id_vars="Region", var_name="Year", value_name="Population")
        pop_long["Year"] = to_number(pop_long["Year"])
        pop_long["Population"] = to_number(pop_long["Population"].replace("State not formed", pd.NA))
        pop_long = pop_long.dropna(subset=["Population"])
        pop_long["Region"] = pop_long["Region"].map(normalize_state)
        if page_filters["states"]:
            pop_long = pop_long[pop_long["Region"].isin(page_filters["states"])]
        pop_long = pop_long[pop_long["Year"].between(*page_filters["year_range"])]
        pop_long = downsample_frame(pop_long, 3000)
        if pop_long.empty:
            show_no_data_message("No population data matches the selected filters.")
        else:
            fig = px.line(pop_long[pop_long["Year"] >= 1951], x="Year", y="Population", color="Region", title="Population Growth Snapshot")
            render_plotly(fig, width="stretch")
            chart_caption("Population growth across the selected state or states.")
    with bottom_right:
        lit_year_plot = lit_year.sort_values("Literacy", ascending=False).head(12)
        fig = px.bar(lit_year_plot, x="State", y="Literacy", title=f"Literacy Ranking in {human_year}")
        render_plotly(fig, width="stretch")

if st.session_state.focus == "state_compare":
    st.subheader("State Compare")
    page_data = load_page_datasets("state_compare")
    infant = page_data["infant"].copy()
    literacy = page_data["literacy"].copy()
    population = page_data["population"].copy()
    poverty_reference = page_data["poverty_reference"].copy()
    compare_col1, compare_col2, compare_col3 = st.columns([1, 1, 1])
    compare_state_options = page_filters["states"] or states
    with compare_col1:
        compare_state_1 = st.selectbox(
            "Pick first state/UT",
            options=compare_state_options,
            index=0,
            key="compare_state_1",
        )
    with compare_col2:
        second_options = [state for state in compare_state_options if state != compare_state_1] or compare_state_options
        compare_state_2 = st.selectbox(
            "Pick second state/UT",
            options=second_options,
            index=0,
            key="compare_state_2",
        )
    with compare_col3:
        compare_year = st.slider("Select year", min_value=1991, max_value=2011, value=2011, step=10)

    literacy_state = parse_literacy_sheet(literacy)
    infant_state = parse_infant_sheet(infant)
    population_state = parse_population_sheet(population)
    poverty_state = parse_poverty_sheet(poverty)

    def year_col(df, year):
        if year in df.columns:
            return year
        if str(year) in df.columns:
            return str(year)
        return None

    literacy_year_col = year_col(literacy_state, compare_year)
    infant_year_col = year_col(infant_state, compare_year)
    population_year_col = year_col(population_state, compare_year)
    poverty_year_col = year_col(poverty_state, compare_year)

    gdp_year = gdp[gdp["Year"] == compare_year]
    emp_year = emp[emp["Year"] == compare_year]
    main_year = main[main["Year"] == compare_year]

    def safe_mean(series):
        series = pd.to_numeric(series, errors="coerce")
        return series.mean() if len(series.dropna()) else pd.NA

    def fmt_value(value, fmt):
        if pd.isna(value):
            return "N/A"
        return fmt.format(value)

    metric_specs = [
        {
            "label": "Literacy",
            "value": lambda state: get_long_value(literacy_state, state, compare_year),
            "india": safe_mean(literacy_state[literacy_year_col]) if literacy_year_col is not None else pd.NA,
            "fmt": "{:.2f}%",
        },
        {
            "label": "GSDP",
            "value": lambda state: safe_mean(gdp_year[gdp_year["State/UT"] == state]["gsdp_rs_crore"]),
            "india": safe_mean(gdp_year["gsdp_rs_crore"]),
            "fmt": "{:,.0f} Cr",
        },
        {
            "label": "Unemployment",
            "value": lambda state: safe_mean(emp_year[emp_year["State"] == state]["UR (%)"]),
            "india": safe_mean(emp_year["UR (%)"]),
            "fmt": "{:.2f}%",
        },
        {
            "label": "Infant Mortality",
            "value": lambda state: get_long_value(infant_state, state, compare_year),
            "india": safe_mean(infant_state[infant_year_col]) if infant_year_col is not None else pd.NA,
            "fmt": "{:.1f}",
        },
        {
            "label": "Population",
            "value": lambda state: get_long_value(population_state, state, compare_year),
            "india": safe_mean(population_state[population_year_col]) if population_year_col is not None else pd.NA,
            "fmt": "{:,.0f}",
        },
        {
            "label": "Poverty",
            "value": lambda state: get_long_value(poverty_state, state, compare_year),
            "india": safe_mean(poverty_state[poverty_year_col]) if poverty_year_col is not None else pd.NA,
            "fmt": "{:.2f}%",
        },
        {
            "label": "HDI",
            "value": lambda state: safe_mean(main_year[main_year["State/UT"] == state]["HDI"]),
            "india": safe_mean(main_year["HDI"]),
            "fmt": "{:.3f}",
        },
    ]

    st.markdown("<div class='section'>Selected year comparison</div>", unsafe_allow_html=True)
    metric_rows = st.columns(3)
    for idx, spec in enumerate(metric_specs):
        with metric_rows[idx % 3]:
            v1 = spec["value"](compare_state_1)
            v2 = spec["value"](compare_state_2)
            main_text = f"{compare_state_1}: {fmt_value(v1, spec['fmt'])}<br>{compare_state_2}: {fmt_value(v2, spec['fmt'])}"
            delta_text = f"India avg: {fmt_value(spec['india'], spec['fmt'])}"
            metric_card(spec["label"], main_text, delta_text)

    compare_table = pd.DataFrame(
        {
            "Indicator": [spec["label"] for spec in metric_specs],
            compare_state_1: [spec["value"](compare_state_1) for spec in metric_specs],
            compare_state_2: [spec["value"](compare_state_2) for spec in metric_specs],
            "India Avg": [spec["india"] for spec in metric_specs],
        }
    )

    comparison_data = {
        spec["label"]: (spec["value"](compare_state_1), spec["value"](compare_state_2), spec["india"], spec["fmt"])
        for spec in metric_specs
    }

    st.markdown("<div class='section'>Metric-by-metric comparison</div>", unsafe_allow_html=True)
    row_one_left, row_one_right = st.columns(2)
    literacy_one, literacy_two, literacy_india, literacy_fmt = comparison_data["Literacy"]
    with row_one_left:
        comparison_chart_or_message(
            "Literacy: state dumbbell comparison",
            [literacy_one, literacy_two, literacy_india],
            lambda: dumbbell_chart(
                "Literacy comparison", "Literacy", compare_state_1, compare_state_2,
                literacy_one, literacy_two, literacy_india, "Literacy rate (%)", literacy_fmt,
            ),
            comparison_insight("literacy", literacy_one, literacy_two, literacy_india, compare_state_1, compare_state_2, literacy_fmt),
        )

    gsdp_one, gsdp_two, gsdp_india, gsdp_fmt = comparison_data["GSDP"]
    with row_one_right:
        comparison_chart_or_message(
            "GSDP: lollipop comparison",
            [gsdp_one, gsdp_two, gsdp_india],
            lambda: lollipop_chart(
                "GSDP comparison", compare_state_1, compare_state_2,
                gsdp_one, gsdp_two, gsdp_india, "GSDP (Rs crore)", gsdp_fmt,
            ),
            comparison_insight("GSDP", gsdp_one, gsdp_two, gsdp_india, compare_state_1, compare_state_2, gsdp_fmt),
        )

    row_two_left, row_two_right = st.columns(2)
    unemployment_one, unemployment_two, unemployment_india, unemployment_fmt = comparison_data["Unemployment"]
    with row_two_left:
        comparison_chart_or_message(
            "Unemployment: progress against India average",
            [unemployment_one, unemployment_two, unemployment_india],
            lambda: bullet_chart(
                "Unemployment comparison", compare_state_1, compare_state_2,
                unemployment_one, unemployment_two, unemployment_india, "Unemployment rate (%)", unemployment_fmt,
            ),
            comparison_insight("unemployment", unemployment_one, unemployment_two, unemployment_india, compare_state_1, compare_state_2, unemployment_fmt, lower_is_better=True),
        )

    mortality_one, mortality_two, mortality_india, mortality_fmt = comparison_data["Infant Mortality"]
    with row_two_right:
        comparison_chart_or_message(
            "Infant mortality: mirrored comparison",
            [mortality_one, mortality_two, mortality_india],
            lambda: butterfly_chart(
                "Infant mortality comparison", compare_state_1, compare_state_2,
                mortality_one, mortality_two, mortality_india, "Infant mortality rate", mortality_fmt,
            ),
            comparison_insight("infant mortality", mortality_one, mortality_two, mortality_india, compare_state_1, compare_state_2, mortality_fmt, lower_is_better=True),
        )

    row_three_left, row_three_right = st.columns(2)
    population_one, population_two, population_india, population_fmt = comparison_data["Population"]
    with row_three_left:
        comparison_chart_or_message(
            "Population: proportional-circle comparison",
            [population_one, population_two, population_india],
            lambda: bubble_chart(
                "Population comparison", compare_state_1, compare_state_2,
                population_one, population_two, population_india, population_fmt,
            ),
            comparison_insight("population", population_one, population_two, population_india, compare_state_1, compare_state_2, population_fmt),
        )

    poverty_one, poverty_two, poverty_india, poverty_fmt = comparison_data["Poverty"]
    with row_three_right:
        comparison_chart_or_message(
            "Poverty: reference-dot comparison",
            [poverty_one, poverty_two, poverty_india],
            lambda: dot_reference_chart(
                "Poverty comparison", "Poverty", compare_state_1, compare_state_2,
                poverty_one, poverty_two, poverty_india, "Poverty ratio (%)", poverty_fmt,
            ),
            comparison_insight("poverty", poverty_one, poverty_two, poverty_india, compare_state_1, compare_state_2, poverty_fmt, lower_is_better=True),
        )

    row_four_left, row_four_right = st.columns(2)
    hdi_one, hdi_two, hdi_india, hdi_fmt = comparison_data["HDI"]
    with row_four_left:
        comparison_chart_or_message(
            "HDI: radial comparison",
            [hdi_one, hdi_two, hdi_india],
            lambda: radial_chart(
                "HDI comparison", compare_state_1, compare_state_2,
                hdi_one, hdi_two, hdi_india, hdi_fmt,
            ),
            comparison_insight("HDI", hdi_one, hdi_two, hdi_india, compare_state_1, compare_state_2, hdi_fmt),
        )
    with row_four_right:
        st.empty()

    st.dataframe(compare_table, width="stretch", hide_index=True)

if st.session_state.focus == "covid_impact":
    st.subheader("COVID-19 Impact Analysis")
    pre_year = 2019
    covid_years = (
        analysis_panel.loc[pd.to_numeric(analysis_panel["COVID Deaths"], errors="coerce").fillna(0) > 0, "Year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    post_year = max([year for year in covid_years if year >= 2020], default=2021)
    # COVID Deaths is sourced from the main CS661 dataset so the impact view stays
    # aligned with the project-wide development panel.
    covid_features = ["GSDP", "Per Capita Income", "Unemployment Rate", "Worker Participation Rate", "Poverty Rate", "HDI", "COVID Deaths"]
    covid_features = [col for col in covid_features if col in analysis_panel.columns]
    pre = analysis_panel[analysis_panel["Year"] == pre_year][["State"] + covid_features].copy()
    post = analysis_panel[analysis_panel["Year"] == post_year][["State"] + covid_features].copy()
    if page_filters["states"]:
        pre = pre[pre["State"].isin(page_filters["states"])]
        post = post[post["State"].isin(page_filters["states"])]
    covid_compare = pre.merge(post, on="State", suffixes=(f" {pre_year}", f" {post_year}"))
    for col in covid_features:
        covid_compare[f"{col} Change"] = covid_compare[f"{col} {post_year}"] - covid_compare[f"{col} {pre_year}"]
        base = covid_compare[f"{col} {pre_year}"].replace(0, np.nan)
        covid_compare[f"{col} Change %"] = (covid_compare[f"{col} Change"] / base) * 100
    if selected_state != "All India":
        covid_compare = covid_compare[covid_compare["State"] == selected_state]
    if covid_compare.empty:
        show_no_data_message("No COVID impact data matches the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("GSDP Change", fmt_value(safe_mean(covid_compare.get("GSDP Change %", pd.Series(dtype=float))), "{:.2f}%"))
        with c2:
            metric_card("PCI Change", fmt_value(safe_mean(covid_compare.get("Per Capita Income Change %", pd.Series(dtype=float))), "{:.2f}%"))
        with c3:
            metric_card("Unemployment Change", fmt_value(safe_mean(covid_compare.get("Unemployment Rate Change", pd.Series(dtype=float))), "{:.2f} pp"))
        with c4:
            metric_card("COVID Deaths", fmt_value(safe_mean(covid_compare.get(f"COVID Deaths {post_year}", pd.Series(dtype=float))), "{:,.0f}"))

        left, right = st.columns(2)
        with left:
            if "GSDP Change %" in covid_compare.columns and not covid_compare.empty:
                fig = px.bar(
                    covid_compare.sort_values("GSDP Change %").head(15),
                    x="GSDP Change %",
                    y="State",
                    orientation="h",
                    title=f"Largest GSDP Slowdowns or Recoveries, {pre_year}-{post_year}",
                    color="GSDP Change %",
                    color_continuous_scale="RdYlGn",
                )
            render_plotly(fig, width="stretch")
            chart_caption("States with the largest GSDP change between the pre- and post-COVID years.")
        with right:
            covid_death_col = f"COVID Deaths {post_year}"
            if {"Unemployment Rate Change", covid_death_col}.issubset(covid_compare.columns):
                covid_plot_data = covid_compare.dropna(subset=[covid_death_col, "Unemployment Rate Change"])
                if not covid_plot_data.empty:
                    covid_plot_data = prepare_size_column(covid_plot_data, f"GSDP {post_year}")
                    color_col = "GSDP Change %" if "GSDP Change %" in covid_plot_data.columns else None
                    fig = px.scatter(
                        covid_plot_data,
                        x=covid_death_col,
                        y="Unemployment Rate Change",
                        size="Bubble Size",
                        color=color_col,
                        hover_name="State",
                        title="Pandemic Burden vs Unemployment Shift",
                        color_continuous_scale="RdYlGn",
                    )
                    render_plotly(fig, width="stretch")
                    chart_caption("COVID deaths compared with unemployment change for the selected states.")
                else:
                    st.info("No valid data for pandemic burden vs unemployment analysis.")

if st.session_state.focus == "regional_patterns":
    st.subheader("Regional Development Pattern Analysis")
    selected_year = local_year_control("Cluster snapshot year", key="regional_year")
    if not SKLEARN_AVAILABLE:
        st.error("Install scikit-learn to use PCA and K-Means clustering.")
    else:
        cluster_features = [
            "GSDP",
            "Per Capita Income",
            "GSDP Growth",
            "Literacy Rate (%)",
            "HDI",
            "Life Expectancy",
            "Poverty Rate",
            "Unemployment Rate",
            "Infant Mortality",
            "MPI",
        ]
        snap = analysis_panel[analysis_panel["Year"] == selected_year].copy()
        if page_filters["states"]:
            snap = snap[snap["State"].isin(page_filters["states"])]
        if snap.empty:
            snap = analysis_panel[analysis_panel["Year"] == latest_available_year(analysis_panel, selected_year, "HDI")].copy()
            if page_filters["states"]:
                snap = snap[snap["State"].isin(page_filters["states"])]
        matrix, used_features = feature_matrix(snap, cluster_features)
        if len(matrix) < 5 or len(used_features) < 2:
            show_no_data_message("No data matches the selected filters for clustering.")
        else:
            k = st.slider("Number of clusters", min_value=2, max_value=min(6, len(matrix) - 1), value=3)
            X_scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(matrix[used_features])
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)
            clusters = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_scaled)
            cluster_df = matrix[["State", "Year"]].copy()
            cluster_df["PC1"] = coords[:, 0]
            cluster_df["PC2"] = coords[:, 1]
            cluster_df["Cluster ID"] = clusters
            cluster_df["Cluster"] = clusters.astype(str)
            cluster_df = cluster_df.join(matrix[used_features])

            r1, r2, r3 = st.columns(3)
            with r1:
                metric_card("Features Used", str(len(used_features)))
            with r2:
                metric_card("States Clustered", str(len(cluster_df)))
            with r3:
                metric_card("PCA Variance", f"{pca.explained_variance_ratio_.sum() * 100:.1f}%")

            left, right = st.columns(2)
            with left:
                fig = px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster", hover_name="State", title="PCA Scatter with K-Means Clusters")
                render_plotly(fig, width="stretch")
                chart_caption("PCA projection of state development patterns with K-Means clusters.")
            with right:
                cluster_sizes = cluster_df.groupby("Cluster", as_index=False).size().rename(columns={"size": "States"})
                fig = px.bar(
                    cluster_sizes,
                    x="Cluster",
                    y="States",
                    color="Cluster",
                    title="Cluster Sizes",
                )
                fig.update_layout(template="plotly_white")
                render_plotly(fig, width="stretch")
                chart_caption("Count of states in each cluster.")
                profile = cluster_df.groupby("Cluster", as_index=False)[used_features[:5]].mean()
                profile_long = profile.melt(id_vars="Cluster", var_name="Indicator", value_name="Average")
                fig = px.bar(profile_long, x="Indicator", y="Average", color="Cluster", barmode="group", title="Cluster Average Profile")
                fig.update_layout(template="plotly_white")
                render_plotly(fig, width="stretch")
                chart_caption("Average feature profile for each cluster.")
            parallel_df = downsample_frame(cluster_df, 1500)
            fig = px.parallel_coordinates(
                parallel_df,
                dimensions=used_features[:6],
                color=parallel_df["Cluster ID"],
                title="Cluster Profiles Across Key Indicators",
                color_continuous_scale="Turbo",
            )
            render_plotly(fig, width="stretch")
            chart_caption("Parallel coordinates view of the selected cluster features.")
            st.dataframe(cluster_df.sort_values("Cluster"), width="stretch", hide_index=True)

if st.session_state.focus == "forecasting":
    st.subheader("Forecasting Economic Indicators")
    if not SKLEARN_AVAILABLE:
        st.error("Install scikit-learn to use forecasting.")
    else:
        forecast_indicators = [col for col in ["GSDP", "Per Capita Income", "GSDP Growth", "Unemployment Rate", "Poverty Rate", "HDI"] if col in analysis_panel.columns]
        fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 1])
        with fcol1:
            forecast_indicator = st.selectbox("Forecast indicator", forecast_indicators, index=0)
        with fcol2:
            forecast_state = st.selectbox("Forecast geography", options=["All India"] + states, index=0)
        with fcol3:
            forecast_horizon = st.slider("Forecast horizon", min_value=3, max_value=10, value=5)

        gdp_forecast_map = {
            "GSDP": "gsdp_rs_crore",
            "Per Capita Income": "per_capita_income_rs",
            "GSDP Growth": "gsdp_growth_pct",
        }
        if forecast_indicator in gdp_forecast_map:
            source_col = gdp_forecast_map[forecast_indicator]
            if forecast_state == "All India":
                series_df = main[main["State/UT"].notna()].groupby("Year", as_index=False)[source_col].mean().rename(columns={source_col: forecast_indicator})
                series_label = "India Average"
            else:
                series_df = (
                    main[main["State/UT"] == forecast_state][["Year", source_col]]
                    .copy()
                    .rename(columns={source_col: forecast_indicator})
                )
                series_label = forecast_state
        elif forecast_state == "All India":
            series_df = analysis_panel.groupby("Year", as_index=False)[forecast_indicator].mean()
            series_label = "India Average"
        else:
            series_df = analysis_panel[analysis_panel["State"] == forecast_state][["Year", forecast_indicator]].copy()
            series_label = forecast_state
        result = temporal_forecast(series_df, forecast_indicator, forecast_horizon)
        if result is None:
            st.warning("Not enough yearly data for an 80/20 chronological train-test forecast.")
        else:
            history, test_out, forecast, metrics = result

            fig = go.Figure()
            validation_window = forecast[forecast["Year"].between(2024, 2026)]
            future_window = forecast[forecast["Year"] > 2026]
            actual_validation = validation_window[validation_window["Validation Status"] == "Used for validation"]
            flagged_validation = validation_window[validation_window["Validation Status"] == "Flagged partial/outlier actual"]
            fig.add_trace(go.Scatter(x=history["Year"], y=history["Value"], mode="lines+markers", name="Training actual", line=dict(color="#2563eb")))
            fig.add_trace(go.Scatter(x=test_out["Year"], y=test_out["Prediction"], mode="lines+markers", name="80/20 test prediction", line=dict(color="#f97316")))
            fig.add_trace(go.Scatter(x=validation_window["Year"], y=validation_window["Prediction"], mode="lines+markers", name="2024-2026 prediction", line=dict(color="#0f766e")))
            if not future_window.empty:
                fig.add_trace(go.Scatter(x=future_window["Year"], y=future_window["Prediction"], mode="lines+markers", name="Future forecast", line=dict(color="#7c3aed")))
            if not actual_validation.empty:
                fig.add_trace(
                    go.Scatter(
                        x=actual_validation["Year"],
                        y=actual_validation["Actual"],
                        mode="markers",
                        name="Real data validation",
                        marker=dict(size=13, symbol="diamond", color="#dc2626"),
                    )
                )
            if not flagged_validation.empty:
                fig.add_trace(
                    go.Scatter(
                        x=flagged_validation["Year"],
                        y=flagged_validation["Actual"],
                        mode="markers",
                        name="Flagged partial/outlier actual",
                        marker=dict(size=13, symbol="x", color="#f59e0b", line=dict(width=2)),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([forecast["Year"], forecast["Year"][::-1]]),
                    y=pd.concat([forecast["Upper"], forecast["Lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(37,99,235,0.16)",
                    line={"color": "rgba(255,255,255,0)"},
                    name="Approx. 95% interval",
                )
            )
            fig.update_layout(
                title=f"{forecast_indicator} Forecast - {series_label}",
                xaxis_title="Year",
                yaxis_title=forecast_indicator,
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#111827"),
                legend=dict(bgcolor="rgba(255,255,255,0.78)", bordercolor="rgba(31,41,55,0.12)", borderwidth=1),
                margin=dict(l=60, r=35, t=70, b=55),
            )
            fig.update_xaxes(gridcolor="rgba(31,41,55,0.14)", zerolinecolor="rgba(31,41,55,0.18)", color="#111827")
            fig.update_yaxes(gridcolor="rgba(31,41,55,0.14)", zerolinecolor="rgba(31,41,55,0.18)", color="#111827")
            render_plotly(fig, width="stretch")
            chart_caption(f"Forecast path for {forecast_indicator} in {series_label}.")
            st.caption("Zero placeholders and incomplete 2024 actuals are ignored. Only non-excluded real-world values are used for validation.")
            with st.expander("Model validation scores"):
                st.dataframe(metrics["Model Scores"].sort_values("RMSE"), width="stretch", hide_index=True)
            st.dataframe(
                forecast[["Year", "Prediction", "Actual", "Validation Status", "Error", "Absolute Error", "Lower", "Upper"]],
                width="stretch",
                hide_index=True,
            )

if st.session_state.focus == "classification":
    st.subheader("Development Category Classification")
    selected_year = local_year_control("Classification snapshot year", key="classification_year")
    if not SKLEARN_AVAILABLE:
        st.error("Install scikit-learn to use Random Forest classification.")
    else:
        class_features = [
            "GSDP",
            "Per Capita Income",
            "GSDP Growth",
            "PCI Growth",
            "Literacy Rate (%)",
            "Poverty Rate",
            "Unemployment Rate",
            "Worker Participation Rate",
            "MPI",
            "Life Expectancy",
            "Infant Mortality",
            "Total Population",
            "Working Population",
        ]
        labelled = add_development_labels(analysis_panel)
        if page_filters["states"]:
            labelled = labelled[labelled["State"].isin(page_filters["states"])]
        labelled = labelled[labelled["Year"].between(*page_filters["year_range"])]
        labelled = labelled[labelled["Development Category"] != "Unknown"].copy()
        matrix, used_features = feature_matrix(labelled, class_features)
        model_df = matrix.merge(
            labelled[["State", "Year", "Development Category", "Development Score"]],
            on=["State", "Year"],
            how="left",
        ).dropna(subset=["Development Category"])
        if len(model_df) < 20 or len(used_features) < 3:
            show_no_data_message("No labelled data matches the selected filters for classification.")
        else:
            X = model_df[used_features]
            y = model_df["Development Category"]
            stratify = y if y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Train Rows", str(len(X_train)))
            with c2:
                metric_card("Test Rows", str(len(X_test)))
            with c3:
                metric_card("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            with c4:
                metric_card("Features", str(len(used_features)))

            rf = clf.named_steps["randomforestclassifier"]
            importance = pd.DataFrame({"Feature": used_features, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False)
            left, right = st.columns(2)
            with left:
                fig = px.bar(importance.head(12), x="Importance", y="Feature", orientation="h", title="Random Forest Feature Importance")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                render_plotly(fig, width="stretch")
                chart_caption("Most important features used by the classifier.")
            with right:
                report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})
                st.dataframe(report_df, width="stretch", hide_index=True)

            current = labelled[labelled["Year"] == selected_year].copy()
            if current.empty:
                current = labelled[labelled["Year"] == latest_available_year(labelled, selected_year, "Development Score")].copy()
            current_matrix, current_features = feature_matrix(current, used_features)
            if current_features:
                predicted = current_matrix[["State", "Year"]].copy()
                predicted["Predicted Category"] = clf.predict(current_matrix[used_features])
                current = current.merge(predicted, on=["State", "Year"], how="left")
                display_cols = ["State", "Year", "Development Score", "Development Category", "Predicted Category"] + used_features[:5]
                st.dataframe(current[display_cols].sort_values("Development Score", ascending=False), width="stretch", hide_index=True)

st.caption(f"Dataset coverage: {len(states)} states/UTs, {years[0]} to {years[-1]} in the development panel, plus supporting sector, policy, poverty, literacy, population, mortality, and employment files from this folder.")
