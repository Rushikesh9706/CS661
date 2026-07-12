import re
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="BharatScope",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    return {
        "main": pd.read_csv("CS661 Dataset - Sheet1.csv"),
        "gdp": pd.read_excel("GDP.xlsx"),
        "employment": pd.read_csv("employment data.csv"),
        "policy": pd.read_excel("Indian Policy Timeline.xlsx"),
        "infant": pd.read_excel("Infant Mortality.xlsx"),
        "literacy": pd.read_excel("Literacy rate.xlsx", header=None),
        "population": pd.read_excel("population.xlsx"),
        "poverty": pd.read_csv("all_states_interpolated_poverty.csv"),
        "sector": pd.read_excel("agriculture_industrial.xlsx"),
    }


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


def clean_frames(raw):
    main = raw["main"].copy()
    gdp = raw["gdp"].copy()
    emp = raw["employment"].copy()
    policy = raw["policy"].copy()
    infant = raw["infant"].copy()
    literacy = raw["literacy"].copy()
    population = raw["population"].copy()
    poverty = raw["poverty"].copy()
    sector = raw["sector"].copy()

    main["State/UT"] = main["State/UT"].map(normalize_state)
    gdp["State/UT"] = gdp["State/UT"].map(normalize_state)
    emp["State"] = emp["State"].map(normalize_state)
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
        if col in main.columns:
            main[col] = to_number(main[col])

    gdp["Year"] = to_number(gdp["Year"])
    gdp["gsdp_rs_crore"] = to_number(gdp["gsdp_rs_crore"])
    gdp["gsdp_growth_pct"] = to_number(gdp["gsdp_growth_pct"])
    emp["Year"] = to_number(emp["Year"])
    emp["WPR (%)"] = to_number(emp["WPR (%)"])
    emp["UR (%)"] = to_number(emp["UR (%)"])
    policy["Launch_Year"] = to_number(policy["Launch_Year"])
    policy["Start_Year"] = to_number(policy["Start_Year"])
    policy["End_Year"] = policy["End_Year"].replace("Present", 2026)
    policy["End_Year"] = to_number(policy["End_Year"])
    poverty.columns = [normalize_state(c) if isinstance(c, str) else c for c in poverty.columns]
    for col in poverty.columns:
        if col != "State":
            poverty[col] = to_number(poverty[col])
    for col in sector.columns:
        if col != "Financial Year":
            sector[col] = to_number(sector[col])
    sector["Year"] = sector["Financial Year"].astype(str).str.extract(r"(\d{4})").astype(float)
    sector["Year"] = to_number(sector["Year"])

    return {
        "main": main,
        "gdp": gdp,
        "employment": emp,
        "policy": policy,
        "infant": infant,
        "literacy": literacy,
        "population": population,
        "poverty": poverty,
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


def wide_year_to_long(df, state_col, value_prefix="value"):
    year_cols = [c for c in df.columns if isinstance(c, (int, float)) or str(c).isdigit()]
    out = df[[state_col] + year_cols].melt(id_vars=state_col, var_name="Year", value_name=value_prefix)
    out["Year"] = to_number(out["Year"])
    out[value_prefix] = to_number(out[value_prefix])
    return out.dropna(subset=["Year"])


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
    for idx in range(1, len(data.columns)):
        data.iloc[:, idx] = pd.to_numeric(
            data.iloc[:, idx].replace(["na", "NA", "n.a.", "-", "--"], pd.NA),
            errors="coerce",
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


raw = load_data()
data = clean_frames(raw)
main = data["main"]
gdp = data["gdp"]
emp = data["employment"]
policy = data["policy"]
infant = data["infant"]
literacy = data["literacy"]
population = data["population"]
poverty = data["poverty"]
sector = data["sector"]

states = main_state_list(main, gdp, emp, poverty)
years = sorted(main["Year"].dropna().astype(int).unique().tolist())
if "focus" not in st.session_state:
    st.session_state.focus = "overview"


st.markdown(
    """
    <style>
        :root {
            --bg1: #f3efe4;
            --bg2: #e8efe8;
            --ink: #1f2937;
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
            color: var(--ink);
        }
        .hero {
            background: linear-gradient(135deg, rgba(17,24,39,0.97), rgba(55,65,81,0.93));
            color: #fff;
            border-radius: 30px;
            padding: 2.1rem 2.2rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 20px 50px rgba(15,23,42,0.20);
        }
        .hero h1 {
            margin: 0;
            font-size: 3rem;
            line-height: 1;
        }
        .hero p {
            margin: 0.55rem 0 0 0;
            color: rgba(255,255,255,0.8);
        }
        .section {
            margin-top: 1rem;
            font-size: 1.1rem;
            font-weight: 800;
        }
        .metric-card, .tile {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
        }
        .metric-label { color: var(--muted); font-size: 0.87rem; margin-bottom: 0.3rem; }
        .metric-value { font-size: 1.65rem; font-weight: 800; line-height: 1.05; }
        .delta { margin-top: 0.25rem; font-size: 0.82rem; color: var(--teal); }
        .nav-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.8rem; }
        .nav-note { color: var(--muted); font-size: 0.9rem; margin-top: 0.35rem; }
        .pill {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: rgba(29,78,216,0.08);
            color: var(--blue);
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

focus_options = ["Overview", "Economic Growth", "Policy Impact", "Human Development", "State Compare"]

with st.sidebar:
    st.title("BharatScope")
    st.caption("Indian Economic Development Analytics Platform")
    current = next((f for f in focus_options if f.lower().replace(" ", "_") == st.session_state.focus), "Overview")
    choice = st.radio("Analysis lens", focus_options, index=focus_options.index(current))
    st.session_state.focus = choice.lower().replace(" ", "_")
    st.divider()
    if st.session_state.focus in {"economic_growth", "policy_impact"}:
        selected_year = st.select_slider("Year", options=list(range(1990, 2024)), value=2023)
    elif st.session_state.focus in {"human_development", "state_compare"}:
        selected_year = years[-1]
    else:
        selected_year = st.select_slider("Year", options=years, value=years[-1])
    selected_state = st.selectbox("State / UT", options=["All India"] + states, index=0)

selected_rows_main = main[main["Year"] == selected_year].copy()
selected_rows_gdp = gdp[gdp["Year"] == selected_year].copy()
if selected_state != "All India":
    selected_rows_main = selected_rows_main[selected_rows_main["State/UT"] == selected_state]
    selected_rows_gdp = selected_rows_gdp[selected_rows_gdp["State/UT"] == selected_state]

st.markdown(
    """
    <div class="hero">
        <h1>BharatScope</h1>
        <p>Indian Economic Development Analytics Platform</p>
        <p>Explore state development, policy eras, human outcomes, and the structural shift across agriculture, industry, and services using the datasets in this folder.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section'>Choose a lens</div>", unsafe_allow_html=True)
nav_cols = st.columns(2)
nav_cards = [
    ("Economic Growth", "GSDP and per-capita performance across states."),
    ("Policy Impact", "Compare reforms, unemployment, and poverty proxies."),
    ("Human Development", "Literacy, life expectancy, and mortality trends."),
    ("State Compare", "Benchmark one state against India and peers."),
]
for i, (label, desc) in enumerate(nav_cards):
    with nav_cols[i % 2]:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.focus = label.lower().replace(" ", "_")
        st.caption(desc)
st.markdown("---")

overview = st.session_state.focus in {"overview", "economic_growth"}

if overview or st.session_state.focus == "economic_growth":
    st.subheader("Economic Growth")
    sector_year = sector[sector["Year"] == (selected_year - 1)].copy()
    sector_year = sector_year.dropna(subset=["Year"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("GSDP", f"?{selected_rows_gdp['gsdp_rs_crore'].mean():,.0f} Cr" if not selected_rows_gdp.empty else "N/A", "Selected year/state")
    with c2:
        metric_card("GSDP Growth", f"{selected_rows_gdp['gsdp_growth_pct'].mean():.2f}%" if not selected_rows_gdp.empty else "N/A")
    with c3:
        metric_card("Per Capita Income", f"?{selected_rows_gdp['per_capita_income_rs'].mean():,.0f}" if not selected_rows_gdp.empty else "N/A")
    with c4:
        metric_card("PCI Growth", f"{selected_rows_gdp['pci_growth_pct'].mean():.2f}%" if not selected_rows_gdp.empty else "N/A")

    s1, s2, s3 = st.columns(3)
    with s1:
        metric_card("Agriculture Share", f"{sector_year['Agriculture - Share to Total GDP'].mean():.2f}%" if not sector_year.empty else "N/A")
    with s2:
        metric_card("Industry Share", f"{sector_year['Industry - Share to Total GDP'].mean():.2f}%" if not sector_year.empty else "N/A")
    with s3:
        metric_card("Services Share", f"{sector_year['Services - Share to Total GDP'].mean():.2f}%" if not sector_year.empty else "N/A")

    left, right = st.columns(2)
    with left:
        fig = px.line(
            gdp,
            x="Year",
            y="gsdp_rs_crore",
            color="State/UT",
            title="GSDP Trends by State",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        top_gdp = selected_rows_gdp.sort_values("gsdp_rs_crore", ascending=False).head(10)
        fig = px.bar(top_gdp, x="State/UT", y="gsdp_rs_crore", color="gsdp_growth_pct", title=f"Top GSDP States in {selected_year}")
        st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        sector_trend = sector.dropna(subset=["Year"])[
            ["Year", "Agriculture - Share to Total GDP", "Industry - Share to Total GDP", "Services - Share to Total GDP"]
        ].copy()
        sector_trend = sector_trend.melt(id_vars="Year", var_name="Sector", value_name="Share")
        fig = px.line(sector_trend, x="Year", y="Share", color="Sector", title="Sector Share in GDP")
        st.plotly_chart(fig, use_container_width=True)
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
            fig = px.bar(sector_cards, x="Sector", y="Share", title=f"Sector Share Snapshot ({int(sector_latest['Year'].mean())})")
            st.plotly_chart(fig, use_container_width=True)

if st.session_state.focus == "policy_impact":
    st.subheader("Policy Impact")
    policy_years = sorted(set(emp["Year"].dropna().astype(int).unique()) & set(poverty["Year"].dropna().astype(int).unique()))
    policy_year = selected_year if selected_year in policy_years else (policy_years[-1] if policy_years else int(emp["Year"].dropna().max()))
    policy_emp = emp[emp["Year"] == policy_year].copy()
    policy_main = main[main["Year"] == policy_year].copy()
    poverty_scope = selected_state if selected_state != "All India" else "All India"
    policy_poverty = poverty_value_from_csv(poverty, policy_year, poverty_scope)
    policy_gdp = gdp[gdp["Year"] == policy_year].copy()
    policy_sector_year = sector[sector["Year"] == (policy_year - 1)].copy()

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

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        metric_card("GSDP", f"₹{policy_gdp['gsdp_rs_crore'].mean():,.0f} Cr" if not policy_gdp.empty else "N/A")
    with g2:
        metric_card(
            "Agriculture Share",
            f"{policy_sector_year['Agriculture - Share to Total GDP'].mean():.2f}%" if not policy_sector_year.empty else "N/A",
        )
    with g3:
        metric_card(
            "Industry Share",
            f"{policy_sector_year['Industry - Share to Total GDP'].mean():.2f}%" if not policy_sector_year.empty else "N/A",
        )
    with g4:
        metric_card(
            "Services Share",
            f"{policy_sector_year['Services - Share to Total GDP'].mean():.2f}%" if not policy_sector_year.empty else "N/A",
        )

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
        fig = px.line(
            policy_trend,
            x="Year",
            y="Value",
            color="Indicator",
            title="Policy-Linked Indicators",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        timeline = policy.copy()
        timeline["Before"] = timeline["Before_Period"].astype(str)
        timeline["After"] = timeline["After_Period"].astype(str)
        fig = px.timeline(
            timeline,
            x_start="Start_Year",
            x_end="End_Year",
            y="Policy_Name",
            color="Category",
            title="Policy Timeline",
        )
        st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        sector_slice = sector.dropna(subset=["Year"])[
            ["Year", "Agriculture - Share to Total GDP", "Industry - Share to Total GDP", "Services - Share to Total GDP"]
        ].copy()
        sector_slice = sector_slice.melt(id_vars="Year", var_name="Sector", value_name="Share")
        fig = px.line(sector_slice, x="Year", y="Share", color="Sector", title="Sector Shares Over Time")
        st.plotly_chart(fig, use_container_width=True)
    with lower_right:
        if not policy_sector_year.empty:
            sector_summary = pd.DataFrame(
                {
                    "Sector": ["Agriculture", "Industry", "Services"],
                    "Share": [
                        policy_sector_year["Agriculture - Share to Total GDP"].mean(),
                        policy_sector_year["Industry - Share to Total GDP"].mean(),
                        policy_sector_year["Services - Share to Total GDP"].mean(),
                    ],
                }
            )
            fig = px.bar(sector_summary, x="Sector", y="Share", title=f"Sector Share Snapshot ({policy_year})")
            st.plotly_chart(fig, use_container_width=True)
    st.dataframe(policy[["Policy_Name", "Category", "Launch_Year", "Description"]], use_container_width=True, hide_index=True)

if st.session_state.focus == "human_development":
    st.subheader("Human Development")
    literacy_long = parse_literacy_sheet(literacy)
    poverty_state = poverty.copy()
    poverty_state = poverty_state.rename(columns={poverty_state.columns[0]: "State"})
    poverty_state["State"] = poverty_state["State"].map(normalize_state)
    poverty_state = poverty_state.melt(id_vars="State", var_name="Year", value_name="Poverty")
    poverty_state["Year"] = to_number(poverty_state["Year"])
    poverty_state["Poverty"] = to_number(poverty_state["Poverty"])
    human_year = st.radio("Select year", [1991, 2001, 2011], index=0, horizontal=True)
    human_rows = main[main["Year"] == human_year].copy()
    if selected_state != "All India":
        human_rows = human_rows[human_rows["State/UT"] == selected_state]
    lit_year = literacy_long[["State", human_year]].rename(columns={human_year: "Literacy"})
    lit_year = lit_year.dropna(subset=["Literacy"])
    if selected_state != "All India":
        lit_year = lit_year[lit_year["State"] == selected_state]
    pov_year = poverty_state[poverty_state["Year"] == human_year]
    if selected_state != "All India":
        pov_year = pov_year[pov_year["State"] == selected_state]
    pop_year = population[["Region", human_year]] if human_year in population.columns else population[[population.columns[0], 2011]]
    pop_year = pop_year.rename(columns={pop_year.columns[0]: "State", human_year if human_year in pop_year.columns else 2011: "Population"})
    pop_year["State"] = pop_year["State"].map(normalize_state)
    if selected_state != "All India":
        pop_year = pop_year[pop_year["State"] == selected_state]
    imr_state = infant.rename(columns={infant.columns[0]: "State"}).copy()
    imr_state["State"] = imr_state["State"].map(normalize_state)
    imr_state = imr_state.rename(columns={c: int(c) if str(c).isdigit() else c for c in imr_state.columns})
    imr_years = [c for c in imr_state.columns if c != "State" and c in {1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2011}]
    imr_long = imr_state.melt(id_vars="State", value_vars=imr_years, var_name="Year", value_name="IMR")
    imr_long["Year"] = to_number(imr_long["Year"])
    imr_long["IMR"] = to_number(imr_long["IMR"])
    if selected_state != "All India":
        imr_long = imr_long[imr_long["State"] == selected_state]

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
        fig = px.line(lit_plot, x="Year", y="Literacy", color="State", title="Literacy Across States")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(imr_long, x="Year", y="IMR", color="State", title="Infant Mortality Trend")
        st.plotly_chart(fig, use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        pop_long = population.melt(id_vars="Region", var_name="Year", value_name="Population")
        pop_long["Year"] = to_number(pop_long["Year"])
        pop_long["Population"] = to_number(pop_long["Population"].replace("State not formed", pd.NA))
        pop_long = pop_long.dropna(subset=["Population"])
        pop_long["Region"] = pop_long["Region"].map(normalize_state)
        if selected_state != "All India":
            pop_long = pop_long[pop_long["Region"] == selected_state]
        fig = px.line(pop_long[pop_long["Year"] >= 1951], x="Year", y="Population", color="Region", title="Population Growth Snapshot")
        st.plotly_chart(fig, use_container_width=True)
    with bottom_right:
        lit_year_plot = lit_year.sort_values("Literacy", ascending=False).head(12)
        fig = px.bar(lit_year_plot, x="State", y="Literacy", title=f"Literacy Ranking in {human_year}")
        st.plotly_chart(fig, use_container_width=True)

if st.session_state.focus == "state_compare":
    st.subheader("State Compare")
    compare_col1, compare_col2, compare_col3 = st.columns([1, 1, 1])
    with compare_col1:
        compare_state_1 = st.selectbox(
            "Pick first state/UT",
            options=states,
            index=states.index(selected_state) if selected_state in states else 0,
            key="compare_state_1",
        )
    with compare_col2:
        second_options = [state for state in states if state != compare_state_1] or states
        compare_state_2 = st.selectbox(
            "Pick second state/UT",
            options=second_options,
            index=0,
            key="compare_state_2",
        )
    with compare_col3:
        compare_year = st.radio("Select year", [1991, 2001, 2011], index=0, horizontal=True)

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

    chart_rows = []
    for state_name in [compare_state_1, compare_state_2, "India Avg"]:
        for spec in metric_specs:
            value = spec["india"] if state_name == "India Avg" else spec["value"](state_name)
            chart_rows.append({"State/UT": state_name, "Indicator": spec["label"], "Value": value})

    chart_df = pd.DataFrame(chart_rows)
    fig = px.bar(
        chart_df,
        x="Indicator",
        y="Value",
        color="State/UT",
        barmode="group",
        title=f"{compare_state_1} and {compare_state_2} vs India average in {compare_year}",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(compare_table, use_container_width=True, hide_index=True)

st.caption(f"Dataset coverage: {len(states)} states/UTs, {years[0]} to {years[-1]} in the development panel, plus supporting sector, policy, poverty, literacy, population, mortality, and employment files from this folder.")
