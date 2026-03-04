import re
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="PSM UMPSA Research and Publication Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "PSM UMPSA Research and Publication Dashboard"
APP_BRIEF = (
    "This interactive application provides a consolidated view of PSM UMPSA's for research grants and "
    "Research Outputs (Publications & Citations) from PNI UMPSA reporting workbooks. " 
)

# Hardcoded citation cumulative stats (from Summary sheet requirement)
CUMULATIVE_CITATIONS = {2021: 23, 2022: 147, 2023: 349, 2024: 520, 2025: 705}
CUMULATIVE_TOTAL_2021_2025 = 1744
TOTAL_CITATIONS_ALL = 1777

# Exact column names (Grants file)
GRANT_COLUMNS = [
    "No", "Vote", "Staff ID", "Name", "Research Title", "Role", "Start Date", "End Date",
    "Extension Date", "Faculty", "FOR (Category)", "Grant Name", "Approved Amount (RM)",
    "Sponsor", "Sponsor Category",
]

# Citation sheet exact column names
CITATION_COLUMNS = [
    "Publication Year", "Document Title", "ISSN", "Journal Title", "Volume", "Issue",
    "<2021", "2021", "2022", "2023", "2024", "2025", "Subtotal", ">2025", "Total",
]

# Index sheet exact column names (Article + Conference)
INDEX_COLUMNS = [
    "DATE", "NO", "STATUS", "ROLE", "STAF ID", "STAF",
    "CO-AUTHOR", "CO-AUTHOR NAME", "CO-AUTHOR2", "CO-AUTHOR NAME2",
    "AUTHOR", "AUTHOR WITH AFFILIATION", "FACULTY 1", "FACULTY 2",
    "TITLE", "SOURCE TITLE",
    "QUARTILE 2022", "JIF 2022", "Q 2023", "JIF 2023",
    "QUARTILE 2023", "JIF 2023(2)", "QUARTILE 2024", "JIF 2024",
    "QUARTILE 2025", "JIF 2025",
    "DOI", "CITED BY", "CITATION",
    "YEAR", "MONTH", "TYPE", "SJR", "SNIP", "CITESCORE",
    "INDEXING", "PUBLISHER", "AFFILIATION", "COUNTRY",
    "FIELD", "FOR", "SCOPUS", "WOS", "ERA",
    "CATEGORY", "RANK", "REMARKS",
    "URL", "PDF", "LINK",
]


# =========================
# ROBUST HELPERS (NO .str CRASH)
# =========================
def clean_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c = "" if c is None else str(c)
        c = re.sub(r"\s+", " ", c).strip()
        out.append(c)
    return out


def clean_staff_id(series: pd.Series) -> pd.Series:
    # Handles apostrophe prefix and preserves leading zeros as string
    s = series.astype("string")
    s = s.str.replace("'", "", regex=False).str.strip()
    s = s.str.replace(r"[^0-9A-Za-z]", "", regex=True)
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def drop_nan_heavy_rows(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df.isna().mean(axis=1) <= threshold].copy()


def safe_str_contains(series: pd.Series, q: str) -> pd.Series:
    # Always safe: cast to pandas string dtype first
    s = series.astype("string")
    return s.str.contains(q, case=False, na=False)


def global_search(df: pd.DataFrame, q: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Safe global search across columns of any dtype (numeric/datetime won't crash).
    """
    if not q:
        return df
    if cols is None:
        cols = list(df.columns)

    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        if c not in df.columns:
            continue
        try:
            mask |= safe_str_contains(df[c], q).to_numpy()
        except Exception:
            # ultimate fallback
            mask |= df[c].astype(str).str.contains(q, case=False, na=False).to_numpy()

    return df.loc[mask].copy()


def format_rm(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"RM {x:,.0f}"


def make_download_button(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# =========================
# LOADERS (CACHED)
# =========================
@st.cache_data(show_spinner=False)
def load_grants_workbook(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Grants workbook sheets:
    - AKTIF 2026
    - PI 2026
    Header row at index 3 (0-based): raw.iloc[3]
    """
    out: Dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(file_bytes)

    for sheet in ["AKTIF 2026", "PI 2026"]:
        if sheet not in xls.sheet_names:
            out[sheet] = pd.DataFrame()
            continue

        raw = pd.read_excel(file_bytes, sheet_name=sheet, header=None)
        header_row = clean_columns(raw.iloc[3].tolist())

        df = raw.iloc[4:].copy()
        df.columns = header_row

        keep = [c for c in GRANT_COLUMNS if c in df.columns]
        df = df.loc[:, keep].copy()
        df = drop_nan_heavy_rows(df, threshold=0.85)

        if "Staff ID" in df.columns:
            df["Staff ID"] = clean_staff_id(df["Staff ID"])
        if "Approved Amount (RM)" in df.columns:
            df["Approved Amount (RM)"] = to_numeric(df["Approved Amount (RM)"])

        for dcol in ["Start Date", "End Date", "Extension Date"]:
            if dcol in df.columns:
                df[dcol] = to_datetime(df[dcol])

        # trim text columns safely
        for c in df.columns:
            if df[c].dtype == "object":
                df[c] = df[c].astype("string").str.strip()

        out[sheet] = df.reset_index(drop=True)

    return out


@st.cache_data(show_spinner=False)
def load_publications_workbook(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Publications workbook sheets of interest:
    - citation (data starts row index 7; columns fixed to 15)
    - index (ARTICLE section then CONFERENCE section)
    Summary cumulative citations is hardcoded per requirement.
    """
    out: Dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(file_bytes)

    # Summary - hardcoded cumulative citations
    out["Summary - Cumulative Citations (Hardcoded)"] = pd.DataFrame({
        "Year": list(CUMULATIVE_CITATIONS.keys()),
        "Cumulative Citations": list(CUMULATIVE_CITATIONS.values()),
    })

    # citation sheet
    if "citation" in xls.sheet_names:
        cit_raw = pd.read_excel(file_bytes, sheet_name="citation", header=None)
        df_cit = cit_raw.iloc[7:].copy().iloc[:, :15]
        df_cit.columns = CITATION_COLUMNS
        df_cit = drop_nan_heavy_rows(df_cit, threshold=0.80)

        df_cit["Publication Year"] = to_numeric(df_cit["Publication Year"])
        for c in ["<2021", "2021", "2022", "2023", "2024", "2025", "Subtotal", ">2025", "Total"]:
            df_cit[c] = to_numeric(df_cit[c])

        for c in ["Document Title", "ISSN", "Journal Title", "Volume", "Issue"]:
            df_cit[c] = df_cit[c].astype("string").str.strip()

        out["citation"] = df_cit.reset_index(drop=True)
    else:
        out["citation"] = pd.DataFrame()

    # index sheet: ARTICLE + CONFERENCE
    if "index" in xls.sheet_names:
        idx_raw = pd.read_excel(file_bytes, sheet_name="index", header=None)

        conf_marker = None
        for i in range(len(idx_raw)):
            v = idx_raw.iat[i, 0]
            if isinstance(v, str) and v.strip().upper() == "CONFERENCE":
                conf_marker = i
                break

        # ARTICLE
        df_art = idx_raw.iloc[4:(conf_marker if conf_marker is not None else len(idx_raw))].copy()
        df_art = df_art.iloc[:, :len(INDEX_COLUMNS)]
        df_art.columns = INDEX_COLUMNS
        df_art = drop_nan_heavy_rows(df_art, threshold=0.80)

        # CONFERENCE
        df_conf = pd.DataFrame(columns=INDEX_COLUMNS)
        if conf_marker is not None:
            df_conf = idx_raw.iloc[(conf_marker + 2):].copy()
            df_conf = df_conf.iloc[:, :len(INDEX_COLUMNS)]
            df_conf.columns = INDEX_COLUMNS
            df_conf = drop_nan_heavy_rows(df_conf, threshold=0.80)

        for df in [df_art, df_conf]:
            if df.empty:
                continue
            df["DATE"] = to_datetime(df["DATE"])
            df["STAF ID"] = clean_staff_id(df["STAF ID"])
            if "CITED BY" in df.columns:
                df["CITED BY"] = to_numeric(df["CITED BY"])
            for c in df.columns:
                if df[c].dtype == "object":
                    df[c] = df[c].astype("string").str.strip()

        out["index - ARTICLE"] = df_art.reset_index(drop=True)
        out["index - CONFERENCE"] = df_conf.reset_index(drop=True)
    else:
        out["index - ARTICLE"] = pd.DataFrame()
        out["index - CONFERENCE"] = pd.DataFrame()

    return out


# =========================
# SIDEBAR: BRANDING + DATA SOURCES
# =========================
#st.sidebar.title("📌 Dashboard")

# NOTE about file:/// link:
# Streamlit server cannot access your Windows D: drive.
# So we provide uploader OR relative path support.
# Sidebar - Logo and Authors
st.sidebar.image(
    "https://psm.umpsa.edu.my/images/ptj-umpsa.svg",  # Replace with your image URL or local path
    #caption="PSM UMPSA",
    use_container_width=True
)
st.sidebar.header("Developers:")
Developers = [
    "Ku Muhammad Naim Ku Khalif"
    # Add additional author names here
]
  
for developer in Developers:
    st.sidebar.write(f"- {developer}")

with st.sidebar.expander("📂 Data Sources (Drag & Drop)", expanded=True):
    st.caption("Upload both Excel files. The app will read them directly.")
    grants_file = st.file_uploader(
        "Grants workbook (DATA PELAPORAN GERAN AKTIF...)",
        type=["xlsx"],
        key="grants_uploader",
    )
    pubs_file = st.file_uploader(
        "Publications workbook (PSM as at Disember 2025...)",
        type=["xlsx"],
        key="pubs_uploader",
    )

with st.sidebar.expander("⚙️ Performance", expanded=False):
    st.write("Fast reload with `@st.cache_data`.")
    if st.button("Clear cached data", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")


# =========================
# NAVIGATION (SPLIT GRANTS vs PUBLICATIONS)
# =========================
section = st.sidebar.radio(
    "Module",
    ["Home", "Grants • Interactive", "Publications • Interactive", "Data Explorer"],
)

# stop early if uploads missing (except Home)
if section != "Home" and (grants_file is None or pubs_file is None):
    st.title(APP_TITLE)
    st.markdown(APP_BRIEF)
    st.warning("Please upload BOTH Excel files in the sidebar to proceed.")
    st.stop()

# load once uploaded
if grants_file is not None and pubs_file is not None:
    grants = load_grants_workbook(grants_file.getvalue())
    pubs = load_publications_workbook(pubs_file.getvalue())
else:
    grants, pubs = {}, {}

df_aktif = grants.get("AKTIF 2026", pd.DataFrame())
df_pi = grants.get("PI 2026", pd.DataFrame())

df_cit = pubs.get("citation", pd.DataFrame())
df_idx_art = pubs.get("index - ARTICLE", pd.DataFrame())
df_idx_conf = pubs.get("index - CONFERENCE", pd.DataFrame())
df_sum_cum = pubs.get("Summary - Cumulative Citations (Hardcoded)", pd.DataFrame())


# =========================
# HOME
# =========================
if section == "Home":
    st.title(APP_TITLE)
    st.markdown(APP_BRIEF)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What you can do (Grants)")
        st.markdown(
            "- Explore KPIs (counts, totals, unique researchers)\n"
            "- Filter by Sponsor Category / Faculty / FOR Category\n"
            "- Search by vote / researcher / sponsor / title\n"
            "- Interactive charts + downloadable tables"
        )
    with c2:
        st.subheader("What you can do (Publications)")
        st.markdown(
            "- View Article vs Conference interactively\n"
            "- Researcher profiles (by Staff ID / Name)\n"
            "- Citation analytics (Top cited documents)\n"
            "- Safe global search (no dtype crashes)"
        )

    st.info("Upload both Excel files in the sidebar to activate the interactive modules.")


# =========================
# GRANTS MODULE (INTERACTIVE PRESENTATION)
# =========================
elif section == "Grants • Interactive":
    st.title("Grants • Interactive Presentation")
    st.caption("AKTIF 2026 + PI 2026 (interactive filtering, charts, tables)")

    if df_aktif.empty:
        st.warning("No data found in 'AKTIF 2026'. Please verify the uploaded file contains this sheet.")
        st.stop()

    # Filters
    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)

        sponsor_cat = sorted(df_aktif.get("Sponsor Category", pd.Series(dtype="string")).dropna().astype("string").unique())
        faculty = sorted(df_aktif.get("Faculty", pd.Series(dtype="string")).dropna().astype("string").unique())
        for_cat = sorted(df_aktif.get("FOR (Category)", pd.Series(dtype="string")).dropna().astype("string").unique())

        sel_sponsor = f1.multiselect("Sponsor Category", sponsor_cat, default=sponsor_cat)
        sel_faculty = f2.multiselect("Faculty", faculty, default=faculty)
        sel_for = f3.multiselect("FOR (Category)", for_cat, default=for_cat)
        q = f4.text_input("Search (Vote / Name / Title / Sponsor)", value="").strip()

    dff = df_aktif.copy()
    if sel_sponsor and "Sponsor Category" in dff.columns:
        dff = dff[dff["Sponsor Category"].astype("string").isin(sel_sponsor)]
    if sel_faculty and "Faculty" in dff.columns:
        dff = dff[dff["Faculty"].astype("string").isin(sel_faculty)]
    if sel_for and "FOR (Category)" in dff.columns:
        dff = dff[dff["FOR (Category)"].astype("string").isin(sel_for)]

    if q:
        dff = global_search(dff, q, cols=[c for c in ["Vote", "Name", "Research Title", "Sponsor", "Grant Name"] if c in dff.columns])

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Grants (filtered)", f"{len(dff):,}")
    k2.metric("Total Approved Amount", format_rm(dff["Approved Amount (RM)"].sum() if "Approved Amount (RM)" in dff.columns else 0))
    k3.metric("Unique Votes", f"{dff['Vote'].nunique():,}" if "Vote" in dff.columns else "-")
    k4.metric("Unique Staff ID", f"{dff['Staff ID'].nunique():,}" if "Staff ID" in dff.columns else "-")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Approved Amount by Sponsor Category")
        if "Sponsor Category" in dff.columns and "Approved Amount (RM)" in dff.columns:
            tmp = dff.groupby("Sponsor Category", dropna=False)["Approved Amount (RM)"].sum().reset_index()
            tmp["Sponsor Category"] = tmp["Sponsor Category"].astype("string").fillna("Unknown")
            fig = px.bar(tmp, x="Sponsor Category", y="Approved Amount (RM)")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Missing required columns.")

    with right:
        st.subheader("Count by FOR (Category)")
        if "FOR (Category)" in dff.columns:
            tmp = dff["FOR (Category)"].astype("string").fillna("Unknown").value_counts().reset_index()
            tmp.columns = ["FOR (Category)", "Count"]
            fig = px.bar(tmp, x="FOR (Category)", y="Count")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column 'FOR (Category)' not found.")

    st.subheader("Data Table (AKTIF 2026)")
    st.dataframe(dff, use_container_width=True, hide_index=True)
    make_download_button(dff, "grants_filtered.csv")

    st.divider()
    st.subheader("PI 2026 (Researcher list)")
    if df_pi.empty:
        st.info("No data found in 'PI 2026'.")
    else:
        q2 = st.text_input("Search PI (Name/Staff ID)", value="").strip()
        dpi = df_pi.copy()
        if q2:
            dpi = global_search(dpi, q2, cols=[c for c in ["Staff ID", "Name"] if c in dpi.columns])
        st.dataframe(dpi, use_container_width=True, hide_index=True)
        make_download_button(dpi, "pi_2026_filtered.csv", "Download PI CSV")


# =========================
# PUBLICATIONS MODULE (INTERACTIVE PRESENTATION)
# =========================
elif section == "Publications • Interactive":
    st.title("Publications • Interactive Presentation")
    st.caption("Index (ARTICLE & CONFERENCE) + Citation analytics")

    tabs = st.tabs(["Overview", "Researcher Profiles", "Index (Article/Conference)", "Citations"])

    # ---- Overview ----
    with tabs[0]:
        total_articles = len(df_idx_art) if not df_idx_art.empty else 0
        total_confs = len(df_idx_conf) if not df_idx_conf.empty else 0
        total_docs_cit = len(df_cit) if not df_cit.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Indexed Articles", f"{total_articles:,}")
        c2.metric("Indexed Conferences", f"{total_confs:,}")
        c3.metric("Citation docs", f"{total_docs_cit:,}")
        c4.metric("Cumulative (2021–2025)", f"{CUMULATIVE_TOTAL_2021_2025:,}")

        st.subheader("Cumulative Citations (Hardcoded from Summary)")
        fig = px.line(df_sum_cum, x="Year", y="Cumulative Citations", markers=True)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Total (All Years): **{TOTAL_CITATIONS_ALL:,}**")

    # ---- Researcher Profiles ----
    with tabs[1]:
        st.subheader("Researcher Profiles")
        st.caption("Search by Name or Staff ID (Publications only).")

        names = set()
        if not df_idx_art.empty and "STAF" in df_idx_art.columns:
            names |= set(df_idx_art["STAF"].dropna().astype("string").tolist())
        if not df_idx_conf.empty and "STAF" in df_idx_conf.columns:
            names |= set(df_idx_conf["STAF"].dropna().astype("string").tolist())
        names = sorted([n for n in names if n and n.lower() != "nan"])

        ids = set()
        if not df_idx_art.empty and "STAF ID" in df_idx_art.columns:
            ids |= set(df_idx_art["STAF ID"].dropna().astype("string").tolist())
        if not df_idx_conf.empty and "STAF ID" in df_idx_conf.columns:
            ids |= set(df_idx_conf["STAF ID"].dropna().astype("string").tolist())
        ids = sorted([i for i in ids if i and i.lower() != "nan"])

        a, b = st.columns([1.2, 1])
        sel_name = a.selectbox("Select Name", [""] + names, index=0)
        sel_id = b.selectbox("Select Staff ID", [""] + ids, index=0)

        if not sel_name and not sel_id:
            st.info("Select a name and/or staff ID.")
        else:
            prof_art = df_idx_art.copy()
            prof_conf = df_idx_conf.copy()

            if not prof_art.empty:
                if sel_id:
                    prof_art = prof_art[prof_art["STAF ID"].astype("string") == str(sel_id)]
                if sel_name:
                    prof_art = prof_art[safe_str_contains(prof_art["STAF"], sel_name)]
            if not prof_conf.empty:
                if sel_id:
                    prof_conf = prof_conf[prof_conf["STAF ID"].astype("string") == str(sel_id)]
                if sel_name:
                    prof_conf = prof_conf[safe_str_contains(prof_conf["STAF"], sel_name)]

            k1, k2 = st.columns(2)
            k1.metric("Articles", f"{len(prof_art):,}")
            k2.metric("Conferences", f"{len(prof_conf):,}")

            st.write("**ARTICLE**")
            st.dataframe(prof_art, use_container_width=True, hide_index=True)
            make_download_button(prof_art, "researcher_articles.csv")

            st.write("**CONFERENCE**")
            st.dataframe(prof_conf, use_container_width=True, hide_index=True)
            make_download_button(prof_conf, "researcher_conferences.csv")

    # ---- Index (Article/Conference) ----
    with tabs[2]:
        st.subheader("Index Publications (ARTICLE vs CONFERENCE)")
        scope = st.radio("Scope", ["ARTICLE", "CONFERENCE", "BOTH"], horizontal=True)

        if scope == "ARTICLE":
            dfx = df_idx_art.copy()
        elif scope == "CONFERENCE":
            dfx = df_idx_conf.copy()
        else:
            dfx = pd.concat(
                [df_idx_art.assign(_SECTION="ARTICLE"), df_idx_conf.assign(_SECTION="CONFERENCE")],
                ignore_index=True
            )

        if dfx.empty:
            st.warning("No publication data in selected scope.")
        else:
            with st.expander("Filters", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                status_opts = sorted(dfx.get("STATUS", pd.Series(dtype="string")).dropna().astype("string").unique())
                role_opts = sorted(dfx.get("ROLE", pd.Series(dtype="string")).dropna().astype("string").unique())
                year_num = pd.to_numeric(dfx.get("YEAR"), errors="coerce")
                year_opts = sorted([int(x) for x in year_num.dropna().unique()])

                sel_status = c1.multiselect("STATUS", status_opts, default=status_opts)
                sel_role = c2.multiselect("ROLE", role_opts, default=role_opts)
                sel_year = c3.multiselect("YEAR", year_opts, default=year_opts)
                q = c4.text_input("Search (Title / Staff / Source)", value="").strip()

            if sel_status and "STATUS" in dfx.columns:
                dfx = dfx[dfx["STATUS"].astype("string").isin(sel_status)]
            if sel_role and "ROLE" in dfx.columns:
                dfx = dfx[dfx["ROLE"].astype("string").isin(sel_role)]
            if sel_year and "YEAR" in dfx.columns:
                dfx["_YEAR_NUM"] = pd.to_numeric(dfx["YEAR"], errors="coerce")
                dfx = dfx[dfx["_YEAR_NUM"].isin(sel_year)]
            if q:
                dfx = global_search(dfx, q, cols=[c for c in ["TITLE", "STAF", "SOURCE TITLE"] if c in dfx.columns])

            k1, k2, k3 = st.columns(3)
            k1.metric("Records", f"{len(dfx):,}")
            k2.metric("Unique Staff", f"{dfx['STAF ID'].nunique():,}" if "STAF ID" in dfx.columns else "-")
            k3.metric("Total CITED BY", f"{int(pd.to_numeric(dfx.get('CITED BY'), errors='coerce').sum()):,}" if "CITED BY" in dfx.columns else "-")

            left, right = st.columns(2)
            with left:
                st.write("**Counts by STATUS**")
                tmp = dfx["STATUS"].astype("string").fillna("Unknown").value_counts().reset_index()
                tmp.columns = ["STATUS", "Count"]
                st.plotly_chart(px.bar(tmp, x="STATUS", y="Count"), use_container_width=True)

            with right:
                st.write("**Top Source Titles**")
                tmp = dfx["SOURCE TITLE"].astype("string").fillna("Unknown").value_counts().head(15).reset_index()
                tmp.columns = ["Source Title", "Count"]
                st.plotly_chart(px.bar(tmp, x="Source Title", y="Count"), use_container_width=True)

            st.dataframe(dfx, use_container_width=True, hide_index=True)
            make_download_button(dfx, "publications_filtered.csv")

    # ---- Citations ----
    with tabs[3]:
        st.subheader("Citations")
        if df_cit.empty:
            st.warning("No citation sheet loaded (missing 'citation' sheet).")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cumulative (2021–2025)", f"{CUMULATIVE_TOTAL_2021_2025:,}")
            c2.metric("Total (All Years)", f"{TOTAL_CITATIONS_ALL:,}")
            c3.metric("Docs", f"{len(df_cit):,}")
            c4.metric("Max doc cites", f"{int(df_cit['Total'].max()):,}" if "Total" in df_cit.columns else "-")

            topn = st.slider("Top N documents", 5, 50, 15)
            top = df_cit.sort_values("Total", ascending=False).head(topn)

            st.dataframe(
                top[["Publication Year", "Document Title", "Journal Title", "Total"]],
                use_container_width=True,
                hide_index=True,
            )
            make_download_button(top, "top_cited_documents.csv", "Download Top Cited CSV")


# =========================
# DATA EXPLORER (BOTH FILES)
# =========================
elif section == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Browse any loaded table with safe global search + download.")

    tables: Dict[str, pd.DataFrame] = {
        "Grants - AKTIF 2026": df_aktif,
        "Grants - PI 2026": df_pi,
        "Publications - Index ARTICLE": df_idx_art,
        "Publications - Index CONFERENCE": df_idx_conf,
        "Publications - Citation": df_cit,
        "Summary - Cumulative Citations (Hardcoded)": df_sum_cum,
    }

    table_name = st.selectbox("Select table", list(tables.keys()))
    df = tables[table_name].copy()

    if df.empty:
        st.info("Selected table is empty.")
        st.stop()

    with st.expander("Filters", expanded=True):
        a, b, c = st.columns([1.4, 1, 1])
        q = a.text_input("Global search (contains)", value="").strip()
        max_rows = b.number_input("Max rows to display", min_value=100, max_value=200000, value=5000, step=100)
        drop_nan = c.checkbox("Drop NaN-heavy rows", value=False)

    if drop_nan:
        df = drop_nan_heavy_rows(df, threshold=0.85)
    if q:
        df = global_search(df, q)

    st.dataframe(df.head(int(max_rows)), use_container_width=True, hide_index=True)
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", table_name).lower()
    make_download_button(df, f"{safe_name}.csv")