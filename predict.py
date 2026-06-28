"""Predict international football match outcomes using TabPFN.

Uses three external data sources on top of the core match history:
  - eloratings.net — independent pre-match Elo ratings per country
  - Transfermarkt  — squad market value, age, and caps per year
Both are cached locally after the first download.

Missing external data is filled with neutral priors so the full training
history is always used (cutting it to match data availability hurts
ELO convergence more than it helps).
"""

import argparse
import gzip
import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss
from tabpfn_client import TabPFNClassifier

TODAY       = pd.Timestamp.now().normalize()
TRAIN_START = pd.Timestamp("2014-01-01")
MAX_TRAIN   = 10000
HOME_ADV    = 65.0
DATA        = "results.csv"
RAW_URL     = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

ELO_CACHE  = "elo_history.parquet"
TM_ZIP_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip"
TM_CACHE   = "tm_squad_values.parquet"

# Priors used when external data is unavailable for a match
EXT_ELO_PRIOR = 1500.0  # neutral starting ELO
TM_MV_PRIOR   = 0.0     # zero diff = no information
TM_AGE_PRIOR  = 26.5    # FIFA squad age benchmark
TM_CAPS_PRIOR = 40.0    # ~median caps for a top-23 player

FEATURES = [
    # base features
    "elo_diff", "home_elo", "away_elo",
    "form5_diff", "form10_diff", "home_form5", "away_form5",
    "home_winrate", "away_winrate",
    "home_gf5", "away_gf5", "home_ga5", "away_ga5", "gd10_diff",
    "home_streak", "away_streak", "home_rest", "away_rest",
    "home_played", "away_played",
    "h2h_n", "h2h_home_winrate", "h2h_draw_rate", "h2h_gd",
    "neutral", "importance",
    # external Elo (eloratings.net)
    "ext_elo_diff", "home_ext_elo", "away_ext_elo",
    # squad value, age, caps (Transfermarkt)
    "squad_mv_diff", "home_log_mv", "away_log_mv",
    "squad_age_diff", "home_squad_age", "away_squad_age",
    "squad_caps_diff", "home_squad_caps", "away_squad_caps",
    # peak historical ELO (pedigree)
    "peak_elo_diff", "home_peak_elo", "away_peak_elo",
]


def importance(t):
    t = t.lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if "confederations" in t:
        return 50.0
    if any(k in t for k in [
        "uefa euro", "copa am", "african cup", "asian cup",
        "gold cup", "nations league", "oceania nations",
    ]):
        return 45.0
    if "qualif" in t:
        return 35.0
    if "friendly" in t:
        return 20.0
    return 30.0


def load_data(refresh=False):
    if refresh or not os.path.exists(DATA):
        df = pd.read_csv(RAW_URL)
        df.to_csv(DATA, index=False)
    else:
        df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] < df["away_score"]],
        ["home_win", "away_win"], default="draw")
    df.loc[df["home_score"].isna(), "outcome"] = np.nan
    df["importance"] = df["tournament"].apply(importance)
    return df


def _fetch_elo_tsv(country: str) -> pd.DataFrame | None:
    """Fetch one country's match-by-match Elo history from eloratings.net.
    Shifts elo_after → elo_before so each row holds the pre-match rating.
    Returns None on network failure."""
    url = f"https://www.eloratings.net/{country.replace(' ', '_')}.tsv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8")
    except Exception:
        return None

    rows = []
    for line in raw.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        try:
            date           = pd.Timestamp(parts[0])
            home_elo_after = float(parts[5])  # col 5 = home elo after match
            away_elo_after = float(parts[6])  # col 6 = away elo after match
            home_team      = parts[1]
            away_team      = parts[2]
        except (ValueError, IndexError):
            continue
        rows.append((date, home_team, home_elo_after, away_team, away_elo_after))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", "home_team", "home_elo_after",
                                     "away_team", "away_elo_after"])
    df = df.sort_values("date").reset_index(drop=True)

    # shift within each team's timeline to get pre-match rating
    records = []
    for team_col, elo_col in [("home_team", "home_elo_after"), ("away_team", "away_elo_after")]:
        tmp = df[["date", team_col, elo_col]].rename(
            columns={team_col: "team", elo_col: "elo_after"})
        tmp = tmp.sort_values("date").reset_index(drop=True)
        tmp["elo_before"] = tmp.groupby("team")["elo_after"].shift(1)
        # first appearance: elo_after is off by one match but beats a flat 1500 prior
        tmp["elo_before"] = tmp["elo_before"].fillna(tmp["elo_after"])
        records.append(tmp[["date", "team", "elo_before"]])

    return pd.concat(records, ignore_index=True).drop_duplicates(["date", "team"])


def load_ext_elo(teams: list[str], refresh=False) -> pd.DataFrame:
    """Download and cache per-match Elo history for all teams.
    Returns a long-form (date, team, ext_elo) DataFrame. Teams that 404 are skipped."""
    if not refresh and os.path.exists(ELO_CACHE):
        return pd.read_parquet(ELO_CACHE)

    print("Fetching external Elo ratings from eloratings.net …")
    frames = []
    for i, team in enumerate(teams, 1):
        df = _fetch_elo_tsv(team)
        if df is not None:
            frames.append(df)
        if i % 20 == 0:
            print(f"  {i}/{len(teams)} teams fetched")

    if not frames:
        print("  WARNING: Could not reach eloratings.net; ext_elo features will use priors.")
        result = pd.DataFrame(columns=["date", "team", "ext_elo"])
    else:
        result = pd.concat(frames, ignore_index=True)
        result = result.rename(columns={"elo_before": "ext_elo"})

    result.to_parquet(ELO_CACHE, index=False)
    return result


def _download_tm_zip() -> bytes:
    req = urllib.request.Request(TM_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def load_squad_values(refresh=False) -> pd.DataFrame:
    """Download and cache Transfermarkt squad stats per (team, year):
    total market value, market-value-weighted age, and weighted caps of the top-23.
    Falls back to an empty DataFrame on failure; fill_ext_priors() handles the gaps."""
    if not refresh and os.path.exists(TM_CACHE):
        return pd.read_parquet(TM_CACHE)

    print("Downloading Transfermarkt datasets zip (~200 MB) …")
    try:
        raw = _download_tm_zip()
    except Exception as e:
        print(f"  WARNING: Could not download Transfermarkt data ({e}). "
              "squad_mv features will use priors.")
        return pd.DataFrame(columns=["team", "year", "squad_mv_eur_m", "squad_age_w", "squad_caps_w"])

    print("  Parsing players and valuations …")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()

        def pick(stem):
            # handles both .csv and .csv.gz depending on the zip version
            for n in names:
                if n.endswith(f"/{stem}.csv") or n == f"{stem}.csv":
                    return n
                if n.endswith(f"/{stem}.csv.gz") or n == f"{stem}.csv.gz":
                    return n
            return None

        def open_entry(path):
            fobj = zf.open(path)
            return gzip.open(fobj) if path.endswith(".gz") else fobj

        players_path    = pick("players")
        valuations_path = pick("player_valuations")

        if not players_path or not valuations_path:
            print(f"  WARNING: Expected CSVs not found in zip (got: {names[:10]}…)")
            return pd.DataFrame(columns=["team", "year", "squad_mv_eur_m", "squad_age_w", "squad_caps_w"])

        players = pd.read_csv(open_entry(players_path),
                              usecols=["player_id", "current_national_team_id",
                                       "date_of_birth", "international_caps"],
                              low_memory=False)
        valuations = pd.read_csv(open_entry(valuations_path),
                                 usecols=["player_id", "date", "market_value_in_eur"],
                                 low_memory=False)

    players = players.dropna(subset=["current_national_team_id"])
    players["dob"] = pd.to_datetime(players["date_of_birth"], errors="coerce")

    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations = valuations.dropna(subset=["date", "market_value_in_eur"])
    valuations["year"] = valuations["date"].dt.year
    valuations["mv_m"] = valuations["market_value_in_eur"] / 1e6

    # snapshot closest to July 1 as the pre-season reference value
    valuations["abs_dist"] = (
        valuations["date"] - valuations["date"].dt.year.map(
            lambda y: pd.Timestamp(f"{y}-07-01"))
    ).abs()
    val_annual = (valuations
                  .sort_values("abs_dist")
                  .groupby(["player_id", "year"], as_index=False)
                  .first()[["player_id", "year", "mv_m", "date"]])

    merged = val_annual.merge(
        players[["player_id", "current_national_team_id", "dob", "international_caps"]],
        on="player_id", how="inner")
    merged["age_at_snap"] = (merged["date"] - merged["dob"]).dt.days / 365.25

    def squad_stats(g):
        top = g.nlargest(23, "mv_m")
        mv_total = top["mv_m"].sum()
        caps = top["international_caps"].fillna(0)
        if mv_total > 0:
            age_w  = (top["mv_m"] * top["age_at_snap"]).sum() / mv_total
            caps_w = (top["mv_m"] * caps).sum() / mv_total
        else:
            age_w  = top["age_at_snap"].mean()
            caps_w = caps.mean()
        return pd.Series({"squad_mv_eur_m": mv_total, "squad_age_w": age_w, "squad_caps_w": caps_w})

    squad = (merged
             .groupby(["current_national_team_id", "year"])
             .apply(squad_stats, include_groups=False)
             .reset_index())

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        nt_path = None
        for n in zf.namelist():
            if "national_teams" in n and (n.endswith(".csv") or n.endswith(".csv.gz")):
                nt_path = n
                break
        if nt_path:
            nt_fobj = gzip.open(zf.open(nt_path)) if nt_path.endswith(".gz") else zf.open(nt_path)
            nt = pd.read_csv(nt_fobj,
                             usecols=lambda c: c in ["national_team_id", "name",
                                                      "national_team_name", "id"],
                             low_memory=False)
            # column names vary slightly between zip versions
            id_col   = next((c for c in nt.columns if "id" in c.lower()), None)
            name_col = next((c for c in nt.columns if "name" in c.lower()), None)
            if id_col and name_col:
                nt = nt.rename(columns={id_col: "tid", name_col: "team"})
                squad = squad.merge(nt[["tid", "team"]],
                                    left_on="current_national_team_id",
                                    right_on="tid", how="left")
            else:
                squad["team"] = squad["current_national_team_id"].astype(str)
        else:
            squad["team"] = squad["current_national_team_id"].astype(str)

    squad = squad[["team", "year", "squad_mv_eur_m", "squad_age_w", "squad_caps_w"]].dropna(subset=["team"])
    squad.to_parquet(TM_CACHE, index=False)
    print(f"  Transfermarkt squad values ready: {len(squad)} rows, "
          f"{squad['team'].nunique()} national teams, years {squad['year'].min()}–{squad['year'].max()}")
    return squad


def attach_ext_elo(feats: pd.DataFrame, elo_long: pd.DataFrame) -> pd.DataFrame:
    """Join the most-recent pre-match ext_elo onto each row using merge_asof.
    Teams with no eloratings.net entry get NaN; fill_ext_priors() handles those."""
    if elo_long.empty:
        feats["home_ext_elo"] = np.nan
        feats["away_ext_elo"] = np.nan
        feats["ext_elo_diff"] = np.nan
        return feats

    elo_sorted = elo_long.sort_values("date")

    def lookup(side: str) -> pd.Series:
        team_col = f"{side}_team"
        tmp = feats[["date", team_col]].copy().reset_index()
        tmp = tmp.sort_values("date")
        merged = pd.merge_asof(
            tmp,
            elo_sorted.rename(columns={"team": team_col, "ext_elo": f"{side}_ext_elo"}),
            on="date",
            by=team_col,
            direction="backward",
        )
        return merged.set_index("index")[f"{side}_ext_elo"]

    feats["home_ext_elo"] = lookup("home")
    feats["away_ext_elo"] = lookup("away")
    # diff is recomputed in fill_ext_priors after NaNs are resolved,
    # so one missing side doesn't poison the whole row
    feats["ext_elo_diff"] = np.nan
    return feats


def attach_squad_values(feats: pd.DataFrame, squad: pd.DataFrame) -> pd.DataFrame:
    """Join squad market value, age, and caps for each match by (team, year).
    Pre-2005 matches and unmatched team names get NaN, filled later by fill_ext_priors()."""
    if squad.empty:
        for col in ["home_log_mv", "away_log_mv", "squad_mv_diff",
                    "home_squad_age", "away_squad_age", "squad_age_diff",
                    "home_squad_caps", "away_squad_caps", "squad_caps_diff"]:
            feats[col] = np.nan
        return feats

    squad_idx = squad.set_index(["team", "year"])

    def lookup_mv(team_series: pd.Series, year_series: pd.Series):
        mv   = np.full(len(team_series), np.nan)
        age  = np.full(len(team_series), np.nan)
        caps = np.full(len(team_series), np.nan)
        for i, (team, year) in enumerate(zip(team_series, year_series)):
            key = (team, year)
            if key in squad_idx.index:
                row = squad_idx.loc[key]
                mv[i]   = row["squad_mv_eur_m"]
                age[i]  = row["squad_age_w"]
                caps[i] = row["squad_caps_w"]
        return mv, age, caps

    year = feats["date"].dt.year
    hmv, hage, hcaps = lookup_mv(feats["home_team"], year)
    amv, aage, acaps = lookup_mv(feats["away_team"], year)

    feats["home_log_mv"]    = np.log1p(hmv)
    feats["away_log_mv"]    = np.log1p(amv)
    feats["squad_mv_diff"]  = feats["home_log_mv"] - feats["away_log_mv"]
    feats["home_squad_age"] = hage
    feats["away_squad_age"] = aage
    feats["squad_age_diff"] = feats["home_squad_age"] - feats["away_squad_age"]
    feats["home_squad_caps"] = hcaps
    feats["away_squad_caps"] = acaps
    feats["squad_caps_diff"] = feats["home_squad_caps"] - feats["away_squad_caps"]
    return feats


def fill_ext_priors(feats: pd.DataFrame) -> pd.DataFrame:
    """Fill missing external features with neutral priors so the full history stays usable.
    Diffs → 0, absolutes → population median or a known benchmark. TabPFN never sees NaN."""
    mv_median = feats["home_log_mv"].median()
    if np.isnan(mv_median):
        mv_median = np.log1p(50)  # fallback: ~50M EUR

    feats["home_ext_elo"]    = feats["home_ext_elo"].fillna(EXT_ELO_PRIOR)
    feats["away_ext_elo"]    = feats["away_ext_elo"].fillna(EXT_ELO_PRIOR)
    feats["home_log_mv"]     = feats["home_log_mv"].fillna(mv_median)
    feats["away_log_mv"]     = feats["away_log_mv"].fillna(mv_median)
    feats["home_squad_age"]  = feats["home_squad_age"].fillna(TM_AGE_PRIOR)
    feats["away_squad_age"]  = feats["away_squad_age"].fillna(TM_AGE_PRIOR)
    feats["home_squad_caps"] = feats["home_squad_caps"].fillna(TM_CAPS_PRIOR)
    feats["away_squad_caps"] = feats["away_squad_caps"].fillna(TM_CAPS_PRIOR)

    # recompute diffs from filled sides so a single missing side doesn't break the pair
    adj = feats["neutral"].map({0: HOME_ADV, 1: 0.0}).fillna(0.0)
    feats["ext_elo_diff"]    = feats["home_ext_elo"] + adj - feats["away_ext_elo"]
    feats["squad_mv_diff"]   = feats["home_log_mv"]  - feats["away_log_mv"]
    feats["squad_age_diff"]  = feats["home_squad_age"] - feats["away_squad_age"]
    feats["squad_caps_diff"] = feats["home_squad_caps"] - feats["away_squad_caps"]
    return feats


def build_features(df):
    """One chronological pass: every feature uses only matches before kickoff."""
    elo      = defaultdict(lambda: 1500.0)
    peak_elo = defaultdict(lambda: 1500.0)
    res = defaultdict(list)
    last_date, h2h = {}, defaultdict(list)

    def team_feats(team):
        r = res[team]
        if not r:
            return elo[team], 1.3, 1.3, 0.33, 1.0, 1.0, 0.0, 0.0, 0
        last5, last10 = r[-5:], r[-10:]
        streak = 0
        for p, *_ in reversed(r):
            if p < 1:
                break
            streak += 1
        return (elo[team],
                np.mean([p for p, *_ in last5]), np.mean([p for p, *_ in last10]),
                np.mean([w for *_, w in last10]),
                np.mean([g for _, g, _, _ in last5]), np.mean([a for _, _, a, _ in last5]),
                np.mean([g - a for _, g, a, _ in last10]), streak, len(r))

    def h2h_feats(home, away):
        m = h2h[tuple(sorted((home, away)))]
        if not m:
            return 0, 0.5, 0.25, 0.0
        n = len(m)
        return (n,
                sum(w == home for _, _, w in m) / n,
                sum(w == "draw" for _, _, w in m) / n,
                np.mean([g if h == home else -g for h, g, _ in m]))

    rows = []
    for r in df.itertuples():
        h, a, adj = r.home_team, r.away_team, HOME_ADV * (1 - r.neutral)
        he, hf5, hf10, hwr, hgf, hga, hgd, hstk, hn = team_feats(h)
        ae, af5, af10, awr, agf, aga, agd, astk, an = team_feats(a)
        nm, h2h_wr, h2h_dr, h2h_gd = h2h_feats(h, a)
        rows.append({
            "elo_diff": he + adj - ae, "home_elo": he, "away_elo": ae,
            "form5_diff": hf5 - af5, "form10_diff": hf10 - af10,
            "home_form5": hf5, "away_form5": af5,
            "home_winrate": hwr, "away_winrate": awr,
            "home_gf5": hgf, "away_gf5": agf, "home_ga5": hga, "away_ga5": aga,
            "gd10_diff": hgd - agd, "home_streak": hstk, "away_streak": astk,
            "home_rest": min((r.date - last_date[h]).days, 90) if h in last_date else 30,
            "away_rest": min((r.date - last_date[a]).days, 90) if a in last_date else 30,
            "home_played": hn, "away_played": an,
            "h2h_n": nm, "h2h_home_winrate": h2h_wr, "h2h_draw_rate": h2h_dr, "h2h_gd": h2h_gd,
            "home_peak_elo": peak_elo[h], "away_peak_elo": peak_elo[a],
            "peak_elo_diff": peak_elo[h] - peak_elo[a],
        })

        if not np.isnan(r.home_score):
            gd = r.home_score - r.away_score
            exp = 1 / (1 + 10 ** ((ae - he - adj) / 400))
            s = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
            # goal-difference multiplier (FIFA-style): bigger wins shift ratings more
            g = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8)
            delta = r.importance * g * (s - exp)
            elo[h] += delta
            elo[a] -= delta
            peak_elo[h] = max(peak_elo[h], elo[h])
            peak_elo[a] = max(peak_elo[a], elo[a])
            res[h].append((3 if gd > 0 else (1 if gd == 0 else 0), r.home_score, r.away_score, gd > 0))
            res[a].append((3 if gd < 0 else (1 if gd == 0 else 0), r.away_score, r.home_score, gd < 0))
            last_date[h] = last_date[a] = r.date
            h2h[tuple(sorted((h, a)))].append((h, gd, h if gd > 0 else (a if gd < 0 else "draw")))

    return df.join(pd.DataFrame(rows, index=df.index))


def train(pool):
    clf = TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
    clf.fit(pool[FEATURES].values, pool["outcome"].values)
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh",    action="store_true",
                        help="Re-download all datasets from source")
    parser.add_argument("--no-ext-elo", action="store_true",
                        help="Skip eloratings.net (use priors for ext_elo features)")
    parser.add_argument("--no-tm",      action="store_true",
                        help="Skip Transfermarkt download (use priors for squad_mv features)")
    args = parser.parse_args()

    df = load_data(refresh=args.refresh)
    latest_date = df[df["date"].notna()]["date"].max()
    print(f"Latest game in dataset: {latest_date.date()}")
    print(f"Data freshness:         {pd.Timestamp.now() - latest_date}")

    # 2026 World Cup Round of 32 — update dates/teams as schedule is confirmed
    R32 = [
        ("2026-06-28", "South Africa",  "Canada"),
        ("2026-06-29", "Brazil",        "Japan"),
        ("2026-06-29", "Germany",       "Paraguay"),
        ("2026-06-29", "Netherlands",   "Morocco"),
        ("2026-06-30", "Ivory Coast",   "Norway"),
        ("2026-06-30", "France",        "Sweden"),
        ("2026-06-30", "Mexico",        "Ecuador"),
        ("2026-07-01", "England",       "DR Congo"),
        ("2026-07-01", "Belgium",       "Senegal"),
        ("2026-07-01", "United States", "Bosnia and Herzegovina"),
        ("2026-07-02", "Spain",         "Austria"),
        ("2026-07-02", "Portugal",      "Croatia"),
        ("2026-07-02", "Switzerland",   "Algeria"),
        ("2026-07-03", "Australia",     "Egypt"),
        ("2026-07-03", "Argentina",     "Cape Verde"),
        ("2026-07-03", "Colombia",      "Ghana"),
    ]
    r32_df = pd.DataFrame([{
        "date":       pd.Timestamp(d),
        "home_team":  h,
        "away_team":  a,
        "home_score": np.nan,
        "away_score": np.nan,
        "tournament": "FIFA World Cup",
        "neutral":    1,
        "importance": 60.0,
        "outcome":    np.nan,
    } for d, h, a in R32])
    df = pd.concat([df, r32_df], ignore_index=True).sort_values("date").reset_index(drop=True)

    feats = build_features(df)

    if not args.no_ext_elo:
        teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        elo_long = load_ext_elo(teams, refresh=args.refresh)
        feats = attach_ext_elo(feats, elo_long)
    else:
        print("Skipping eloratings.net (--no-ext-elo)")
        for c in ["home_ext_elo", "away_ext_elo", "ext_elo_diff"]:
            feats[c] = np.nan

    if not args.no_tm:
        squad = load_squad_values(refresh=args.refresh)
        feats = attach_squad_values(feats, squad)
    else:
        print("Skipping Transfermarkt (--no-tm)")
        for c in ["home_log_mv", "away_log_mv", "squad_mv_diff",
                  "home_squad_age", "away_squad_age", "squad_age_diff",
                  "home_squad_caps", "away_squad_caps", "squad_caps_diff"]:
            feats[c] = np.nan

    feats = fill_ext_priors(feats)

    played_mask = feats["outcome"].notna() & (feats["date"] >= TRAIN_START)
    for col in ["home_ext_elo", "home_log_mv", "home_squad_age", "home_squad_caps"]:
        pct = feats.loc[played_mask, col].notna().mean()
        print(f"  {col}: {pct:.0%} rows have real data (rest use priors)")

    played = feats[played_mask]
    future = feats[feats["home_score"].isna() & (feats["date"] >= TODAY)].sort_values("date")

    month = (TODAY.to_period("M") - 1)
    test = played[(played["date"] >= month.start_time) & (played["date"] < (month + 1).start_time)]
    if len(test):
        clf = train(played[played["date"] < month.start_time].tail(MAX_TRAIN))
        proba = clf.predict_proba(test[FEATURES].values)
        pred  = clf.classes_[proba.argmax(1)]
        print(f"\nBacktest {month} ({len(test)} matches): "
              f"accuracy {accuracy_score(test['outcome'], pred):.0%}, "
              f"log-loss {log_loss(test['outcome'], proba, labels=clf.classes_):.3f}")

    clf   = train(played.tail(MAX_TRAIN))
    proba = clf.predict_proba(future[FEATURES].values)
    cols  = {c: proba[:, i] for i, c in enumerate(clf.classes_)}

    out = future[["date", "home_team", "away_team"]].copy()
    out["predicted"]   = clf.classes_[proba.argmax(1)]
    out["p_home_win"]  = cols["home_win"]
    out["p_draw"]      = cols["draw"]
    out["p_away_win"]  = cols["away_win"]

    today_str = pd.Timestamp.now().strftime("%Y%m%d")
    filename  = f"predictions_{today_str}.csv"
    out.to_csv(filename, index=False)

    print(f"\n{len(out)} fixture predictions -> {filename}\n")
    for r in out.itertuples():
        print(f"  {r.date.date()}  {r.home_team:>20} vs {r.away_team:<20}  "
              f"-> {r.predicted:<9}  H {r.p_home_win:4.0%} | D {r.p_draw:4.0%} | A {r.p_away_win:4.0%}")


if __name__ == "__main__":
    main()
