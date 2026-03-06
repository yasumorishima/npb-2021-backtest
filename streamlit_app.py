"""
NPB 2026 予測（Marcel法 + ベイズ推定）

Marcel法の予測結果に外国人選手のベイズ推定を追加した順位表を表示。
Marcel版（npb-prediction）との差分が一目でわかる。

Data sources:
- プロ野球データFreak (https://baseball-data.com)
- 日本野球機構 NPB (https://npb.jp)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from config import DATA_END_YEAR, TARGET_YEAR
from translations import TEAM_NAME_EN, TEXTS


def t(key: str) -> str:
    """Return translated string for the current language."""
    lang = st.session_state.get("lang", "日本語")
    dict_key = "en" if lang == "English" else "ja"
    return TEXTS.get(dict_key, TEXTS["ja"]).get(key, key)


def team_disp(team_ja: str) -> str:
    """Return English team name when in English mode."""
    if st.session_state.get("lang", "日本語") == "English":
        return TEAM_NAME_EN.get(team_ja, team_ja)
    return team_ja


# --- Team definitions ---
NPB_TEAM_COLORS = {
    "DeNA": "#0055A5", "巨人": "#F97709", "阪神": "#FFE201",
    "広島": "#EE1C25", "中日": "#00468B", "ヤクルト": "#006AB6",
    "ソフトバンク": "#F5C70E", "日本ハム": "#004B97", "楽天": "#860029",
    "ロッテ": "#000000", "オリックス": "#C4A400", "西武": "#102A6F",
}
NPB_TEAM_GLOW = {
    "DeNA": "#00aaff", "巨人": "#ff9933", "阪神": "#ffe44d",
    "広島": "#ff4444", "中日": "#4488ff", "ヤクルト": "#44aaff",
    "ソフトバンク": "#ffdd33", "日本ハム": "#4488ff", "楽天": "#cc3366",
    "ロッテ": "#888888", "オリックス": "#ddcc33", "西武": "#4466cc",
}
TEAMS = list(NPB_TEAM_COLORS.keys())
CENTRAL_TEAMS = ["DeNA", "巨人", "阪神", "広島", "中日", "ヤクルト"]
PACIFIC_TEAMS = ["ソフトバンク", "日本ハム", "楽天", "ロッテ", "オリックス", "西武"]

BASE_URL = "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/"
BAYES_PROJ_URL = "https://raw.githubusercontent.com/yasumorishima/npb-bayes-projection/main/"

_VARIANT_MAP = str.maketrans("﨑髙濵澤邊齋齊國島嶋櫻", "崎高浜沢辺斎斉国島島桜")


def _norm(name: str) -> str:
    return name.replace("\u3000", " ").strip()


def _fuzzy(s: str) -> str:
    return s.replace(" ", "").replace("\u3000", "").translate(_VARIANT_MAP)


def _is_foreign_player(name: str) -> bool:
    cleaned = name.replace("\u3000", "").replace(" ", "")
    if not cleaned:
        return False
    katakana = sum(1 for c in cleaned if "\u30A0" <= c <= "\u30FF")
    return katakana / len(cleaned) > 0.5


def _pythagorean_wpct(rs: float, ra: float, k: float = 1.72) -> float:
    if ra == 0:
        return 1.0
    return rs**k / (rs**k + ra**k)


# --- Data loading ---
@st.cache_data(ttl=3600)
def load_csv(path: str) -> pd.DataFrame:
    url = BASE_URL + path
    try:
        df = pd.read_csv(url, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    if "player" in df.columns:
        df["player"] = df["player"].apply(_norm)
    if "team" in df.columns:
        df["team"] = df["team"].apply(_norm)
    return df


@st.cache_data(ttl=3600)
def _load_stan_csv(path: str) -> pd.DataFrame | None:
    """Load Stan CSV from npb-bayes-projection. Returns None on failure."""
    url = BAYES_PROJ_URL + path
    try:
        df = pd.read_csv(url, encoding="utf-8-sig")
    except Exception:
        return None
    if "player" in df.columns:
        df["player"] = df["player"].apply(_norm)
    return df


def _merge_stan_corrections(data: dict) -> None:
    """Merge Stan (Ridge) delta corrections for Japanese players."""
    stan_h = _load_stan_csv(f"data/projections/stan_hitters_{TARGET_YEAR}.csv")
    stan_p = _load_stan_csv(f"data/projections/stan_pitchers_{TARGET_YEAR}.csv")

    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]

    if stan_h is not None and not mh.empty:
        cols = ["player", "stan_wOBA", "stan_delta_wOBA"]
        cols = [c for c in cols if c in stan_h.columns]
        if len(cols) >= 2:
            stan_cols = stan_h[cols].drop_duplicates("player")
            data["marcel_hitters"] = mh.merge(stan_cols, on="player", how="left")

    if stan_p is not None and not mp.empty:
        cols = ["player", "stan_ERA", "stan_delta_ERA"]
        cols = [c for c in cols if c in stan_p.columns]
        if len(cols) >= 2:
            stan_cols = stan_p[cols].drop_duplicates("player")
            data["marcel_pitchers"] = mp.merge(stan_cols, on="player", how="left")


def load_all() -> dict:
    from roster_current import get_all_roster_names, get_team_for_player

    result = {
        "marcel_hitters": load_csv(f"data/projections/marcel_hitters_{TARGET_YEAR}.csv"),
        "marcel_pitchers": load_csv(f"data/projections/marcel_pitchers_{TARGET_YEAR}.csv"),
        "sabermetrics": load_csv(f"data/projections/npb_sabermetrics_2015_{DATA_END_YEAR}.csv"),
    }
    roster_names = get_all_roster_names()
    for key in ("marcel_hitters", "marcel_pitchers"):
        df = result[key]
        if df.empty or "player" not in df.columns:
            continue
        df = df[df["player"].apply(_fuzzy).isin(roster_names)].copy()
        for idx, row in df.iterrows():
            new_team = get_team_for_player(row["player"])
            if new_team:
                df.at[idx, "team"] = new_team
        result[key] = df

    _enrich_woba(result)
    _merge_stan_corrections(result)
    return result


def _enrich_woba(data: dict) -> None:
    """打者にwOBA/wRAA を追加（順位表計算用）"""
    mh = data["marcel_hitters"]
    saber = data.get("sabermetrics", pd.DataFrame())
    if mh.empty or saber.empty:
        return
    df_fit = saber[saber["PA"] >= 100].dropna(subset=["wOBA", "OBP", "SLG"])
    if len(df_fit) < 10:
        return
    X = np.column_stack([df_fit["OBP"].values, df_fit["SLG"].values, np.ones(len(df_fit))])
    coeffs, _, _, _ = np.linalg.lstsq(X, df_fit["wOBA"].values, rcond=None)
    a_obp, b_slg, intercept_w = coeffs
    recent_s = saber[saber["year"] >= 2022]
    lg_woba = recent_s[recent_s["PA"] >= 50]["wOBA"].mean()
    woba_scale = 1.15
    mh["wOBA_est"] = (a_obp * mh["OBP"] + b_slg * mh["SLG"] + intercept_w).round(3)
    mh["wRAA_est"] = ((mh["wOBA_est"] - lg_woba) / woba_scale * mh["PA"]).round(1)
    data["_lg_woba"] = lg_woba


def _get_league_averages(data: dict) -> tuple[float, float]:
    """Return (lg_woba, lg_era)."""
    saber = data.get("sabermetrics", pd.DataFrame())
    mp = data["marcel_pitchers"]
    if not saber.empty and "wOBA" in saber.columns:
        recent_s = saber[saber["year"] >= 2022]
        lg_woba = recent_s[recent_s["PA"] >= 50]["wOBA"].mean()
    else:
        lg_woba = 0.310
    lg_era = (
        (mp["ERA"] * mp["IP"]).sum() / mp["IP"].sum()
        if not mp.empty and mp["IP"].sum() > 0 else 3.50
    )
    return lg_woba, lg_era


def _get_missing_players(data: dict) -> dict:
    """ロースター登録済みだがMarcel予測対象外の選手をチーム別に返す。"""
    from roster_current import ROSTER_CURRENT
    from foreign_bayes import get_foreign_predictions, predict_no_prev_stats

    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]
    if mh.empty or mp.empty:
        return {}
    calculated = set(mh["player"].apply(_fuzzy)) | set(mp["player"].apply(_fuzzy))

    lg_woba, lg_era = _get_league_averages(data)
    bayes_preds = get_foreign_predictions(lg_woba, lg_era)

    result = {}
    for team, players in ROSTER_CURRENT.items():
        missing = []
        for p in players:
            if _fuzzy(p) not in calculated:
                kind = "foreign" if _is_foreign_player(p) else "rookie"
                display = p.replace("\u3000", " ").strip()
                bayes = None
                if kind == "foreign":
                    for bname, bpred in bayes_preds.items():
                        if _fuzzy(bname) == _fuzzy(p):
                            bayes = bpred
                            break
                    if bayes is None:
                        pred = predict_no_prev_stats("hitter", lg_woba, lg_era)
                        bayes = {
                            "pred": pred, "wraa_est": 0.0, "unc_wins": 1.5,
                            "type": "unknown", "has_prev": False,
                            "has_historical": True,
                            "stat_label": "", "stat_value": 0, "stat_range": (0, 0),
                        }
                missing.append({"name": display, "kind": kind, "bayes": bayes})
        result[team] = missing
    return result


@st.cache_data(ttl=3600)
def _load_park_factors() -> dict[str, float]:
    """Load PF_5yr (latest year) per team from npb-prediction."""
    url = BASE_URL + "data/projections/npb_park_factors.csv"
    try:
        pf_df = pd.read_csv(url, encoding="utf-8-sig")
    except Exception:
        return {}
    latest_year = pf_df["year"].max()
    latest = pf_df[pf_df["year"] == latest_year][["team", "PF_5yr"]].copy()
    return dict(zip(latest["team"], latest["PF_5yr"]))


def _apply_park_factor(rs: float, ra: float, team: str,
                       pf_map: dict[str, float]) -> tuple[float, float]:
    """Remove park effect: divide RS/RA by (PF + 1) / 2."""
    pf = pf_map.get(team)
    if pf and pf > 0:
        pf_factor = (pf + 1.0) / 2.0
        return rs / pf_factor, ra / pf_factor
    return rs, ra


def _build_standings_marcel_only(data: dict) -> pd.DataFrame:
    """Marcel法のみの順位表（外国人選手 = wRAA=0）"""
    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]
    if mh.empty or mp.empty:
        return pd.DataFrame()

    lg_avg_rs = 550.0
    lg_avg_ra = 550.0
    pf_map = _load_park_factors()

    if "wRAA_est" not in mh.columns:
        return pd.DataFrame()

    rows = []
    for team in TEAMS:
        h = mh[mh["team"] == team]
        p = mp[mp["team"] == team]
        rs_raw = lg_avg_rs + (h["wRAA_est"].sum() if not h.empty else 0)
        ra_raw = lg_avg_ra + ((p["ERA"] - (mp["ERA"] * mp["IP"]).sum() / mp["IP"].sum()) * p["IP"] / 9.0).sum() if not p.empty else lg_avg_ra
        rs_raw, ra_raw = _apply_park_factor(rs_raw, ra_raw, team, pf_map)
        league = "CL" if team in CENTRAL_TEAMS else "PL"
        rows.append({"league": league, "team": team, "rs_raw": rs_raw, "ra_raw": ra_raw})

    df = pd.DataFrame(rows)
    rs_scale = lg_avg_rs / df["rs_raw"].mean()
    ra_scale = lg_avg_ra / df["ra_raw"].mean()
    df["pred_RS"] = df["rs_raw"] * rs_scale
    df["pred_RA"] = df["ra_raw"] * ra_scale
    df["pred_WPCT"] = df.apply(lambda r: _pythagorean_wpct(r["pred_RS"], r["pred_RA"]), axis=1)
    df["pred_W"] = df["pred_WPCT"] * 143
    df["pred_L"] = 143 - df["pred_W"]
    return df[["league", "team", "pred_RS", "pred_RA", "pred_WPCT", "pred_W", "pred_L"]]


def _build_standings_bayes(data: dict, missing_all: dict) -> pd.DataFrame:
    """Marcel法 + Stan補正 + ベイズ推定の順位表"""
    from foreign_bayes import simulate_team_wins_mc

    mh = data["marcel_hitters"]
    mp = data["marcel_pitchers"]
    if mh.empty or mp.empty:
        return pd.DataFrame()

    lg_avg_rs = 550.0
    lg_avg_ra = 550.0
    lg_woba, lg_era = _get_league_averages(data)
    woba_scale = 1.15
    pf_map = _load_park_factors()

    if "wRAA_est" not in mh.columns:
        return pd.DataFrame()

    # Apply Stan corrections to wRAA for Japanese hitters
    has_stan_h = "stan_delta_wOBA" in mh.columns
    has_stan_p = "stan_delta_ERA" in mp.columns

    rows = []
    for team in TEAMS:
        h = mh[mh["team"] == team]
        p = mp[mp["team"] == team]

        # Hitters: add Stan delta to wRAA (delta_wOBA * PA / woba_scale)
        if has_stan_h and not h.empty:
            stan_wraa = h["wRAA_est"].copy()
            mask = h["stan_delta_wOBA"].notna()
            stan_wraa.loc[mask] = stan_wraa.loc[mask] + (
                h.loc[mask, "stan_delta_wOBA"] / woba_scale * h.loc[mask, "PA"]
            )
            rs_raw = lg_avg_rs + stan_wraa.sum()
        else:
            rs_raw = lg_avg_rs + (h["wRAA_est"].sum() if not h.empty else 0)

        # Pitchers: use stan_ERA where available
        if has_stan_p and not p.empty:
            era_col = p["ERA"].copy()
            mask = p["stan_delta_ERA"].notna()
            era_col.loc[mask] = era_col.loc[mask] + p.loc[mask, "stan_delta_ERA"]
            ra_raw = lg_avg_ra + (((era_col - lg_era) * p["IP"] / 9.0).sum())
        else:
            ra_raw = lg_avg_ra + (((p["ERA"] - lg_era) * p["IP"] / 9.0).sum() if not p.empty else 0)

        # Bayes foreign player contributions
        team_missing = missing_all.get(team, [])
        for m in team_missing:
            b = m.get("bayes")
            if b and (b.get("has_prev") or b.get("has_historical")):
                if b["type"] == "hitter":
                    rs_raw += b.get("wraa_est", 0)
                elif b["type"] == "pitcher":
                    ra_raw += b.get("ra_above_avg", 0)

        # Park factor correction
        rs_raw, ra_raw = _apply_park_factor(rs_raw, ra_raw, team, pf_map)

        league = "CL" if team in CENTRAL_TEAMS else "PL"
        rows.append({
            "league": league, "team": team, "rs_raw": rs_raw, "ra_raw": ra_raw,
            "missing_count": len(team_missing),
        })

    df = pd.DataFrame(rows)
    rs_scale = lg_avg_rs / df["rs_raw"].mean()
    ra_scale = lg_avg_ra / df["ra_raw"].mean()
    df["pred_RS"] = df["rs_raw"] * rs_scale
    df["pred_RA"] = df["ra_raw"] * ra_scale
    df["pred_WPCT"] = df.apply(lambda r: _pythagorean_wpct(r["pred_RS"], r["pred_RA"]), axis=1)
    df["pred_W"] = df["pred_WPCT"] * 143
    df["pred_L"] = 143 - df["pred_W"]

    # Monte Carlo prediction ranges
    for i, row in df.iterrows():
        team_missing = missing_all.get(row["team"], [])
        if team_missing:
            mc = simulate_team_wins_mc(
                pred_rs=row["pred_RS"], pred_ra=row["pred_RA"],
                missing_players=team_missing,
                rs_scale=rs_scale, ra_scale=ra_scale,
                lg_woba=lg_woba, lg_era=lg_era,
            )
            df.at[i, "pred_W_low"] = mc["pred_W_low"]
            df.at[i, "pred_W_high"] = mc["pred_W_high"]
        else:
            df.at[i, "pred_W_low"] = row["pred_W"]
            df.at[i, "pred_W_high"] = row["pred_W"]

    return df[["league", "team", "pred_RS", "pred_RA", "pred_WPCT",
               "pred_W", "pred_L", "missing_count", "pred_W_low", "pred_W_high"]]


# ==========================================================================
# Pages
# ==========================================================================

def page_standings(data: dict):
    st.markdown(f"### {t('standings_title')}")
    st.info(t("standings_info"))
    st.caption(t("standings_caption"))

    missing_all = _get_missing_players(data)
    marcel_only = _build_standings_marcel_only(data)
    bayes = _build_standings_bayes(data, missing_all)

    if marcel_only.empty or bayes.empty:
        st.error(t("no_data"))
        return

    for league, label in [("CL", t("central_league")), ("PL", t("pacific_league"))]:
        st.markdown(f"## {label}")
        col_marcel, col_bayes = st.columns(2)

        lg_m = marcel_only[marcel_only["league"] == league].sort_values("pred_WPCT", ascending=False).reset_index(drop=True)
        lg_b = bayes[bayes["league"] == league].sort_values("pred_WPCT", ascending=False).reset_index(drop=True)

        with col_marcel:
            st.markdown(f"**{t('marcel_only')}**")
            _render_standings_cards(lg_m, show_range=False)
            _render_standings_chart(lg_m, show_range=False)

        with col_bayes:
            st.markdown(f"**{t('marcel_bayes')}**")
            _render_standings_cards(lg_b, show_range=True, marcel_df=lg_m)
            _render_standings_chart(lg_b, show_range=True)

    st.info(t("pred_range_brief"))
    with st.expander(t("method_expander")):
        st.markdown(t("method_content"))


def _render_standings_cards(lg: pd.DataFrame, show_range: bool = False, marcel_df: pd.DataFrame | None = None):
    # Build Marcel wins lookup for delta display
    marcel_wins = {}
    if marcel_df is not None and not marcel_df.empty:
        marcel_wins = dict(zip(marcel_df["team"], marcel_df["pred_W"]))

    cards = ""
    for i, (_, row) in enumerate(lg.iterrows()):
        glow = NPB_TEAM_GLOW.get(row["team"], "#00e5ff")
        rank = i + 1
        medal = {1: "👑", 2: "🥈", 3: "🥉"}.get(rank, "")
        mc = int(row.get("missing_count", 0)) if "missing_count" in row.index else 0

        if show_range and mc > 0:
            w_lo = int(row.get("pred_W_low", row["pred_W"]))
            w_hi = int(row.get("pred_W_high", row["pred_W"]))
            w_cell = (
                f'<div style="display:flex;flex-direction:column;align-items:flex-start;">'
                f'<span style="color:#00e5ff;font-size:16px;font-weight:bold;">{row["pred_W"]:.0f}{t("wins_suffix")}</span>'
                f'<span style="color:#ff9944;font-size:9px;">{t("pred_range").format(lo=w_lo, hi=w_hi)}</span>'
                f'</div>'
            )
        else:
            w_cell = f'<span style="color:#00e5ff;font-size:16px;font-weight:bold;">{row["pred_W"]:.0f}{t("wins_suffix")}</span>'

        # Delta from Marcel (Bayes effect)
        delta_html = ""
        if row["team"] in marcel_wins:
            delta = row["pred_W"] - marcel_wins[row["team"]]
            if abs(delta) >= 0.1:
                if delta > 0:
                    delta_label = t("bayes_delta_pos").format(delta=f"{delta:+.1f}")
                    delta_color = "#44dd88"
                else:
                    delta_label = t("bayes_delta_neg").format(delta=f"{delta:.1f}")
                    delta_color = "#ff6666"
                delta_html = f'<span style="color:{delta_color};font-size:10px;font-weight:bold;">{delta_label}</span>'

        cards += f"""
        <div style="display:flex;align-items:center;gap:5px;padding:8px 10px;
                    background:#0d0d24;border-left:3px solid {glow};border-radius:5px;
                    font-family:'Segoe UI',sans-serif;margin:3px 0;">
          <span style="min-width:20px;font-size:14px;text-align:center;">{medal or rank}</span>
          <span style="min-width:60px;color:{glow};font-weight:bold;font-size:13px;">{team_disp(row['team'])}</span>
          {w_cell}
          <span style="color:#888;font-size:11px;">{row['pred_L']:.0f}{t("losses_suffix")}</span>
          <span style="color:#aaa;font-size:10px;">.{row['pred_WPCT']:.3f}</span>
          {delta_html}
        </div>"""

    card_h = 60 if show_range else 50
    components.html(f"<div>{cards}</div>", height=len(lg) * card_h + 10)


def _render_standings_chart(lg: pd.DataFrame, show_range: bool = False):
    fig = go.Figure()
    teams_reversed = [team_disp(t) for t in lg["team"].tolist()[::-1]]
    wins_reversed = lg["pred_W"].tolist()[::-1]
    colors_reversed = [NPB_TEAM_COLORS.get(t, "#333") for t in lg["team"].tolist()[::-1]]

    err_plus = err_minus = None
    if show_range and "pred_W_high" in lg.columns:
        err_plus = (lg["pred_W_high"] - lg["pred_W"]).tolist()[::-1]
        err_minus = (lg["pred_W"] - lg["pred_W_low"]).tolist()[::-1]

    fig.add_trace(go.Bar(
        name=t("pred_wins_label"), y=teams_reversed, x=wins_reversed,
        orientation="h", marker_color=colors_reversed,
        error_x=dict(
            type="data", array=err_plus, arrayminus=err_minus,
            visible=True, color="#ff9944", thickness=2, width=5,
        ) if err_plus else None,
    ))
    max_w = max(lg["pred_W_high"] if "pred_W_high" in lg.columns else lg["pred_W"]) if show_range else max(lg["pred_W"])
    fig.update_layout(
        height=250, xaxis_title=t("pred_wins_label"),
        xaxis_range=[0, max_w * 1.1],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), margin=dict(l=80, t=10, b=30),
        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


def page_foreign_players(data: dict):
    st.markdown(f"### {t('foreign_title')}")
    st.caption(t("foreign_caption"))

    from foreign_bayes import get_foreign_predictions, load_foreign_2026

    lg_woba, lg_era = _get_league_averages(data)
    preds = get_foreign_predictions(lg_woba, lg_era)
    raw_data = load_foreign_2026()

    # Build source lookup
    source_map = {row["npb_name"]: row.get("source", "") for row in raw_data}

    # Get team mapping
    from roster_current import ROSTER_CURRENT
    player_team_map = {}
    for team, players in ROSTER_CURRENT.items():
        for p in players:
            player_team_map[_fuzzy(p)] = team

    rows = []
    for name, pred in preds.items():
        team = player_team_map.get(_fuzzy(name), "?")
        ptype = t("hitter") if pred["type"] == "hitter" else t("pitcher")

        if pred["has_prev"]:
            if pred["type"] == "hitter":
                prev_stat = f"wOBA {pred['prev_stat']:.3f}"
                pred_stat = f"wOBA {pred['stat_value']:.3f}"
                ci = f"{pred['stat_range'][0]:.3f}–{pred['stat_range'][1]:.3f}"
                contrib = t("wraa_contribution").format(val=pred.get("wraa_est", 0))
            else:
                prev_stat = f"ERA {pred['prev_stat']:.2f}"
                pred_stat = f"ERA {pred['stat_value']:.2f}"
                ci = f"{pred['stat_range'][0]:.2f}–{pred['stat_range'][1]:.2f}"
                contrib = t("ra_contribution").format(val=pred.get("ra_above_avg", 0))
        else:
            prev_stat = t("no_prev_stats")
            if pred["type"] == "hitter":
                pred_stat = f"wOBA {pred['stat_value']:.3f}" if pred["stat_value"] else "—"
            elif pred["type"] == "pitcher":
                pred_stat = f"ERA {pred['stat_value']:.2f}" if pred["stat_value"] else "—"
            else:
                pred_stat = "—"
            ci = "—"
            contrib = t("historical_foreign_note")

        rows.append({
            t("col_player"): name,
            t("col_team"): team_disp(team),
            t("col_type"): ptype,
            t("col_prev_stat"): prev_stat,
            t("col_pred_stat"): pred_stat,
            t("col_range"): ci,
            t("col_contribution"): contrib,
            t("col_source"): source_map.get(name, ""),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                      height=min(800, len(df) * 40 + 60))
    else:
        st.warning(t("no_data"))


def page_historical(data: dict):
    st.markdown(f"### {t('historical_title')}")
    st.caption(t("historical_caption"))

    from foreign_bayes import load_foreign_historical

    hist = load_foreign_historical()

    # Summary
    h = hist["hitter"]
    p = hist["pitcher"]
    st.info(t("hist_summary").format(
        n_h=h["n"], mean_woba=h["mean_woba"], std_woba=h["std_woba"],
        n_p=p["n"], mean_era=p["mean_era"], std_era=p["std_era"],
    ))

    col1, col2 = st.columns(2)

    # Hitter wOBA distribution
    with col1:
        st.markdown(f"**{t('hitter_dist_title')}**")
        fig_h = go.Figure()
        # Generate samples from mean/std for display
        np.random.seed(42)
        samples_h = np.random.normal(h["mean_woba"], h["std_woba"], h["n"])
        fig_h.add_trace(go.Histogram(
            x=samples_h, nbinsx=20,
            marker_color="#00e5ff", opacity=0.7,
            name="wOBA",
        ))
        fig_h.add_vline(x=h["mean_woba"], line_dash="dash", line_color="#ff9944",
                        annotation_text=f'{t("hist_mean_label")}: {h["mean_woba"]:.3f}')
        fig_h.update_layout(
            height=300, xaxis_title="wOBA", yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), margin=dict(l=50, t=30, b=40, r=20),
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        )
        st.plotly_chart(fig_h, use_container_width=True, config={"staticPlot": True})
        st.caption(f"{t('hist_n_label')}: {h['n']} / {t('hist_std_label')}: {h['std_woba']:.3f}")

    # Pitcher ERA distribution
    with col2:
        st.markdown(f"**{t('pitcher_dist_title')}**")
        fig_p = go.Figure()
        samples_p = np.clip(np.random.normal(p["mean_era"], p["std_era"], p["n"]), 0, 10)
        fig_p.add_trace(go.Histogram(
            x=samples_p, nbinsx=20,
            marker_color="#ff6644", opacity=0.7,
            name="ERA",
        ))
        fig_p.add_vline(x=p["mean_era"], line_dash="dash", line_color="#ff9944",
                        annotation_text=f'{t("hist_mean_label")}: {p["mean_era"]:.2f}')
        fig_p.update_layout(
            height=300, xaxis_title="ERA", yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), margin=dict(l=50, t=30, b=40, r=20),
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        )
        st.plotly_chart(fig_p, use_container_width=True, config={"staticPlot": True})
        st.caption(f"{t('hist_n_label')}: {p['n']} / {t('hist_std_label')}: {p['std_era']:.2f}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    st.set_page_config(page_title="NPB 2026 Marcel+Stan/Bayes", layout="wide", page_icon="⚾")

    with st.sidebar:
        lang = st.radio(t("lang_toggle"), ["日本語", "English"], key="lang")
        st.markdown(f"## {t('app_title')}")
        st.caption(t("app_subtitle"))

        pages = {
            t("page_standings"): page_standings,
            t("page_foreign"): page_foreign_players,
            t("page_historical"): page_historical,
        }
        page_name = st.radio(t("nav_label"), list(pages.keys()))
        st.markdown("---")
        st.markdown(t("data_source"))

    data = load_all()
    pages[page_name](data)


if __name__ == "__main__":
    main()
