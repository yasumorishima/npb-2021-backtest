"""Bayesian foreign player prediction module.

Uses posterior parameters from npb-bayes-projection's Stan model (v1)
to predict NPB performance for foreign players based on previous league stats.
Stan/cmdstanpy not required — numpy sampling with hardcoded posterior parameters.

Hitter model:
    npb_wOBA = lg_avg + beta_woba * z_woba + beta_K * z_K + beta_BB * z_BB + noise

Pitcher model:
    npb_ERA = lg_avg + beta_era * z_era + beta_fip * z_fip + beta_K * z_K + beta_BB * z_BB + noise

Features are standardized using training-set (2015-2019) mean/sd.
Missing K%/BB%/FIP → z-score = 0 (= training mean, neutral contribution).
"""

import csv
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Stan v1 posterior parameters (from npb-bayes-projection CI run 2026-03-03)
# Each tuple is (mean, sd) of the posterior marginal
# ---------------------------------------------------------------------------
HITTER_PARAMS = {
    "beta_woba": (-0.0104, 0.0073),  # 90% CI: [-0.0247, 0.0040]
    "beta_K": (0.0043, 0.0074),      # 90% CI: [-0.0104, 0.0186]
    "beta_BB": (-0.0050, 0.0077),    # 90% CI: [-0.0201, 0.0101]
    "sigma": (0.0530, 0.0057),       # 90% CI: [0.0430, 0.0654]
}

PITCHER_PARAMS = {
    "beta_era": (0.0515, 0.1638),    # 90% CI: [-0.2732, 0.3686]
    "beta_fip": (-0.1160, 0.1676),   # 90% CI: [-0.4470, 0.2096]
    "beta_K": (-0.1828, 0.1387),     # 90% CI: [-0.4532, 0.0902]
    "beta_BB": (0.2545, 0.1393),     # 90% CI: [-0.0196, 0.5265]
    "sigma": (1.1007, 0.1042),       # 90% CI: [0.9146, 1.3229]
}

# Standardization params (training set 2015-2019, from player_conversion_details.csv)
HITTER_STD = {
    "woba_mean": 0.2576, "woba_sd": 0.0536,
    "k_mean": 26.5531, "k_sd": 9.0861,
    "bb_mean": 5.9938, "bb_sd": 4.1380,
}

PITCHER_STD = {
    "era_mean": 5.7924, "era_sd": 1.7693,
    "fip_mean": 5.1114, "fip_sd": 1.1899,
    "k_mean": 17.9643, "k_sd": 4.4411,
    "bb_mean": 9.8857, "bb_sd": 3.2621,
}

# Historical NPB foreign player first-year averages
# Source: npb-bayes-projection foreign_players_master.csv (367 players, 2015-2025)
# Hitter: PA≥50, Pitcher: IP≥20
HISTORICAL_FOREIGN_DEFAULTS = {
    "hitter": {"mean_woba": 0.318, "std_woba": 0.053, "n": 117},
    "pitcher": {"mean_era": 3.412, "std_era": 1.436, "n": 151},
}

_N_SAMPLES = 5000
_RNG = np.random.default_rng(42)


def _summarize(samples: np.ndarray) -> dict:
    """Summarize posterior predictive samples."""
    return {
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "hdi_80": (float(np.percentile(samples, 10)),
                   float(np.percentile(samples, 90))),
        "hdi_95": (float(np.percentile(samples, 2.5)),
                   float(np.percentile(samples, 97.5))),
    }


def _zscore(val: float | None, mean: float, sd: float) -> float:
    """Standardize a value. None → 0 (neutral, = training mean)."""
    if val is None:
        return 0.0
    return (val - mean) / sd


def predict_foreign_hitter(prev_woba: float,
                           league_avg_woba: float = 0.310,
                           prev_K_pct: float | None = None,
                           prev_BB_pct: float | None = None,
                           n_samples: int = _N_SAMPLES) -> dict:
    """Predict NPB wOBA for a foreign hitter with previous league stats."""
    z_w = _zscore(prev_woba, HITTER_STD["woba_mean"], HITTER_STD["woba_sd"])
    z_k = _zscore(prev_K_pct, HITTER_STD["k_mean"], HITTER_STD["k_sd"])
    z_bb = _zscore(prev_BB_pct, HITTER_STD["bb_mean"], HITTER_STD["bb_sd"])

    p = HITTER_PARAMS
    beta_woba = _RNG.normal(p["beta_woba"][0], p["beta_woba"][1], n_samples)
    beta_K = _RNG.normal(p["beta_K"][0], p["beta_K"][1], n_samples)
    beta_BB = _RNG.normal(p["beta_BB"][0], p["beta_BB"][1], n_samples)
    sigma = np.abs(_RNG.normal(p["sigma"][0], p["sigma"][1], n_samples))

    mu = league_avg_woba + beta_woba * z_w + beta_K * z_k + beta_BB * z_bb
    npb_woba = mu + _RNG.normal(0, sigma)
    return _summarize(npb_woba)


def predict_foreign_pitcher(prev_era: float,
                            league_avg_era: float = 3.50,
                            prev_fip: float | None = None,
                            prev_K_pct: float | None = None,
                            prev_BB_pct: float | None = None,
                            n_samples: int = _N_SAMPLES) -> dict:
    """Predict NPB ERA for a foreign pitcher with previous league stats."""
    z_e = _zscore(prev_era, PITCHER_STD["era_mean"], PITCHER_STD["era_sd"])
    z_f = _zscore(prev_fip, PITCHER_STD["fip_mean"], PITCHER_STD["fip_sd"])
    z_k = _zscore(prev_K_pct, PITCHER_STD["k_mean"], PITCHER_STD["k_sd"])
    z_bb = _zscore(prev_BB_pct, PITCHER_STD["bb_mean"], PITCHER_STD["bb_sd"])

    p = PITCHER_PARAMS
    beta_era = _RNG.normal(p["beta_era"][0], p["beta_era"][1], n_samples)
    beta_fip = _RNG.normal(p["beta_fip"][0], p["beta_fip"][1], n_samples)
    beta_K = _RNG.normal(p["beta_K"][0], p["beta_K"][1], n_samples)
    beta_BB = _RNG.normal(p["beta_BB"][0], p["beta_BB"][1], n_samples)
    sigma = np.abs(_RNG.normal(p["sigma"][0], p["sigma"][1], n_samples))

    mu = league_avg_era + beta_era * z_e + beta_fip * z_f + beta_K * z_k + beta_BB * z_bb
    npb_era = mu + _RNG.normal(0, sigma)
    npb_era = np.clip(npb_era, 0, None)
    return _summarize(npb_era)


def predict_no_prev_stats(player_type: str,
                          league_avg_woba: float = 0.310,
                          league_avg_era: float = 3.50,
                          n_samples: int = _N_SAMPLES) -> dict:
    """Predict for a foreign player with no previous league stats.

    Uses historical foreign player first-year averages as center
    (better than league average since NPB teams recruit above-average foreigners).
    """
    hist = _get_historical()
    if player_type == "hitter":
        center = hist["hitter"]["mean_woba"]
        sigma = HITTER_PARAMS["sigma"]
    else:
        center = hist["pitcher"]["mean_era"]
        sigma = PITCHER_PARAMS["sigma"]

    sigma_s = np.abs(_RNG.normal(sigma[0], sigma[1], n_samples))
    samples = center + _RNG.normal(0, sigma_s)
    if player_type == "pitcher":
        samples = np.clip(samples, 0, None)
    return _summarize(samples)


def _sample_hitter_woba(prev_woba: float, lg_woba: float,
                        prev_K_pct: float | None, prev_BB_pct: float | None,
                        n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n wOBA values from posterior for a foreign hitter."""
    z_w = _zscore(prev_woba, HITTER_STD["woba_mean"], HITTER_STD["woba_sd"])
    z_k = _zscore(prev_K_pct, HITTER_STD["k_mean"], HITTER_STD["k_sd"])
    z_bb = _zscore(prev_BB_pct, HITTER_STD["bb_mean"], HITTER_STD["bb_sd"])

    p = HITTER_PARAMS
    beta_woba = rng.normal(p["beta_woba"][0], p["beta_woba"][1], n)
    beta_K = rng.normal(p["beta_K"][0], p["beta_K"][1], n)
    beta_BB = rng.normal(p["beta_BB"][0], p["beta_BB"][1], n)
    sigma = np.abs(rng.normal(p["sigma"][0], p["sigma"][1], n))

    mu = lg_woba + beta_woba * z_w + beta_K * z_k + beta_BB * z_bb
    return mu + rng.normal(0, sigma)


def _sample_pitcher_era(prev_era: float, lg_era: float,
                        prev_fip: float | None,
                        prev_K_pct: float | None, prev_BB_pct: float | None,
                        n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n ERA values from posterior for a foreign pitcher."""
    z_e = _zscore(prev_era, PITCHER_STD["era_mean"], PITCHER_STD["era_sd"])
    z_f = _zscore(prev_fip, PITCHER_STD["fip_mean"], PITCHER_STD["fip_sd"])
    z_k = _zscore(prev_K_pct, PITCHER_STD["k_mean"], PITCHER_STD["k_sd"])
    z_bb = _zscore(prev_BB_pct, PITCHER_STD["bb_mean"], PITCHER_STD["bb_sd"])

    p = PITCHER_PARAMS
    beta_era = rng.normal(p["beta_era"][0], p["beta_era"][1], n)
    beta_fip = rng.normal(p["beta_fip"][0], p["beta_fip"][1], n)
    beta_K = rng.normal(p["beta_K"][0], p["beta_K"][1], n)
    beta_BB = rng.normal(p["beta_BB"][0], p["beta_BB"][1], n)
    sigma = np.abs(rng.normal(p["sigma"][0], p["sigma"][1], n))

    mu = lg_era + beta_era * z_e + beta_fip * z_f + beta_K * z_k + beta_BB * z_bb
    npb_era = mu + rng.normal(0, sigma)
    return np.clip(npb_era, 0, None)


def _sample_no_prev(player_type: str, lg_woba: float, lg_era: float,
                    n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample for a foreign player with no previous league stats.

    Uses historical foreign player first-year averages as center.
    """
    hist = _get_historical()
    if player_type == "hitter":
        center = hist["hitter"]["mean_woba"]
        sigma = HITTER_PARAMS["sigma"]
    else:
        center = hist["pitcher"]["mean_era"]
        sigma = PITCHER_PARAMS["sigma"]
    sigma_s = np.abs(rng.normal(sigma[0], sigma[1], n))
    samples = center + rng.normal(0, sigma_s)
    if player_type != "hitter":
        samples = np.clip(samples, 0, None)
    return samples


def simulate_team_wins_mc(
    pred_rs: float,
    pred_ra: float,
    missing_players: list,
    rs_scale: float,
    ra_scale: float,
    lg_woba: float,
    lg_era: float,
    woba_scale: float = 1.15,
    k: float = 1.72,
    n_samples: int = 5000,
) -> dict:
    """Monte Carlo simulation: posterior samples → team wins distribution.

    Perturbs the center estimate (pred_rs, pred_ra) by sampling from each
    missing player's posterior, then applies the Pythagorean formula to get
    the full wins distribution.  Diversification effect is captured naturally.
    """
    rng = np.random.default_rng(42)

    delta_rs = np.zeros(n_samples)
    delta_ra = np.zeros(n_samples)
    delta_wins = np.zeros(n_samples)

    for m in missing_players:
        b = m.get("bayes")
        kind = m.get("kind", "rookie")

        if kind == "rookie" or b is None:
            # Rookie: perturb directly in wins space (σ = 1.5 / 1.28 ≈ 1.17)
            delta_wins += rng.normal(0, 1.17, n_samples)
            continue

        if b.get("has_prev"):
            if b["type"] == "hitter":
                prev_stat = b.get("prev_stat")
                prev_K_pct = b.get("prev_K_pct")
                prev_BB_pct = b.get("prev_BB_pct")
                expected_pt = b.get("expected_pt", 400)
                woba_samples = _sample_hitter_woba(
                    prev_stat, lg_woba, prev_K_pct, prev_BB_pct,
                    n_samples, rng)
                wraa_samples = (woba_samples - lg_woba) / woba_scale * expected_pt
                wraa_mean = b.get("wraa_est", 0)
                delta_rs += (wraa_samples - wraa_mean) * rs_scale
            else:  # pitcher
                prev_stat = b.get("prev_stat")
                prev_fip = b.get("prev_fip")
                prev_K_pct = b.get("prev_K_pct")
                prev_BB_pct = b.get("prev_BB_pct")
                expected_pt = b.get("expected_pt", 100)
                era_samples = _sample_pitcher_era(
                    prev_stat, lg_era, prev_fip, prev_K_pct, prev_BB_pct,
                    n_samples, rng)
                ra_samples = (era_samples - lg_era) * expected_pt / 9.0
                ra_mean = b.get("ra_above_avg", 0)
                delta_ra += (ra_samples - ra_mean) * ra_scale
        else:
            # Foreign without prev stats
            if b["type"] == "hitter":
                woba_samples = _sample_no_prev("hitter", lg_woba, lg_era,
                                               n_samples, rng)
                expected_pt = b.get("expected_pt", 400)
                wraa_samples = (woba_samples - lg_woba) / woba_scale * expected_pt
                delta_rs += wraa_samples * rs_scale  # mean ≈ 0
            elif b["type"] == "pitcher":
                era_samples = _sample_no_prev("pitcher", lg_woba, lg_era,
                                              n_samples, rng)
                expected_pt = b.get("expected_pt", 100)
                ra_samples = (era_samples - lg_era) * expected_pt / 9.0
                delta_ra += ra_samples * ra_scale  # mean ≈ 0
            else:
                # unknown type → treat as rookie-like uncertainty
                delta_wins += rng.normal(0, 1.17, n_samples)

    sim_rs = pred_rs + delta_rs
    sim_ra = pred_ra + delta_ra
    sim_wpct = sim_rs**k / (sim_rs**k + sim_ra**k)
    sim_wins = np.clip(sim_wpct * 143 + delta_wins, 0, 143)

    return {
        "pred_W_low": float(np.percentile(sim_wins, 10)),
        "pred_W_high": float(np.percentile(sim_wins, 90)),
        "std_W": float(np.std(sim_wins)),
    }


def woba_to_wraa(pred_woba: float, lg_woba: float,
                 woba_scale: float = 1.15, pa: float = 400) -> float:
    """Convert predicted wOBA to wRAA estimate."""
    return (pred_woba - lg_woba) / woba_scale * pa


def era_to_ra_above_avg(pred_era: float, lg_era: float,
                        ip: float = 100) -> float:
    """Convert predicted ERA to runs allowed above average."""
    return (pred_era - lg_era) * ip / 9.0


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def load_foreign_2026() -> list[dict]:
    """Load data/foreign_2026.csv with previous league stats."""
    csv_path = Path(__file__).parent / "data" / "foreign_2026.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("prev_wOBA", "prev_ERA", "expected_pa", "expected_ip",
                        "prev_FIP", "prev_K_pct", "prev_BB_pct"):
                if row.get(key):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            rows.append(row)
    return rows


def load_foreign_historical() -> dict:
    """Load historical foreign player first-year stats from CSV.

    Returns dict with 'hitter' and 'pitcher' sub-dicts containing
    mean, std, and n for wOBA (hitter) / ERA (pitcher).
    Falls back to HISTORICAL_FOREIGN_DEFAULTS if CSV not found.
    """
    csv_path = Path(__file__).parent / "data" / "foreign_historical.csv"
    if not csv_path.exists():
        return HISTORICAL_FOREIGN_DEFAULTS

    hitter_wobas = []
    pitcher_eras = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptype = row.get("player_type", "").strip()
            if ptype == "hitter":
                pa_str = row.get("npb_first_year_PA", "")
                woba_str = row.get("npb_first_year_wOBA", "")
                if pa_str and woba_str:
                    try:
                        pa = float(pa_str)
                        woba = float(woba_str)
                        if pa >= 50:
                            hitter_wobas.append(woba)
                    except ValueError:
                        pass
            elif ptype == "pitcher":
                ip_str = row.get("npb_first_year_IP", "")
                era_str = row.get("npb_first_year_ERA", "")
                if ip_str and era_str:
                    try:
                        ip = float(ip_str)
                        era = float(era_str)
                        if ip >= 20:
                            pitcher_eras.append(era)
                    except ValueError:
                        pass

    result = {}
    if hitter_wobas:
        arr = np.array(hitter_wobas)
        result["hitter"] = {
            "mean_woba": float(np.mean(arr)),
            "std_woba": float(np.std(arr)),
            "n": len(arr),
        }
    else:
        result["hitter"] = HISTORICAL_FOREIGN_DEFAULTS["hitter"]

    if pitcher_eras:
        arr = np.array(pitcher_eras)
        result["pitcher"] = {
            "mean_era": float(np.mean(arr)),
            "std_era": float(np.std(arr)),
            "n": len(arr),
        }
    else:
        result["pitcher"] = HISTORICAL_FOREIGN_DEFAULTS["pitcher"]

    return result


_historical_cache: dict | None = None


def _get_historical() -> dict:
    """Return cached historical foreign player stats."""
    global _historical_cache
    if _historical_cache is None:
        _historical_cache = load_foreign_historical()
    return _historical_cache


def get_foreign_predictions(lg_woba: float = 0.310,
                            lg_era: float = 3.50,
                            woba_scale: float = 1.15) -> dict[str, dict]:
    """Compute Bayes predictions for all players in foreign_2026.csv.

    Returns: {npb_name: {pred, wraa_est|ra_above_avg, unc_wins, type, has_prev,
                         stat_label, stat_value, stat_range}}
    """
    data = load_foreign_2026()
    results: dict[str, dict] = {}

    for row in data:
        name = row["npb_name"]
        ptype = row["player_type"]

        if ptype == "hitter":
            if row["prev_wOBA"] is not None:
                pred = predict_foreign_hitter(
                    row["prev_wOBA"], lg_woba,
                    prev_K_pct=row.get("prev_K_pct"),
                    prev_BB_pct=row.get("prev_BB_pct"),
                )
                pa = row["expected_pa"] or 400
                wraa = woba_to_wraa(pred["mean"], lg_woba, woba_scale, pa)
                wraa_hi = woba_to_wraa(pred["hdi_80"][1], lg_woba, woba_scale, pa)
                wraa_lo = woba_to_wraa(pred["hdi_80"][0], lg_woba, woba_scale, pa)
                unc_wins = (wraa_hi - wraa_lo) / 10.0 / 2
                has_prev = True
                prev_stat = row["prev_wOBA"]
                expected_pt = pa
            else:
                pred = predict_no_prev_stats("hitter", lg_woba, lg_era)
                expected_pt = 400
                wraa = woba_to_wraa(pred["mean"], lg_woba, woba_scale,
                                    expected_pt)
                unc_wins = 1.5
                has_prev = False
                prev_stat = None

            results[name] = {
                "pred": pred, "wraa_est": wraa, "unc_wins": unc_wins,
                "type": ptype, "has_prev": has_prev,
                "has_historical": not has_prev,
                "prev_stat": prev_stat, "expected_pt": expected_pt,
                "prev_K_pct": row.get("prev_K_pct"),
                "prev_BB_pct": row.get("prev_BB_pct"),
                "stat_label": "wOBA",
                "stat_value": pred["mean"],
                "stat_range": pred["hdi_80"],
            }
        else:  # pitcher
            if row["prev_ERA"] is not None:
                pred = predict_foreign_pitcher(
                    row["prev_ERA"], lg_era,
                    prev_fip=row.get("prev_FIP"),
                    prev_K_pct=row.get("prev_K_pct"),
                    prev_BB_pct=row.get("prev_BB_pct"),
                )
                ip = row["expected_ip"] or 100
                ra_above = era_to_ra_above_avg(pred["mean"], lg_era, ip)
                ra_hi = era_to_ra_above_avg(pred["hdi_80"][1], lg_era, ip)
                ra_lo = era_to_ra_above_avg(pred["hdi_80"][0], lg_era, ip)
                unc_wins = abs(ra_hi - ra_lo) / 10.0 / 2
                has_prev = True
                prev_stat = row["prev_ERA"]
                expected_pt = ip
            else:
                pred = predict_no_prev_stats("pitcher", lg_woba, lg_era)
                expected_pt = 100
                ra_above = era_to_ra_above_avg(pred["mean"], lg_era,
                                               expected_pt)
                unc_wins = 1.5
                has_prev = False
                prev_stat = None

            results[name] = {
                "pred": pred, "ra_above_avg": ra_above, "unc_wins": unc_wins,
                "type": ptype, "has_prev": has_prev,
                "has_historical": not has_prev,
                "prev_stat": prev_stat, "expected_pt": expected_pt,
                "prev_fip": row.get("prev_FIP"),
                "prev_K_pct": row.get("prev_K_pct"),
                "prev_BB_pct": row.get("prev_BB_pct"),
                "stat_label": "ERA",
                "stat_value": pred["mean"],
                "stat_range": pred["hdi_80"],
            }

    return results
