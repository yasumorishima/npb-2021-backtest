"""Translation strings for NPB Prediction Bayes dashboard (Japanese / English)."""

TEXTS: dict[str, dict[str, str]] = {
    "ja": {
        "app_title": "NPB 2026 予測（Marcel法 + ベイズ推定）",
        "app_subtitle": "Marcel予測 ＋ 外国人選手のベイズ推定で順位を比較",
        "lang_toggle": "Language",

        # --- Page names ---
        "page_standings": "順位表比較",
        "page_foreign": "外国人選手一覧",
        "page_historical": "歴代外国人分析",
        "nav_label": "ページ選択",

        # --- Common ---
        "central_league": "セ・リーグ",
        "pacific_league": "パ・リーグ",
        "wins_suffix": "勝",
        "losses_suffix": "敗",
        "no_data": "データが読み込めませんでした",
        "data_source": "データソース: [プロ野球データFreak](https://baseball-data.com) / [日本野球機構 NPB](https://npb.jp)",

        # --- Standings comparison ---
        "standings_title": "2026年 順位予測比較",
        "standings_caption": "Marcel法のみ vs Marcel法＋ベイズ推定（外国人選手込み）",
        "marcel_only": "Marcel法のみ",
        "marcel_bayes": "Marcel法 + ベイズ推定",
        "pred_wins_label": "予測勝数",
        "pred_range": "幅: {lo}〜{hi}勝",
        "pred_range_brief": "オレンジの縦線 = 予測幅。Monte Carloシミュレーション（5,000回）で算出",
        "standings_info": (
            "⚠️ **これは統計モデルの自動計算結果です。作者の予想・応援とは無関係です。**\n\n"
            "左列: Marcel法のみ（外国人選手はリーグ平均 wRAA=0 で計算）\n"
            "右列: Marcel法＋ベイズ推定（外国人選手の前リーグ成績を変換して加味）"
        ),
        "method_expander": "予測方法の説明",
        "method_content": (
            "**Marcel法のみ（左列）**\n"
            "- 過去3年のNPB成績を5:4:3で加重平均し、年齢で調整\n"
            "- NPBでの過去データがない選手の貢献はwRAA=0（リーグ平均）として扱う\n\n"
            "**Marcel法＋ベイズ推定（右列）**\n"
            "- Marcel法の結果をベースに、NPBデータがない外国人選手にベイズ推定を適用\n"
            "- 前リーグ成績（wOBA/ERA）をNPBスケールに変換（Shrinkageモデル）\n"
            "- 前リーグ成績がない外国人は歴代NPB外国人初年度平均を使用\n"
            "- 予測幅はMonte Carloシミュレーション（5,000回）で算出\n\n"
            "**ピタゴラス勝率**: 得点^1.72 ÷ (得点^1.72 + 失点^1.72) × 143試合"
        ),

        # --- Foreign players ---
        "foreign_title": "2026年 外国人選手予測一覧",
        "foreign_caption": "ベイズ推定による予測値と信頼区間。前リーグ成績をNPBスケールに変換",
        "col_player": "選手名",
        "col_team": "チーム",
        "col_type": "種別",
        "col_prev_stat": "前リーグ成績",
        "col_pred_stat": "NPB予測",
        "col_range": "80%信頼区間",
        "col_contribution": "貢献",
        "col_source": "ソース",
        "hitter": "打者",
        "pitcher": "投手",
        "no_prev_stats": "前リーグ成績なし",
        "historical_foreign_note": "歴代外国人初年度平均を基準に計算",
        "wraa_contribution": "wRAA {val:+.1f}",
        "ra_contribution": "RA {val:+.1f}",

        # --- Historical analysis ---
        "historical_title": "歴代NPB外国人選手 初年度分析",
        "historical_caption": "2015〜2025年にNPBでプレーした外国人選手367名の初年度成績",
        "hitter_dist_title": "打者wOBA分布（PA≥50）",
        "pitcher_dist_title": "投手ERA分布（IP≥20）",
        "hist_mean_label": "平均",
        "hist_n_label": "対象人数",
        "hist_std_label": "標準偏差",
        "hist_summary": "打者{n_h}名: 平均wOBA {mean_woba:.3f}（σ={std_woba:.3f}） / 投手{n_p}名: 平均ERA {mean_era:.2f}（σ={std_era:.2f}）",
    },

    "en": {
        "app_title": "NPB 2026 Predictions (Marcel + Bayesian)",
        "app_subtitle": "Marcel projections + Bayesian foreign player estimation",
        "lang_toggle": "Language",

        # --- Page names ---
        "page_standings": "Standings Comparison",
        "page_foreign": "Foreign Players",
        "page_historical": "Historical Analysis",
        "nav_label": "Navigation",

        # --- Common ---
        "central_league": "Central League",
        "pacific_league": "Pacific League",
        "wins_suffix": "W",
        "losses_suffix": "L",
        "no_data": "Failed to load data",
        "data_source": "Data: [Baseball Data Freak](https://baseball-data.com) / [NPB Official](https://npb.jp)",

        # --- Standings comparison ---
        "standings_title": "2026 Projected Standings Comparison",
        "standings_caption": "Marcel-only vs Marcel + Bayesian estimation (with foreign players)",
        "marcel_only": "Marcel Only",
        "marcel_bayes": "Marcel + Bayesian",
        "pred_wins_label": "Projected Wins",
        "pred_range": "Range: {lo}–{hi}W",
        "pred_range_brief": "Orange bars = prediction range via Monte Carlo simulation (5,000 draws)",
        "standings_info": (
            "⚠️ **These are automated statistical model outputs — not the author's predictions.**\n\n"
            "Left column: Marcel-only (foreign players set to league-average wRAA=0)\n"
            "Right column: Marcel + Bayesian (foreign player prior stats converted to NPB scale)"
        ),
        "method_expander": "Methodology",
        "method_content": (
            "**Marcel Only (left column)**\n"
            "- 3-year weighted average (5:4:3) of NPB stats with age adjustment\n"
            "- Players without NPB history are treated as wRAA=0 (league average)\n\n"
            "**Marcel + Bayesian (right column)**\n"
            "- Marcel projections as base, Bayesian estimation for foreign newcomers\n"
            "- Prior league stats (wOBA/ERA) converted to NPB scale (Shrinkage model)\n"
            "- Foreign players without prior stats use historical NPB foreign 1st-year averages\n"
            "- Prediction ranges from Monte Carlo simulation (5,000 draws)\n\n"
            "**Pythagorean Win%**: RS^1.72 / (RS^1.72 + RA^1.72) × 143 games"
        ),

        # --- Foreign players ---
        "foreign_title": "2026 Foreign Player Projections",
        "foreign_caption": "Bayesian projections with credible intervals. Prior league stats converted to NPB scale",
        "col_player": "Player",
        "col_team": "Team",
        "col_type": "Type",
        "col_prev_stat": "Prior League Stats",
        "col_pred_stat": "NPB Projection",
        "col_range": "80% CI",
        "col_contribution": "Contribution",
        "col_source": "Source",
        "hitter": "Batter",
        "pitcher": "Pitcher",
        "no_prev_stats": "No prior stats",
        "historical_foreign_note": "Based on historical foreign 1st-year average",
        "wraa_contribution": "wRAA {val:+.1f}",
        "ra_contribution": "RA {val:+.1f}",

        # --- Historical analysis ---
        "historical_title": "Historical NPB Foreign Players — First-Year Analysis",
        "historical_caption": "367 foreign players who played in NPB from 2015 to 2025",
        "hitter_dist_title": "Batter wOBA Distribution (PA≥50)",
        "pitcher_dist_title": "Pitcher ERA Distribution (IP≥20)",
        "hist_mean_label": "Mean",
        "hist_n_label": "Sample Size",
        "hist_std_label": "Std Dev",
        "hist_summary": "Batters (n={n_h}): Mean wOBA {mean_woba:.3f} (σ={std_woba:.3f}) / Pitchers (n={n_p}): Mean ERA {mean_era:.2f} (σ={std_era:.2f})",
    },
}

# English team name mapping
TEAM_NAME_EN: dict[str, str] = {
    "DeNA": "BayStars",
    "巨人": "Giants",
    "阪神": "Tigers",
    "広島": "Carp",
    "中日": "Dragons",
    "ヤクルト": "Swallows",
    "ソフトバンク": "Hawks",
    "日本ハム": "Fighters",
    "楽天": "Eagles",
    "ロッテ": "Marines",
    "オリックス": "Buffaloes",
    "西武": "Lions",
}
