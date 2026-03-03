# NPB 2026 Prediction — Marcel + Bayesian Estimation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://npb-prediction-bayes.streamlit.app/)

Marcel法の予測結果をベースに、外国人選手のベイズ推定を追加した NPB 順位予測ダッシュボード。

## Marcel版との違い

| | [npb-prediction](https://github.com/yasumorishima/npb-prediction) | **npb-prediction-bayes**（本リポ） |
|---|---|---|
| 外国人選手の扱い | wRAA=0（リーグ平均） | ベイズ推定（前リーグ成績をNPB変換） |
| 予測幅 | なし | Monte Carlo simulation（5,000回） |
| 歴代外国人分析 | なし | 367名の初年度平均データ |

## ページ構成

1. **順位表比較** — Marcel-only vs Marcel+Bayes の勝数を左右に並べて比較
2. **外国人選手一覧** — 24名の予測詳細（ベイズ推定/歴代平均/前リーグ成績）
3. **歴代外国人分析** — 367名の初年度平均データ（打者wOBA分布、投手ERA分布）

## ベイズ推定の仕組み

NPB_stat = w × (prev_stat × cf_i) + (1 - w) × league_avg + noise

- `w ≈ 0.14`: 前リーグ成績の重み（残りはリーグ平均に回帰）
- `cf_i`: 個人変換係数（Normal分布）
- パラメータは [npb-bayes-projection](https://github.com/yasumorishima/npb-bayes-projection) の事後分布から取得

## データソース

- [プロ野球データFreak](https://baseball-data.com)
- [日本野球機構 NPB](https://npb.jp)

## セットアップ

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
