# npb-prediction-bayes

NPB（日本プロ野球）の外国人選手初年度成績をベイズ推定し、チーム順位予測に組み込むプロジェクト。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://npb-prediction-bayes.streamlit.app/)

**Article:** [Zenn（日本語）](https://zenn.dev/shogaku/articles/npb-bayes-projection-story) / [DEV.to（English）](https://dev.to/yasumorishima/beyond-marcel-adding-bayesian-regression-to-npb-baseball-predictions-a-15-step-journey-5f86)

---

## 何が違うのか — [npb-prediction](https://github.com/yasumorishima/npb-prediction) との比較

| 項目 | npb-prediction | **npb-prediction-bayes（本リポ）** |
|---|---|---|
| 外国人選手の扱い | wRAA = 0（リーグ平均と同じ） | ベイズ推定（前リーグ成績 → NPB換算） |
| 予測幅（不確実性） | なし | Monte Carlo simulation（5,000回） |
| 歴代外国人データ | なし | 367名の初年度実績（2015-2025） |
| 依存リポ | なし | [npb-stan-research](https://github.com/yasumorishima/npb-stan-research)（後験パラメータ） |

**問題意識**: npb-predictionでは新外国人選手をwRAA=0（リーグ平均相当）と仮定している。主力外国人が複数在籍するチームではこの誤差が順位予測に影響しうる。

---

## 開発の背景（15ステップの試行錯誤）

このプロジェクトは [npb-stan-research](https://github.com/yasumorishima/npb-stan-research) での15ステップにわたる統計モデリングの成果をアプリに統合したもの。

詳細な開発記録 → **[Marcel法の限界を超えたい — NPB予測にベイズ回帰を導入した15ステップの記録（Zenn）](https://zenn.dev/shogaku/articles/npb-bayes-projection-story)**

### なぜK%/BB%が有効か

| 指標 | 環境依存 | 特性 |
|---|---|---|
| wOBA / ERA | 高い（球場・リーグレベルに依存） | 「結果指標」。リーグ間変換精度が低い |
| **K% / BB%** | **低い** | **「スキル指標」。三振・四球は選手本来の能力を反映** |

前リーグのK%/BB%を特徴量に追加することで、環境依存の高いwOBA/ERAだけを使う場合より予測精度が改善した。

---

## ベイズ推定の仕組み

### モデル式

```
# 打者
npb_wOBA = lg_avg + β_woba × z_woba + β_K × z_K + β_BB × z_BB + ε

# 投手
npb_ERA = lg_avg + β_era × z_era + β_fip × z_fip + β_K × z_K + β_BB × z_BB + ε
```

特徴量は前リーグでの成績（wOBA / ERA / FIP / K% / BB%）をz-score変換したもの。前リーグ成績がない選手はz=0（= 歴代外国人平均）として扱う。

### 後験パラメータ（npb-bayes-projection Stan v1 モデルから）

| パラメータ | 打者 mean (sd) | 投手 mean (sd) |
|---|---|---|
| β_woba / β_era | −0.0104 (0.0073) | +0.0515 (0.1638) |
| β_K | +0.0043 (0.0074) | −0.1828 (0.1387) |
| β_BB | −0.0050 (0.0077) | +0.2545 (0.1393) |
| σ（ノイズ） | 0.0530 (0.0057) | 1.1007 (0.1042) |

- β_woba の前リーグ→NPB寄与は小さく（−0.010）、**NPBリーグ平均への回帰が支配的**
- 投手β_K（−0.183）は「奪三振が多い投手がNPBでも抑えやすい」傾向を反映
- σが大きい（打者0.053 / 投手1.10）= 個人差のばらつきも大きい

### 変換係数（重み）

- `w ≈ 0.14`: 前リーグ成績の寄与（残り0.86はリーグ平均への回帰）
- cmdstanpy 不要 — 後験パラメータをハードコードし、NumPyで Monte Carlo サンプリング

---

## 歴代外国人選手データ（2015-2025）

### 初年度平均実績（367名）

| カテゴリ | 対象 | 平均 | 標準偏差 |
|---|---|---|---|
| 打者 wOBA | PA ≥ 50（117名） | .318 | .053 |
| 投手 ERA | IP ≥ 20（151名） | 3.41 | 1.44 |

前リーグ成績データがない選手（独立リーグ出身等）には、この歴代平均を初期値として使用。

### 変換係数（リーグ間）

| 出身リーグ | wOBA比 [95%CI] | ERA比 [95%CI] | サンプル |
|---|---|---|---|
| MLB → NPB | 1.235 [1.14, 1.32] | 0.579 [0.51, 0.67] | 打者56 / 投手74 |
| AAA → NPB | 1.271 [1.01, 1.47] | 0.462 [0.25, 0.69] | 打者9 / 投手6 |

打者のwOBAは約24%向上、投手のERAは約42%改善（NPBとMLBのレベル差を反映）。

---

## 検証結果（外国人モデル）

### バックテスト（2020-2025）

| モデル | MAE | ベースライン | 改善率 |
|---|---|---|---|
| 打者 v0（wOBAのみ） | 0.0330 | 0.0337 | -2.1% |
| **打者 v1（+K%/BB%）** | **0.0325** | 0.0337 | **-3.8%** |
| 投手 v0（ERAのみ） | 0.749 | 0.749 | ±0% |
| **投手 v1（+K%/BB%/FIP）** | **0.736** | 0.749 | **-1.7%** |

ベースライン = 外国人初年度平均（リーグ平均相当）。K%/BB% 追加で打者の精度が改善。

---

## 2026年 外国人選手（24名）

各選手について「ベイズ推定値」「歴代平均」「前リーグ成績」を5,000回サンプリングして予測幅を算出。

出身リーグ構成: MLB / AAA / 独立リーグ / NPB他球団 など。

---

## Streamlitアプリ — 3ページ構成

**https://npb-prediction-bayes.streamlit.app/**

### 1. 順位表比較

Marcel-only（外国人=平均）と Marcel+Bayes（ベイズ補正あり）を左右に並べて比較。チームごとの「ベイズ効果」（勝利数差）と予測幅（80%信頼区間）を表示。

### 2. 外国人選手一覧

24名全員の予測詳細（ベイズ推定値 + 90%信頼区間 / 歴代初年度平均との比較 / 前リーグ成績）。

### 3. 歴代外国人分析

367名の初年度実績データ（2015-2025）を可視化（出身リーグ別分布 / 前リーグ成績とNPB初年度成績の散布図）。

---

## ファイル構成

| ファイル | 内容 |
|---|---|
| `streamlit_app.py` | Streamlit ダッシュボード本体（日本語/英語対応） |
| `foreign_bayes.py` | ベイズ推定モジュール（Monte Carlo サンプリング） |
| `roster_current.py` | 2026年NPB支配下登録選手名簿 |
| `config.py` | 対象年度設定 |
| `translations.py` | 日本語/英語翻訳辞書 |

### データ

| ファイル | 内容 |
|---|---|
| `data/foreign_2026.csv` | 2026年外国人選手24名の前リーグ成績 |
| `data/foreign_historical.csv` | 歴代外国人選手367名の初年度実績（2015-2025） |
| `data/foreign_2026_names.csv` | 外国人選手の表記ゆれ対応マスタ |
| `data/projections/` | Marcel法予測CSV（npb-predictionと共有） |

---

## 主な発見

1. **前リーグのwOBA/ERAは単独では使えない** — 環境依存が強く、リーグ間変換でノイズが大きい。K%/BB%（スキル指標）との組み合わせが有効
2. **β_wobа の寄与は小さい（−0.010）** — 前リーグ成績よりもNPBリーグ平均への回帰が支配的（w ≈ 0.14）
3. **σが大きい（打者0.053 / 投手1.10）** — 個人差のばらつきが本質的に大きく、幅のある予測（Monte Carlo）が必要
4. **外国人モデルとチームレベルの改善は限定的** — 外国人選手は全選手の1割程度。チーム順位予測への影響はベイズ補正があっても数勝以内

---

## 予測の限界と注意事項

- **前リーグ成績の信頼性**: 少数イニング/独立リーグのデータは信頼区間が広くなる
- **NPB適応の個人差**: モデルは平均的な変換係数を使用。個人の適応力は反映できない
- **外国人選手以外**: Marcel法ベース（npb-prediction と同じ）

---

## セットアップ

```bash
git clone https://github.com/yasumorishima/npb-prediction-bayes
cd npb-prediction-bayes
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 今後の予定

- [ ] **パークファクター補正**: [npb-prediction](https://github.com/yasumorishima/npb-prediction) の試合別スコアデータからPFを自動計算し、球場補正を予測に反映（甲子園・神宮など投手有利球場の過小評価を修正）
- [ ] 外国人選手初年度実績の集計 → 計算対象外選手への初期値実装
- [ ] 精度改善（打球データ等の追加特徴量）
- [ ] GCP統合 — BigQuery データロード + BQML モデル + Cloud Run API（[npb-prediction](https://github.com/yasumorishima/npb-prediction) と同パターンで構築予定）

---

## 関連リポジトリ

| リポジトリ | 内容 |
|---|---|
| [npb-prediction](https://github.com/yasumorishima/npb-prediction) | Marcel法ベース版（Stan/Ridge補正含む） |
| [npb-stan-research](https://github.com/yasumorishima/npb-stan-research) | Stanベイズ統計モデル（本リポの後験パラメータ生成元） |

## データソース

- [プロ野球データFreak](https://baseball-data.com)
- [日本野球機構 NPB](https://npb.jp)
