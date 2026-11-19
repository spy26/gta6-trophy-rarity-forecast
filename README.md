# GTA 6 Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/spy26/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/spy26/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/competitions/gta6-trophy-rarity-forecast-01):
forecast, for every GTA VI trophy on PlayStation 5, five quantiles of its earn
rate on Exophase at the case's target. One program, `main.py` (Python 3.13,
standard library only, no network, deterministic).

## The model

A hybrid of two structural bets, averaged in logit space (x = ln(1 + hours
since the game's release midnight)):

- **Trend (B).** Per trophy, OLS of logit(rate) on x over the last 24 h of
  reads, read at the target. When the page's history is smooth this is the
  best extrapolator: trophy rates saturate slowly in log-time.
- **Decomposition (C).** The shown rate is earners / players tracked, and the
  two move for different reasons: earners grow as players play, while the
  tracked base grows as new players join Exophase — and a surge of young
  joiners dilutes every rate at once, which a rate-level trend would misread
  as trophies getting rarer. Earners E(t) and the tracked base N(t) are each
  fitted log-linear in x over the last 24 h (each trophy's earner slope shrunk
  toward its tier's mean, slopes taken conservatively as the smaller of the
  full-window and recent-third fits), both extrapolated with a saturating
  step, forecast rate = 100 * E / N. Spread grows with horizon, drift and
  fit residual (~1.0 logit units at the final's 4.5-day horizon).

When the history moves, the median is the logit-average of the two medians
with the decomposition's spread. When the history is **frozen** (every read
identical, no refresh seen — the practice cases), the forecast is the
decomposition alone with a −1.1 logit median shift (the next refresh brings
a wave of newly tracked players; the house's own account says the first
refresh took every rate but the 100% one down) and a 1.2 logit spread. The
trophy at 100% in the last read keeps a near-100% row ([98.5, 99.7, 99.99,
99.99, 99.99]): it survived a six-fold growth of the tracked base without
dropping. Persistence is only a fallback for trophies with too little
history to fit.

## Benchmarks

Local scorer implements the pinball loss on the logit scale exactly as the
rules define it. Two caveats, stated plainly:

1. **The kit ships no practice answers**, so the practice loss cannot be
   computed exactly locally. The practice rows below score each variant
   against three dilution scenarios for the answer, built on the one fact the
   rules state about the 04:30 refresh (3,868 players tracked, every rate but
   the 100% one lower): earners unchanged / ×1.5 / ×2.5 while the tracked
   base grows 638 → 3,868 (p00) or → 5,200 (p01). The house baseline's
   stated practice loss is ~1.51, which sits between the ×1.0 and ×1.5
   scenario columns — the scenarios bracket the truth.
2. **Backtest cuts from the real history are degenerate**: every practice
   read is byte-identical (frozen page), so any persistence-like model scores
   ~0 and the cuts carry no information about the refresh jump.

| Evaluation | Baseline (starter) | A persistence σ=1.5 | B log-time | C E/N | **D hybrid (this entry)** |
|---|---|---|---|---|---|
| Practice scenarios, mean of 6 cells | 1.398 | 0.807 | 1.372 | 0.351 | **0.351** |
| Synthetic 90 min (frozen → 1st refresh) | 0.961 | 0.525 | 0.927 | 1.198 | 1.198 |
| Synthetic 6 h | 0.208 | 0.368 | 0.251 | 0.255 | 0.245 |
| Synthetic 24 h | 0.252 | 0.393 | 0.115 | 0.177 | 0.156 |
| Synthetic 4.5 d, R1 fast-saturating base | 0.492 | 0.495 | 0.231 | 0.230 | **0.213** |
| Synthetic 4.5 d, R2 still-surging base | 0.467 | 0.487 | 0.233 | 0.238 | **0.219** |
| Synthetic 4.5 d, R3 fast saturation | 0.423 | 0.470 | 0.244 | 0.221 | **0.212** |
| Synthetic 4.5 d, R4 tracked-base jump +40% | 0.448 | 0.471 | 0.301 | 0.234 | 0.249 |
| **Synthetic 4.5 d, mean of R1–R4** | **0.458** | **0.481** | **0.252** | **0.231** | **0.223** |

The synthetic series is a first-principles Exophase simulation (cohort model:
players join the tracked base over time, a joiner's earn chance depends on
how long they have played, so young cohorts dilute the shown rate; the page
refreshes discretely). D wins the practice scenarios outright, wins 3 of 4
long-horizon regimes, and has the best 4.5-day mean — it is the only variant
that is never the worst anywhere except the synthetic 90-minute cut, where
its frozen-history dilution shift is wrong for that simulation (its rates
rise at the first refresh) but right for the real launch night (the rules
state the real rates fell). B is the best smooth extrapolator and C the most
robust to tracked-base jumps; averaging them in logit space halves the
damage when either structure is wrong for the week. A (persistence) cannot
win the final: its median never moves, and at 4.5 days the truth moves ~8.9
points on average.

Coverage: D's 90% band covers 0.98 of trophies in the practice scenarios
and 0.96–1.00 in the synthetic cuts (slightly wide against the noiseless
synthetic truth, deliberately — the real final adds refresh staleness and
model risk the simulation lacks).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`): `{"trophies": [{"id":
n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per trophy, five
non-decreasing quantiles in 0..100, under 65,536 bytes. Two runs write
byte-identical files; a two-case practice run takes ~0.06 s.

## Files

- `main.py` — the whole entry (the only entrypoint; no `main.ts`).
- `LICENSE` — MIT.
