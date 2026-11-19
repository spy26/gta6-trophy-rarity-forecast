"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at the case's target instant.

The model is a hybrid of two structural bets, averaged in logit space:

  * Trend (B). For each trophy, a straight line is fitted to the logit of its
    earn rate against log-time -- x = ln(1 + hours since the game's release
    midnight) -- over the last 24 hours of reads, and read at the target.
    When the page's history is smooth this is the best extrapolator: trophy
    rates saturate slowly in log-time, so the recent slope carries a long way.

  * Decomposition (C). The shown rate is earners divided by players tracked,
    and the two move for different reasons: earners grow as players play,
    while the tracked base grows as new players join Exophase -- and a surge
    of young joiners dilutes every rate at once, which a rate-level trend
    would misread as trophies getting rarer. So earners E(t) and the tracked
    base N(t) are each fitted as log-linear in x over the last 24 hours
    (each trophy's earner slope shrunk toward its tier's mean slope), both
    extrapolated with a saturating step, and the forecast rate is 100 * E / N.

When the history moves, the forecast median is the logit-average of the two
medians and the spread is the decomposition's: the average halves the damage
when either structure is the wrong one for the week. When the history is
frozen -- every read identical, no refresh seen -- neither trend exists, so
the forecast is the decomposition alone with two adjustments that the
launch-night data forces: the median is shifted down 1.1 logit units (the
next refresh brings a wave of newly tracked players, and the house's own
account says the first refresh took every rate but the 100% one down), and
the spread is widened to 1.2 logit units (we have seen no refresh at all,
so the jump's size is mostly unknown).

The trophy sitting at 100% in the last read is special: it survived a
six-fold growth of the tracked base without dropping, so its median stays at
99.99% with only a modest lower tail.

Spread, when the history moves, is a logit standard deviation built from a
floor (0.60), a term growing with the square root of the horizon in days
(0.34), a term proportional to how far the median moves from the last read
(0.30 of the drift), and the earner fit's residual -- about 1.0 logit units
at the final's 4.5-day horizon, which is the honest scale of a launch-week
forecast (trend error, saturation risk, refresh staleness, tracked-base
jumps). Persistence is only a fallback, for trophies with fewer than two
usable reads.

Python 3.13, standard library only. No network, no randomness, no state
between runs: two runs write byte-identical files. Reads every <id>.in in
$MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

WINDOW_HOURS = 24.0
# Saturating extrapolation: the step taken is TAU * (1 - exp(-dx / TAU)).
TAU_DAMP = 2.0
# Earner slopes shrink toward the tier mean with this pseudo-count.
SHRINK_K = 6.0
# Spread parameters (logit units).
SIGMA0 = 0.60
SIGMA_H = 0.34
DRIFT_FRAC = 0.30
DRIFT_CAP = 2.5
# Frozen-history branch (no refresh seen at all).
FROZEN_SIGMA = 1.20
FROZEN_SHIFT = -1.10
# Fallback spread when a trophy has too little history to fit.
FALLBACK_SIGMA = 1.0


def clamp(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (the starter's convention)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def quantiles(lg, sigma):
    return [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(case, when):
    """x = ln(1 + hours since the game's release midnight, UTC)."""
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    hours = max(0.0, (when - release).total_seconds() / 3600.0)
    return math.log1p(hours)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, n) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        s = math.sqrt(max(var, 0.0))
    else:
        s = 0.0
    return (a, b, s, n)


def conservative_slope(xs, ys):
    """The smaller-|slope| of the full-window fit and the fit over the last
    third of the window. Growth curves bend in log-time, so the full-window
    slope overstates the forward slope; the recent slope is the honest one,
    and taking the smaller of the two guards against a noisy recent fit."""
    full = ols(xs, ys)
    if full is None:
        return None
    if len(xs) >= 6:
        k = max(3, len(xs) // 3)
        recent = ols(xs[-k:], ys[-k:])
        if recent is not None and abs(recent[1]) < abs(full[1]):
            return recent
    return full


def last_known_rate(case, index):
    """The trophy's rate in the latest read that lists it, else None."""
    for read in reversed(case["history"]):
        rate = read["rates"][index]
        if rate is not None:
            return rate
    return None


def window_reads(case):
    """Reads from the last WINDOW_HOURS before the checkpoint; all reads if
    that leaves fewer than three."""
    at = parse_time(case["at"])
    win = [h for h in case["history"]
           if (at - parse_time(h["at"])).total_seconds() / 3600.0 <= WINDOW_HOURS]
    return win if len(win) >= 3 else list(case["history"])


def history_is_frozen(case, win):
    """True when no refresh has been seen: every rate and the tracked count
    are identical across the window."""
    if len({h["players_tracked"] for h in win}) > 1:
        return False
    for i in range(len(case["trophies"])):
        if len({h["rates"][i] for h in win if h["rates"][i] is not None}) > 1:
            return False
    return True


def trend_median(case, win, win_x, index, x_target):
    """Variant B: logit(rate) linear in log-time, read at the target."""
    xs, ys = [], []
    for read, xv in zip(win, win_x):
        rate = read["rates"][index]
        if rate is None:
            continue
        xs.append(xv)
        ys.append(logit(rate))
    fit = ols(xs, ys)
    if fit is None:
        return None
    a, b, _s, _n = fit
    return a + b * x_target


def decomposition_forecast(case, win, win_x, index, x_last, x_target, days,
                           tier_mean_slope):
    """Variant C: model earners E(t) and the tracked base N(t) separately.

    Returns (median_logit, sigma) for the trophy, or None when the trophy
    has too little history to fit. The median is the logit of 100 * E / N
    at the target, with the move from the last read capped; the spread grows
    with the horizon, the size of the move, and the earner fit's residual."""
    xs_e, ys_e = [], []
    for read, xv in zip(win, win_x):
        rate = read["rates"][index]
        if rate is None:
            continue
        earners = max(0.5, rate * max(1, read["players_tracked"]) / 100.0)
        xs_e.append(xv)
        ys_e.append(math.log(earners))
    fit_e = conservative_slope(xs_e, ys_e)
    if fit_e is None:
        return None
    a_e, g_e, s_e, n_e = fit_e
    g_e = min(4.0, max(-1.0, g_e))
    # Shrink the earner slope toward the tier's mean slope: trophies of one
    # tier face similar grind, so a noisy per-trophy slope borrows strength.
    g_star = (n_e * g_e + SHRINK_K * tier_mean_slope) / (n_e + SHRINK_K)
    log_e_last = a_e + g_e * win_x[-1]

    xs_n = win_x
    ys_n = [math.log(max(1, h["players_tracked"])) for h in win]
    fit_n = conservative_slope(xs_n, ys_n)
    if fit_n is None:
        log_n_last = math.log(max(1, case["history"][-1]["players_tracked"]))
        g_n = 0.0
    else:
        a_n, g_n, _s_n, _n = fit_n
        g_n = min(3.0, max(-0.5, g_n))
        log_n_last = a_n + g_n * win_x[-1]

    dx = max(0.0, x_target - x_last)
    step = TAU_DAMP * (1.0 - math.exp(-dx / TAU_DAMP)) if dx > 0 else 0.0
    log_e_t = log_e_last + g_star * step
    log_n_t = log_n_last + g_n * step
    rate_med = 100.0 * math.exp(log_e_t - log_n_t)
    median = logit(rate_med)
    last = last_known_rate(case, index)
    drift = median - logit(last if last is not None else 1.0)
    drift = min(DRIFT_CAP, max(-DRIFT_CAP, drift))
    median = logit(last if last is not None else 1.0) + drift
    sigma = math.sqrt(SIGMA0 ** 2 + (SIGMA_H ** 2) * days
                      + (DRIFT_FRAC * abs(drift)) ** 2 + (0.5 * s_e) ** 2)
    return (median, sigma)


def forecast_case(case):
    if not case["history"] or not case["trophies"]:
        # No history at all: a flat 1% median with a wide spread still beats
        # crashing (a crash scores the case as persistence anyway).
        return {"trophies": [{"id": t["id"],
                              "q": quantiles(logit(1.0), 1.5)}
                             for t in case["trophies"]]}
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    x_last = log_time(case, at)
    x_target = log_time(case, target)
    days = max(0.0, case["horizon_minutes"] / 1440.0)
    win = window_reads(case)
    win_x = [log_time(case, parse_time(h["at"])) for h in win]
    frozen = history_is_frozen(case, win)

    # Tier mean earner slopes, for the shrinkage in the decomposition.
    tier_slopes = {}
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for read, xv in zip(win, win_x):
            rate = read["rates"][i]
            if rate is None:
                continue
            xs.append(xv)
            ys.append(math.log(max(0.5, rate * max(1, read["players_tracked"]) / 100.0)))
        fit = conservative_slope(xs, ys)
        if fit is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(
                min(4.0, max(-1.0, fit[1])))
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last = last_known_rate(case, i)
        if last is None:
            rows.append({"id": trophy["id"], "q": quantiles(logit(1.0), 1.5)})
            continue
        if last >= 99.99:
            # Survived a six-fold tracked-base growth at 100%: stays ~100%.
            rows.append({"id": trophy["id"], "q": [98.5, 99.7, 99.99, 99.99, 99.99]})
            continue
        dec = decomposition_forecast(case, win, win_x, i, x_last, x_target,
                                     days, tier_mean.get(trophy["tier"], 0.0))
        if dec is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles(logit(last), FALLBACK_SIGMA)})
            continue
        median_c, sigma_c = dec
        if frozen:
            # No refresh seen: expect the pending refresh to dilute, and
            # admit how little we know about the jump's size.
            rows.append({"id": trophy["id"],
                         "q": quantiles(median_c + FROZEN_SHIFT, FROZEN_SIGMA)})
            continue
        trend = trend_median(case, win, win_x, i, x_target)
        if trend is None:
            rows.append({"id": trophy["id"], "q": quantiles(median_c, sigma_c)})
            continue
        # Hybrid median: the logit-average of the two structural bets.
        median = 0.5 * (trend + median_c)
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma_c)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
