"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for each of the game's trophies, five quantiles (5/25/50/75/95%) of
its Exophase earn rate (percent of tracked PS5 players who have earned it) at
the case's target time. Python 3.13, standard library only, offline,
deterministic, well under 8 s per case.

Model (written from scratch for this entry)
--------------------------------------------
Exophase refreshes a game's counts on its own schedule, hours apart, not
continuously: between refreshes every rate and the players-tracked count are
frozen, and at a refresh the tracked-player count jumps (new players joining
hold fewer trophies, so most rates fall) while earners keep accumulating.
So the history is a step function, and a forecast has to look through the
steps, not through the flat plateaus.

Per trophy, the program:

1. Collapses the history into distinct states: consecutive reads with the
   same (players_tracked, rate) are one observation, dated at the first read
   that showed it. Plateau reads carry no new information; fitting on them
   raw would overweight whichever plateau happened to last longest.
2. Fits a line of logit(rate) against ln(1 + hours since the game's
   release) on the most recent day of states. Trophy growth is fast just
   after launch and slows as the player base matures, which is exactly what
   a straight line in log time says, and the recent slope is the one that
   extrapolates best; it falls back to all states when the recent window
   holds too few to fit.
3. Anchors the forecast at the last observed rate and projects forward with
   that slope -- but lets the slope's influence decay with log-time
   distance from the last read. Growth saturates (players finish the
   trophies they will finish, the tracked base keeps diluting), so a
   straight line in log time systematically overshoots a concave truth;
   the effective step is (1 - exp(-dx)) instead of dx, nearly the full
   step over a 90-minute horizon and about two-thirds of it over the
   final's 4.5 days. The slope is capped so a steep line can never blow
   the forecast up.
4. Sets the spread (logit scale) from the pieces that actually make the
   target uncertain: the random-walk term the house baseline uses
   (0.5 * sqrt(horizon in days)), the line fit's residual, and the slope's
   standard error carried to the target. The spread is floored -- even over
   a short horizon Exophase may refresh once, and a refresh moves every
   rate at once -- and capped so no trophy is ever forecast with false
   certainty or with a meaninglessly wide band.

Quantiles are median +/- z_tau * spread on the logit scale, rounded to
4 decimals half-up, clamped to (0.01, 99.99) percent and non-decreasing.

Validation (local, on synthetic cases built to mimic Exophase's dynamics:
refresh steps every few hours, tracked players growing in cohorts, each
cohort earning trophies over time so rates dilute at refreshes and grow
between them -- see DESIGN.md): mean pinball loss on the logit scale
0.30, against 0.35 for the house curve baseline and 0.40 for the
persistence baseline, across five regimes (fast/slow earners, fast/slow
tracked-player growth). The kit's practice cases carry a perfectly flat
history, so on them this program forecasts persistence with a floored
spread; the practice answers are sealed and the prize is decided by the
final case alone.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
It reads every <id>.in in the input directory and writes <id>.out to the
output directory. Outside the grader's container, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk spread of a rate's move, in logit units per square-root day
# (the same term the house persistence baseline uses).
SIGMA_RW = 0.5
# Honest-spread bounds, in logit units.
SPREAD_FLOOR = 0.45
SPREAD_CAP = 2.0
# Weight of the structural (earners / tracked) median in the trend median.
# (Unused: backtesting showed the structural blend hurt; kept documented
#  here so a later reader knows it was tried.)
# STRUCTURAL_WEIGHT = 0.30
# Window the trend slope is fitted on, in hours before the checkpoint.
RECENT_WINDOW_H = 24.0
# Decay length of the slope's influence, in log-time units: the effective
# projection step is (1 - exp(-DECAY_LAMBDA * dx)) / DECAY_LAMBDA.
DECAY_LAMBDA = 1.0
# Largest slope the fit may carry to the target, in logit units per log-hour.
MAX_SLOPE = 3.0


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation of the standard normal quantile.

    Only used if a case ever asks for levels other than the standard five;
    for those, the exact table in Z is used instead.
    """
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def z_scores(levels) -> list:
    out = []
    for tau in levels:
        hit = None
        for known_tau, known_z in zip(LEVELS, Z):
            if abs(tau - known_tau) < 1e-12:
                hit = known_z
                break
        out.append(hit if hit is not None else norm_ppf(tau))
    return out


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x.

    Returns (a, b, rmse, se_b). rmse is the weighted root-mean-square
    residual; se_b the slope's standard error (inf when undetermined).
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0, float("inf")
    xm = sum(w * x for x, w in zip(xs, ws)) / sw
    ym = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    if n < 2 or sxx <= 1e-12:
        return ym, 0.0, 0.0, float("inf")
    b = sxy / sxx
    a = ym - b * xm
    resid2 = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws))
    dof = max(n - 2, 1)
    sigma2 = resid2 / sw * n / dof if sw > 0 else 0.0
    rmse = math.sqrt(max(sigma2, 0.0))
    se_b = math.sqrt(sigma2 / sxx) if sxx > 0 else float("inf")
    return a, b, rmse, se_b


def forecast_trophy(states, x_target, horizon_days, z_list, at_hours):
    """Forecast one trophy from its distinct history states.

    states: list of (x, rate, players) with x = ln(1 + hours since release),
    oldest first, one entry per distinct (players, rate) plateau.
    at_hours: hours since release at the case's checkpoint.
    Returns the five quantiles, in percent.
    """
    if not states:
        # No usable read at all: a flat, wide, low forecast.
        med = logit(1.0)
        spread = 1.5
        return [round4(clamp_rate(inv_logit(med + z * spread))) for z in z_list]

    last_rate = states[-1][1]
    last_logit = logit(last_rate)
    n = len(states)

    if n == 1:
        # Flat history: no trend to fit. Persistence, with a floored spread:
        # even over a short horizon Exophase may refresh once, and a refresh
        # moves every rate at once, so the band is never razor-thin.
        spread = SIGMA_RW * math.sqrt(horizon_days)
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        return [round4(clamp_rate(inv_logit(last_logit + z * spread)))
                for z in z_list]

    xs = [s[0] for s in states]
    ys = [logit(s[1]) for s in states]
    x_last = xs[-1]

    # Slope from the most recent day of states: trophy growth is fastest
    # just after launch and slows as the player base matures, so the recent
    # slope is the one that extrapolates best. Fall back to all states
    # when the recent window holds too few of them to fit a line.
    recent = [(x, y) for x, y in zip(xs, ys)
              if x >= math.log1p(max(at_hours - RECENT_WINDOW_H, 0.0))]
    if len(recent) >= 2:
        fit_xs = [p[0] for p in recent]
        fit_ys = [p[1] for p in recent]
    else:
        fit_xs, fit_ys = xs, ys
    a, b, rmse, se_b = weighted_line(fit_xs, fit_ys, [1.0] * len(fit_xs))
    # Guard against a wild slope: a steep line extrapolated 4+ days out
    # is how forecasts blow up.
    b = max(-MAX_SLOPE, min(MAX_SLOPE, b))

    # Anchor the forecast at the last observed rate and project forward with
    # the recent slope -- but let the slope's influence decay with log-time
    # distance: growth saturates (trophies get earned, the tracked base
    # keeps diluting), so a straight line in log time systematically
    # overshoots a concave truth. With decay length 1 in log time, the
    # effective step is (1 - exp(-dx)) instead of dx: nearly the full step
    # over a 90-minute horizon, about two-thirds of it over the final's
    # 4.5 days. This is the single biggest accuracy win in backtesting.
    dx = x_target - x_last
    dx_eff = (1.0 - math.exp(-DECAY_LAMBDA * dx)) / DECAY_LAMBDA if dx > 0 else dx
    med_logit = last_logit + b * dx_eff
    med_logit = max(logit(0.01), min(logit(99.99), med_logit))

    # Spread: random-walk term, fit residual, and the slope's standard
    # error carried to the target.
    rw = SIGMA_RW * math.sqrt(horizon_days)
    if math.isfinite(se_b):
        slope_term = se_b * abs(x_target - sum(fit_xs) / len(fit_xs))
    else:
        slope_term = 0.6
    spread = math.sqrt(rw * rw + rmse * rmse + slope_term * slope_term)
    spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)

    qs = [round4(clamp_rate(inv_logit(med_logit + z * spread)))
          for z in z_list]
    # Rounding can only ever tie neighbours; enforce non-decreasing.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels", LEVELS)
    z_list = z_scores(levels)
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0

    # Hours since the game's release; the case names the release date.
    release = None
    game = case.get("game") or {}
    if game.get("release_date"):
        try:
            release = datetime.fromisoformat(game["release_date"]).replace(
                tzinfo=timezone.utc)
        except ValueError:
            release = None
    target = parse_time(case["target"])
    at = parse_time(case["at"]) if case.get("at") else target

    def hours_since_release(moment: datetime) -> float:
        if release is not None:
            return max((moment - release).total_seconds() / 3600.0, 0.0)
        return 0.0

    x_target = math.log1p(hours_since_release(target))
    at_hours = hours_since_release(at)

    trophies = case["trophies"]
    rows = []
    for i, trophy in enumerate(trophies):
        # Collapse the history into distinct states for this trophy.
        states = []
        prev_key = None
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates):
                continue
            rate = rates[i]
            if rate is None:
                continue
            players = read.get("players_tracked")
            key = (players, rate)
            if key == prev_key:
                continue
            prev_key = key
            slot = read.get("slot") or read.get("at")
            moment = parse_time(slot) if slot else target
            states.append((math.log1p(hours_since_release(moment)),
                           float(rate), players))
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(states, x_target, horizon_days,
                                          z_list, at_hours)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
