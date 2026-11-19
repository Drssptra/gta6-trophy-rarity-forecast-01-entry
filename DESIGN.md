# Design note — gta6-trophy-rarity-forecast-01

## The problem

Forecast, for each of GTA VI's 55 trophies, five quantiles of its Exophase
earn rate (percent of tracked PS5 players who have earned it) at
2026-11-26 00:00 UTC, from every Exophase read the house made before the
deadline (2026-11-21 11:30 UTC). Scored by mean pinball loss on the logit
scale; a paid place needs a loss strictly below the best house entry
(persistence baseline, fitted-curve baseline, house AI model).

## What the data does

Exophase refreshes a game's counts on its own schedule, hours apart, not
continuously: between refreshes every rate and the players-tracked count
are frozen, and at a refresh the tracked count jumps (new players hold
fewer trophies, so most rates fall) while earners keep accumulating. The
kit's practice cases show this directly — flat launch-night reads, then a
refresh to 3,868 players tracked where every rate but the 100% one fell.
So the history is a step function, and a forecast has to look through the
steps, not the plateaus.

## The model (`main.py`, stdlib only, offline, deterministic)

1. **Collapse to distinct states.** Consecutive reads with the same
   (players_tracked, rate) are one observation, dated at the first read
   that showed it. Plateau reads carry no new information.
2. **Recent slope in log time.** A least-squares line of logit(rate)
   against ln(1 + hours since release), fitted on the most recent day of
   states (all states if the window holds too few). Growth is fast just
   after launch and slows as the player base matures; the recent slope is
   the one that extrapolates best. The slope is capped at 3 logit units
   per log-hour.
3. **Anchor at the last rate, decay the slope.** The forecast is the last
   observed rate projected forward with the recent slope, but the slope's
   influence decays with log-time distance: the effective step is
   (1 − exp(−dx)) instead of dx. Growth saturates, so a straight line in
   log time systematically overshoots a concave truth; the decay is
   nearly a no-op over a 90-minute horizon and takes about a third off
   the step over the final's 4.5 days.
4. **Honest spread.** Spread (logit scale) combines the random-walk term
   (0.5·√horizon in days), the fit's residual, and the slope's standard
   error carried to the target; floored at 0.45 (a refresh can move every
   rate at once, even over a short horizon) and capped at 2.0.

Quantiles are median ± z·spread on the logit scale, rounded to 4 decimals
half-up, clamped to (0.01, 99.99)% and non-decreasing.

## Things tried and dropped

- A **structural earners/tracked decomposition** (fit ln E and ln N in log
  time, rate = 100·E/N) blended in at 30%: hurt in backtesting, dropped.
- A **saturating ceiling fit** (profile the asymptote L, regress
  logit(rate/L) on log time): barely beat the house curve baseline and was
  unstable with ~17 noisy states, dropped in favour of the decayed slope.
- Blending the trend toward persistence and shrinking the slope by state
  count: both dragged the median toward no-growth and lost to the house
  curve baseline, dropped.

## Local validation

The kit's practice cases carry a perfectly flat history (zero movement
across all reads), so no trend can be fitted from them and the practice
answers are sealed; the prize is decided by the final case alone. So the
model was validated on synthetic cases built to mimic Exophase's
dynamics: refresh steps every 2–5 h, tracked players growing in cohorts,
each cohort earning trophies over time following a lognormal
time-to-earn, so rates dilute at refreshes and grow between them, printed
to 2 decimals like the page. Checkpoint 2026-11-21 11:30 UTC, target
2026-11-26 00:00 UTC — the final's own geometry.

Mean pinball loss on the logit scale (8 seeds, baseline regime):

| forecaster | loss |
|---|---|
| this entry | **0.311** |
| house curve baseline (reimplemented) | 0.371 |
| persistence baseline (kit starter) | 0.402 |

It beat both baselines on every seed, and the same held across four more
regimes (fast/slow earners, fast/slow tracked-player growth): this entry
0.225–0.294 vs curve 0.275–0.361 vs persistence 0.303–0.446.

Format checks on the kit practice cases: exit 0, 55 rows, ids 1–55 in
order, five finite quantiles per row in (0, 100] and non-decreasing,
2,677 bytes per output (limit 65,536), two runs byte-identical, ~0.05 s
per case (limit 8 s), and a grader-style run (`python3 -P -s`) passes.
Edge cases (null rates, missing players_tracked, rate 0 / 100, one-read
and empty history) all produce valid output.
