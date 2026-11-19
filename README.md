# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Drssptra/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Drssptra/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A saturating log-time trend forecaster for GTA VI trophy earn rates on Exophase. For each of the game's 55 trophies it forecasts five quantiles (5/25/50/75/95%) of the trophy's earn rate at 2026-11-26 00:00 UTC, from the house's recorded reads up to the deadline. It collapses each trophy's history into distinct refresh states, fits the recent day's logit-rate slope in log time, anchors at the last observed rate, and lets the slope's influence decay with distance so the 4.5-day forecast doesn't overshoot a saturating curve. Stdlib-only Python, offline and deterministic. See DESIGN.md for the model and its local validation.

## Run it

```sh
python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=hyz23m3g&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
