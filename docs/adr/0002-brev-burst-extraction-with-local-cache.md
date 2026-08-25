# Burst GPU rental on Brev with a local feature cache

Rent Brev instances for feature extraction. Run probe training and analysis on the M4
Max against cached features. Extract each model, Stage set, and resolution once into
resumable shards that every later experiment can reuse.

The planning ceiling is about $150. Drop 896² extraction before dropping models if cost
passes that limit. University clusters add queue delays, while BF16 extraction of six
Towers would take weeks on MPS. Neither is suitable for the main extraction runs.
