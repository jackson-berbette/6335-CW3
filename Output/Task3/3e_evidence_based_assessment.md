# Task 3(e): Evidence for the performance judgment

This is an analytical draft; review it against your team’s interpretation before submission.

- Observed average daily closing slot occupancy: 66.23%; peak 74.84%.
- Annual slot travel: 113,349,196 round-trip feet; slot labor: 13,264.05 hours.
- Fastest 60 SKUs: 50.91% of slot travel.
- Relaxed (non-feasible) benchmark: 40.63% travel and 25.51% slot labor.
- Capacity-aware greedy scenario: 21.87% travel and 11.37% slot labor; this is a scenario, not a guarantee.

## Inherent conditions
Historical demand, pallet dimensions, facility geometry, fixed dock locations, and opening inventory constrain the possible outcome.

## Controllable choices
Which vacant compatible slot receives each incoming pallet can change travel, lift time, and unused height.

## Cautions
Do not interpret occupancy below 100% as proof of surplus *usable* capacity: slot-height compatibility matters.
The greedy counterfactual does not simulate forklift interference, service reliability, or truck departures.
Its end-of-year pallet locations need not match historical current_stock.csv; pallet identities and counts do.
The relaxed benchmark is not necessarily an attainable target, and its labor and travel minima can use different slots.
The scorecard does not prove that client service was poor; it only evaluates metrics supported by the files.
