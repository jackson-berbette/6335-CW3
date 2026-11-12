# StackBay Warehousing Storage Design — Phase 1
## Combined README for Tasks 1, 2 and 3

**Casework:** ISyE 6202 / ISyE 6335 — Casework 3, StackBay Warehousing Storage Design, Phase 1 (Fall 2026)  
**Purpose:** Reproduce the historical inventory, warehouse operations diagnostic, and baseline performance analyses using the supplied case data.

## Requirements

- Python 3.10 or newer
- pandas and numpy

```bash
python -m pip install pandas numpy
```

## Folder layout

Run the script from the casework folder, which must contain:

```text
6335-CW3/
├── Task1.py
├── Task2.py
├── Task3b.py
├── Data/
│   ├── slots.csv
│   ├── skus.csv
│   ├── inbound.csv
│   ├── outbound.csv
│   └── opening_stock.csv
└── Output/
    └── Task1b_SKU_Velocity.csv     (written by Task1.py)
```


# Task 3b: Baseline Performance Measures

`Task3b.py` computes the 10 performance measures defined in Task 3a for the historical year as operated (7 July 2025 to 5 July 2026). It reads the casework data files and the velocity classes from Task 1. It does not depend on any other Task 3 script.


## What the script does

### 1. Build the list of slot moves

Every put-away (`inbound.csv`) and every pick (`outbound.csv`) is one slot move. That gives 200,896 moves: 100,245 put-aways and 100,651 picks. Each move is joined to its slot (`slots.csv`: level, opening height, max pallet height, cube) and to its SKU (`skus.csv`: pallet height, cases per pallet, case cube).

The time for each move comes from modeling rule 3:

```text
minutes = 2 × travel_ft / 264 + 2 × lift_ft / 60 + 1.5
```

### 2. Compute the measures

| Area | Measure | Computation |
|---|---|---|
| Travel | Average one-way travel | mean of `travel_ft` over all slot moves |
| Travel | Total travel per year | 2 × sum of `travel_ft` ÷ 5,280, in round-trip miles |
| Travel | Average lift | mean of `lift_ft` over all slot moves |
| Travel | Moves to level 4 or above | share of moves whose slot has `level` ≥ 4 |
| Labour | Slot-move labour (rule 4) | sum of move minutes ÷ 60, per year and ÷ 52 per week |
| Labour | Minutes per handled pallet | sum of move minutes ÷ number of slot moves (pallets received + shipped) |
| Labour | Dock-move hours | 1.8 min × number of slot moves ÷ 60 |
| Space | Average and peak slot occupancy | end-of-day stock = opening stock + cumulative receipts − cumulative shipments, divided by 8,365 slots |
| Space | Cube fill of occupied slots | sum of (`cases_per_pallet` × `case_cube_ft3`) ÷ sum of slot `cube_ft3`, over put-aways |
| Fit | Unused height | mean of (`opening_height_in` − `pallet_height_in`) over put-aways |
| Fit | Unused height beyond the clearance | mean of (`max_pallet_in` − `pallet_height_in`) over put-aways |
| Fit | Put-aways with ≥ 24 in spare | share of put-aways where `max_pallet_in` − `pallet_height_in` ≥ 24 |
| Fit | Short pallets in tall slots | share of put-aways into slots with `max_pallet_in` > 72 in that were pallets ≤ 72 in |
| Fit | Pallets not fitting | count of put-aways with `pallet_height_in` > `max_pallet_in` (rule 5) |
| Inventory | Days in stock | mean of `days_in_stock` in `outbound.csv`, overall and by velocity class |
| Inventory | FIFO compliance | share of picks with no older pallet left behind (see below) |

### 3. FIFO check

FIFO is checked at day level. A pick breaks FIFO if another pallet of the same SKU:

- was received earlier than the picked pallet,
- was already in stock on the morning of the pick (received before the ship date), and
- was still in stock at the end of that day (shipped later, or never).

Pallets come from `opening_stock.csv` and `inbound.csv`, with their ship dates from `outbound.csv`. This check does not look at the order of picks within one day. Task 2(c) audits that order separately.

## Results (historical year)

| Area | Measure | Value |
|---|---|---|
| Travel | Average one-way travel per slot move | 282.1 ft |
| Travel | Total round-trip travel | 21,468 miles/yr |
| Travel | Average lift / moves to level ≥ 4 | 9.7 ft / 29.2% |
| Labour | Slot-move labour (rule 4) | 255.1 h/week (13,264 h/yr) |
| Labour | Minutes per handled pallet | 3.96 min |
| Labour | Dock-move hours | 6,027 h/yr |
| Space | Occupancy, average / peak | 66.2% / 74.8% (5,540 / 6,260 pallets) |
| Space | Cube fill of occupied slots | 66.9% |
| Fit | Unused height / beyond the 4 in clearance | 21.1 in / 17.2 in |
| Fit | Put-aways with ≥ 24 in spare | 25.1% |
| Fit | Short pallets (≤ 72 in) in tall slots | 25.6% of tall-slot put-aways |
| Fit | Pallets not fitting their slot | 0 |
| Inventory | Average days in stock | 20.2 days (A 11.5, B 22.4, C 37.8, D 120.4) |
| Inventory | FIFO compliance | 100% at day level |

## Assumptions and limits

- **Year:** 52 weeks, used for the weekly labour figures.
- **Velocity classes:** taken as they are from Task 1 (`Output/Task1b_SKU_Velocity.csv`).
- **Daily stock:** counted at the end of each day, after the day's picks and the night's receipts.
- **Tall cut-off:** the 72 in threshold for "tall" pallets and slots is our own choice. Change it in the Fit section of the script to test other values.
- **Cube fill:** uses the product cube (cases × case cube), not the full pallet envelope. It is measured over put-aways.
- **Labour hours:** move time only, from the modeling rules. They include no allowance for breaks, waiting or congestion. Task 2(d) covers those with the Digital Twin.
