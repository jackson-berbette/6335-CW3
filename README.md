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


# Task 4b: Slots Required by Each Storage Policy

`Task4b.py` sizes the policies selected in 4a (Product-Dedicated-COL, Product-Class-COL, Product-Shared-COL, Duration-of-Stay-Target, with Today as the reference). Each policy is sized twice: with tall openings (> 72 in) protected for tall pallets, and with heights mixed. It reads the Data files and `Output/sku_velocity_and_stock.csv` (Task 1f). Run it from the casework folder: `python Task4b.py` (needs pandas, numpy, matplotlib, IPython).

## Method

1. **Stock per pallet.** Every pallet of the year (opening stock plus receipts) is in stock at the end of day *d* when it was received on or before *d* and ships after *d*. This gives the same daily stock as Task 1: peak 6,260 and average 5,540. Both are asserted.
2. **Zones.** Classes are the Phase 2 reporting classes: A is the first 50% of pallets shipped, B the next 30%, C the rest. Duration-of-stay targets use each SKU's expected stay, average stock ÷ pallets shipped per day. That is the only stay information known at put-away, and `sku_zones.csv` can only express it per SKU. The targets are T1 ≤ 14 d, T2 14–30 d, T3 30–60 d and T4 > 60 d.
3. **Slots required.** Each zone's requirement is its own peak, because a zone cannot borrow another zone's empty slots. Dedicated storage needs the sum of the SKU peaks, class storage the sum of the class peaks, and shared storage the peak of the total.
4. **Height feasibility.** A pallet of height *h* fits any slot with `max_pallet_in` ≥ *h* (Rule 5). So the check is nested: for every pallet height *h*, the slots that take *h* must be at least the pallets of height ≥ *h* held at the zones' peaks. Protecting tall openings splits this into two pools. Pallets ≤ 72 in go only in openings ≤ 72 in (4,954 slots), and pallets > 72 in only in tall openings (3,305 slots). The margin is the smallest spare over all heights. The 106 slots that are too low for any pallet are excluded, which leaves 8,259 usable slots.
5. **Zone footprint.** Zones are cut nearest-first, in order of slots ranked by `dist_dock_ft` and then by level. Each zone extends until it holds enough slots of every height. The farthest slot of each zone feeds the travel estimate in 4c.

## Results (`Output/Task4b_Feasibility.csv`)

| Policy | Tall openings | Slots required | ≤ 48 in | 49–72 in | > 72 in | Margin (slots) | Feasible |
|---|---|---:|---:|---:|---:|---:|---|
| Today (reference) | mixed | 6,260 | 2,528 | 1,942 | 1,907 | 1,398 | yes |
| Product-Dedicated-COL | protected | 9,747 | 3,780 | 3,050 | 2,917 | −1,876 | **no** |
| Product-Dedicated-COL | mixed | 9,747 | 3,780 | 3,050 | 2,917 | −1,488 | **no** |
| Product-Class-COL | protected | 6,440 | 2,579 | 2,069 | 1,995 | 353 | yes |
| Product-Class-COL | mixed | 6,440 | 2,579 | 2,069 | 1,995 | 1,310 | yes |
| Product-Shared-COL | protected | 6,260 | 2,528 | 1,942 | 1,907 | 508 | yes |
| Product-Shared-COL | mixed | 6,260 | 2,528 | 1,942 | 1,907 | 1,398 | yes |
| Duration-of-Stay-Target | protected | 6,447 | 2,632 | 2,003 | 2,100 | 427 | yes |
| Duration-of-Stay-Target | mixed | 6,447 | 2,632 | 2,003 | 2,100 | 1,205 | yes |

The height columns are the peak of each height group, summed over zones. Each group peaks on its own day, so the columns add to more than the total. Zone-level detail (peak pallets and footprint slots by height, and distance band) is in `Output/Task4b_Slots_By_Zone_Height.csv`. The figure is `Output/Task4b_Slots_Required.png`.

## Key findings

- **Dedicated is infeasible.** The sum of SKU peaks (9,747) exceeds the 8,259 usable slots. Tall SKUs alone would fit (2,917 of 3,305 tall openings), but pallets ≤ 72 in would need 6,830 of the 4,954 slots that are not tall. Sizing each SKU below its own peak makes it fit only with a shared overflow, as `Output/Task4b_Dedicated_Sensitivity.csv` shows. At the 85th percentile of each SKU's daily stock, the dedicated slots fit with tall openings protected (6,988), but on 6.5% of SKU-days the SKU's stock exceeds its slots, with up to 404 overflow pallets at once.
- **Zoning costs slots.** Class zones need 180 more slots than shared storage (6,440 vs 6,260), and duration-of-stay targets need 187 more. That is the cost of zones peaking at different times.
- **Protecting tall openings is what makes space tight.** With protection, the binding pool is pallets ≤ 72 in in openings ≤ 72 in. Margins drop to 353 (class), 427 (duration of stay) and 508 (shared) slots, about 5–8% of need. Meanwhile about 1,200–1,400 tall openings stay empty at peak. With heights mixed, the binding pool is the tall openings and the margin is 1,200–1,400 slots. Today, on the peak day (15 July 2025), 812 pallets ≤ 72 in sit in tall openings (`Output/Task4b_Today_Peak_Day_Fit.csv`). Partial protection, which reserves only part of the tall openings, sits between the two cases and is worth testing in Task 5.
- **Class A needs reach.** To hold its 858 tall pallets, the class A zone has to stretch to 233 ft and take 2,045 slots, against a peak of 1,844 pallets.
