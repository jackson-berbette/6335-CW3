import pandas as pd
import numpy as np

DATA = 'Data/'
OUT = 'Output/'

slots = pd.read_csv(DATA + 'slots.csv')
skus = pd.read_csv(DATA + 'skus.csv')
inbound = pd.read_csv(DATA + 'inbound.csv')
outbound = pd.read_csv(DATA + 'outbound.csv')
opening = pd.read_csv(DATA + 'opening_stock.csv')
velocity = pd.read_csv(OUT + 'Task1b_SKU_Velocity.csv')[['sku_id', 'velocity_class']]  # from Task 1

N_SLOTS = len(slots)
WEEKS = 52

# ---------- One table with every slot move (put-aways + picks) ----------
puts = inbound[['sku_id', 'putaway_slot', 'travel_ft', 'lift_ft']].rename(columns={'putaway_slot': 'slot_id'})
picks = outbound[['sku_id', 'picked_from_slot', 'travel_ft', 'lift_ft']].rename(columns={'picked_from_slot': 'slot_id'})
puts['move'] = 'putaway'
picks['move'] = 'pick'
moves = pd.concat([puts, picks], ignore_index=True)
moves = moves.merge(slots[['slot_id', 'level', 'opening_height_in', 'max_pallet_in', 'cube_ft3']], on='slot_id')
moves = moves.merge(skus[['sku_id', 'pallet_height_in', 'cases_per_pallet', 'case_cube_ft3']], on='sku_id')

# Rule 3: slot move time in minutes
moves['minutes'] = 2 * moves.travel_ft / 264 + 2 * moves.lift_ft / 60 + 1.5
n_moves = len(moves)

# ---------- Travel ----------
avg_travel = moves.travel_ft.mean()
total_miles = 2 * moves.travel_ft.sum() / 5280
avg_lift = moves.lift_ft.mean()
share_level4 = 100 * (moves.level >= 4).mean()

# ---------- Labour ----------
slot_hours_year = moves.minutes.sum() / 60
slot_hours_week = slot_hours_year / WEEKS
min_per_pallet = moves.minutes.sum() / n_moves          # pallets received + shipped
dock_hours_year = 1.8 * n_moves / 60

# ---------- Space: daily stock = opening + receipts - shipments ----------
days = pd.date_range('2025-07-07', '2026-07-05').strftime('%Y-%m-%d')
rec = inbound.groupby('receipt_date').size().reindex(days, fill_value=0)
shp = outbound.groupby('ship_date').size().reindex(days, fill_value=0)
stock = len(opening) + rec.cumsum() - shp.cumsum()       # end of each day
avg_occ = 100 * stock.mean() / N_SLOTS
peak_occ = 100 * stock.max() / N_SLOTS

pa = moves[moves.move == 'putaway']
cube_fill = 100 * (pa.cases_per_pallet * pa.case_cube_ft3).sum() / pa.cube_ft3.sum()

# ---------- Fit ----------
unused_height = (pa.opening_height_in - pa.pallet_height_in).mean()
excess_clearance = (pa.max_pallet_in - pa.pallet_height_in).mean()
share_24in = 100 * ((pa.max_pallet_in - pa.pallet_height_in) >= 24).mean()
short_in_tall = 100 * ((pa.max_pallet_in > 72) & (pa.pallet_height_in <= 72)).sum() / (pa.max_pallet_in > 72).sum()
not_fitting = int((pa.pallet_height_in > pa.max_pallet_in).sum())

# ---------- Inventory: days in stock and FIFO ----------
avg_days = outbound.days_in_stock.mean()
by_class = (outbound.merge(velocity, on='sku_id')
            .groupby('velocity_class').days_in_stock
            .agg(pallets='size', mean='mean', median='median', max='max').round(1))

# FIFO: a pick breaks FIFO if an OLDER pallet of the same SKU was already in stock
# that morning and was still in stock at the end of that day.
allp = pd.concat([
    opening[['pallet_id', 'sku_id', 'received_date']],
    inbound[['pallet_id', 'sku_id', 'receipt_date']].rename(columns={'receipt_date': 'received_date'})])
allp = allp.merge(outbound[['pallet_id', 'ship_date']], on='pallet_id', how='left')
allp['ship_date'] = allp.ship_date.fillna('9999-12-31')   # still in stock at year end
fifo_breaks = 0
for sku, g in allp.groupby('sku_id'):
    recv = g.received_date.to_numpy()
    out = g.ship_date.to_numpy()
    shipped = out != '9999-12-31'
    for r, d in zip(recv[shipped], out[shipped]):
        older_still_there = (recv < r) & (recv < d) & (out > d)
        fifo_breaks += older_still_there.any()
fifo = 100 * (1 - fifo_breaks / len(outbound))

# ---------- Results ----------
results = [
    ('Travel', 'Average one-way travel per slot move', avg_travel, 'ft'),
    ('Travel', 'Total round-trip travel per year', total_miles, 'miles'),
    ('Travel', 'Average lift per slot move', avg_lift, 'ft'),
    ('Travel', 'Share of moves to level 4 or above', share_level4, '%'),
    ('Labour', 'Slot-move labour (Rule 4)', slot_hours_week, 'h/week'),
    ('Labour', 'Slot-move labour (Rule 4)', slot_hours_year, 'h/year'),
    ('Labour', 'Minutes per handled pallet', min_per_pallet, 'min'),
    ('Labour', 'Dock-move hours', dock_hours_year, 'h/year'),
    ('Space', 'Average slot occupancy', avg_occ, '%'),
    ('Space', 'Peak slot occupancy', peak_occ, '%'),
    ('Space', 'Average stock', stock.mean(), 'pallets'),
    ('Space', 'Peak stock', stock.max(), 'pallets'),
    ('Space', 'Cube fill of occupied slots', cube_fill, '%'),
    ('Fit', 'Unused height above pallet', unused_height, 'in'),
    ('Fit', 'Unused height beyond 4 in clearance', excess_clearance, 'in'),
    ('Fit', 'Put-aways with >= 24 in spare', share_24in, '%'),
    ('Fit', 'Short pallets (<= 72 in) in tall slots (> 72 in)', short_in_tall, '% of tall-slot put-aways'),
    ('Fit', 'Pallets not fitting their slot', not_fitting, 'pallets'),
    ('Inventory', 'Average days in stock', avg_days, 'days'),
    ('Inventory', 'FIFO compliance', fifo, '% of picks'),
]
table = pd.DataFrame(results, columns=['area', 'measure', 'value', 'unit'])
table['value'] = table.value.round(2)
# table.to_csv(OUT + 'Task3b_Baseline.csv', index=False)
# by_class.to_csv(OUT + 'Task3b_Days_In_Stock_By_Class.csv')

print(table.to_string(index=False))
print('\nDays in stock by velocity class:\n', by_class)
