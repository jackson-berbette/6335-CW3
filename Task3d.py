import pandas as pd
import numpy as np


slots = pd.read_csv('Data/slots.csv').set_index('slot_id', drop=False)
skus = pd.read_csv('Data/skus.csv').set_index('sku_id')
velocity = pd.read_csv('Output/Task1b_SKU_Velocity.csv')
opening = pd.read_csv('Data/opening_stock.csv')
inbound = pd.read_csv('Data/inbound.csv')
outbound = pd.read_csv('Data/outbound.csv')
stock = pd.read_csv('Output/sku_stock_summary.csv')

# Each SKU gets a preferred zone based on its velocity rank and its average
# inventory footprint. Zone boundaries are fixed; positions are shared.
priority = velocity[['sku_id','pallets_yr']].merge(stock[['sku_id','avg_stock']],on='sku_id')
priority = priority.sort_values(['pallets_yr','sku_id'], ascending=[False,True]).reset_index(drop=True)
priority['cum_avg_stock'] = priority.avg_stock.cumsum() - priority.avg_stock/2
priority['footprint_share'] = priority.cum_avg_stock / priority.avg_stock.sum()
priority['preferred_zone'] = np.select([priority.footprint_share < .30, priority.footprint_share < .65], [0,1],default=2)
zone_of_sku = priority.set_index('sku_id').preferred_zone.to_dict()
velocity_of_sku = priority.set_index('sku_id').pallets_yr.to_dict()

slots = slots.reset_index(drop=True).sort_values(['dist_dock_ft','elev_ft','slot_id']).set_index('slot_id',drop=False).copy()
n = len(slots)
slots['zone'] = np.where(np.arange(n)<int(n*.30),0,np.where(np.arange(n)<int(n*.65),1,2))
# Sorted per-zone compatible slot candidate arrays. Prefer lower elevation on ties.
slot_ids = slots.slot_id.to_numpy()
slot_height = slots.max_pallet_in.to_dict()
slot_distance = slots.dist_dock_ft.to_dict()
slot_elev = slots.elev_ft.to_dict()
slot_level = slots.level.to_dict()
pallet_height = skus.pallet_height_in.to_dict()
slot_zone = slots.zone.to_dict()

# Prefer the designated zone. Overflow to neighboring zones; fast SKUs
# prefer near overflow, slow SKUs prefer far overflow.
zone_order = {0:[0,1,2], 1:[1,0,2], 2:[2,1,0]}
zone_slots = {z:slots.loc[slots.zone==z,'slot_id'].tolist() for z in (0,1,2)}
free = set(slot_ids)
occupied = {}  # pallet_id -> slot_id

def allocate(pallet, sku):
    height = pallet_height[sku]
    for z in zone_order[zone_of_sku[sku]]:
        # For high-velocity SKUs, choose closest slot in the preferred zone.
        # For low-velocity SKUs, choose farther slots first within their zone,
        # preserving the closest slots for faster products.
        candidates = zone_slots[z] if zone_of_sku[sku] != 2 else reversed(zone_slots[z])
        for s in candidates:
            if s in free and slot_height[s] >= height:
                free.remove(s)
                occupied[pallet] = s
                return s
    raise RuntimeError(f'No compatible slot for pallet={pallet}, sku={sku}')

# Initial redistribution: optimistic one-time setup; relocation cost excluded.
opening = opening.merge(priority[['sku_id','pallets_yr']],on='sku_id')
opening = opening.sort_values(['pallets_yr','sku_id','pallet_id'],ascending=[False,True,True])
for row in opening.itertuples(index=False):
    allocate(row.pallet_id,row.sku_id)

# Historical travel distances include door/path-specific effects. Preserve
# each move's original travel and adjust by the difference in slot distance.
# Do the analogous adjustment for lift elevation.

def record(row, new_slot, old_slot, kind):
    travel = max(0.,float(row.travel_ft) + slot_distance[new_slot] - slot_distance[old_slot])
    lift = max(0.,float(row.lift_ft) + slot_elev[new_slot] - slot_elev[old_slot])
    return (kind,row.sku_id,row.pallet_id,old_slot,new_slot,travel,lift,slot_level[new_slot])

outbound['ship_date'] = pd.to_datetime(outbound.ship_date)
inbound['receipt_date'] = pd.to_datetime(inbound.receipt_date)
outbound = outbound.sort_values(['ship_date','picked_at','pallet_id'])
inbound = inbound.sort_values(['receipt_date','arrived_at','pallet_id'])
outs_by_day = {day:g for day,g in outbound.groupby('ship_date')}
ins_by_day = {day:g for day,g in inbound.groupby('receipt_date')}
records = []
peak_occupied = len(occupied)
all_dates = pd.date_range(min(outbound.ship_date.min(),inbound.receipt_date.min()),max(outbound.ship_date.max(),inbound.receipt_date.max()))
for day in all_dates:
    # Day shift: all picks before any receiving on the same calendar day.
    if day in outs_by_day:
        for row in outs_by_day[day].itertuples(index=False):
            if row.pallet_id not in occupied:
                raise RuntimeError(f'Pallet not present for outbound pick: {row.pallet_id} on {day.date()}')
            new_slot = occupied.pop(row.pallet_id)
            free.add(new_slot)
            records.append(record(row,new_slot,row.picked_from_slot,'pick'))
    # Night shift: prioritize fastest SKUs among receipts for that night.
    if day in ins_by_day:
        arrivals = ins_by_day[day].copy()
        arrivals['velocity'] = arrivals.sku_id.map(velocity_of_sku)
        arrivals = arrivals.sort_values(['velocity','arrived_at','pallet_id'],ascending=[False,True,True])
        for row in arrivals.itertuples(index=False):
            new_slot = allocate(row.pallet_id,row.sku_id)
            records.append(record(row,new_slot,row.putaway_slot,'putaway'))
    peak_occupied = max(peak_occupied,len(occupied))

moves = pd.DataFrame(records,columns=['move','sku_id','pallet_id','old_slot','new_slot','travel_ft','lift_ft','level'])
# Same formula and number of moves as Task 3b.
moves['minutes'] = 2*moves.travel_ft/264 + 2*moves.lift_ft/60 + 1.5
historical = pd.concat([
    inbound[['travel_ft','lift_ft','putaway_slot']].rename(columns={'putaway_slot':'slot_id'}),
    outbound[['travel_ft','lift_ft','picked_from_slot']].rename(columns={'picked_from_slot':'slot_id'})
],ignore_index=True).merge(slots[['level']],left_on='slot_id',right_index=True,validate='many_to_one')
historical['minutes'] = 2*historical.travel_ft/264+2*historical.lift_ft/60+1.5

def metrics(df):
    return pd.Series({
        'avg_one_way_ft':df.travel_ft.mean(),
        'round_trip_miles':2*df.travel_ft.sum()/5280,
        'avg_lift_ft':df.lift_ft.mean(),
        'pct_level_4_plus':100*(df.level>=4).mean(),
        'slot_hours_year':df.minutes.sum()/60,
        'slot_hours_week':df.minutes.sum()/60/52,
        'minutes_per_pallet':df.minutes.mean(),
        'dock_hours_year':1.8*len(df)/60
    })

comparison = pd.DataFrame({'historical':metrics(historical),'zoned':metrics(moves)})
comparison['savings'] = comparison.historical-comparison.zoned
comparison['savings_pct'] = np.where(comparison.historical!=0,100*comparison.savings/comparison.historical,np.nan)
print(comparison.round(2).to_string())