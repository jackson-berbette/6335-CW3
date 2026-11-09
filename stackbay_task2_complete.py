#!/usr/bin/env python3
#run in vs code : python -m pip install -r requirements.txt
#run in vs code : python task2_analysis.py --data input_data --out outputs --velocity-file task1_velocity_stock.csv
"""StackBay Casework 3 Phase 1 Task 2(a-d): reproducible operations diagnostic.
Run: python task2_analysis.py --data input_data --out outputs
Digital Twin experiments must be performed manually; no simulated results are invented. Includes two FIFO audits and sampled feasible-slot alternatives.
"""
import argparse
import heapq
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REQUIRED = ['slots','docks','skus','clients','opening_stock','inbound','outbound','current_stock']
START = pd.Timestamp('2025-07-07')
END = pd.Timestamp('2026-07-05')


def check(condition, message):
    if not bool(condition):
        raise ValueError(message)


def save(df, folder, name):
    df.to_csv(folder / f'{name}.csv', index=False)


def plot(folder, name):
    plt.tight_layout()
    plt.savefig(folder / f'{name}.png', dpi=175, bbox_inches='tight')
    plt.close()


def load(data):
    d = {}
    for name in REQUIRED:
        path = data / (name + '.csv')
        check(path.exists(), f'Missing input file: {path}')
        d[name] = pd.read_csv(path)
    return d


def velocity_classes(skus, outbound):
    """EXACT Task 1(b) rule: prior cumulative share <80% A; <95% B; else C."""
    counts = outbound.groupby('sku_id').size().rename('pallets_shipped')
    v = skus[['sku_id','client_id','pallet_height_in','pallet_weight_lb']].merge(
        counts, left_on='sku_id', right_index=True, how='left')
    v['pallets_shipped'] = v['pallets_shipped'].fillna(0).astype(int)
    v = v.sort_values(['pallets_shipped','sku_id'], ascending=[False,True]).reset_index(drop=True)
    prior = (v['pallets_shipped'].cumsum() - v['pallets_shipped']) / len(outbound)
    v['velocity_class'] = np.select([prior < .80, prior < .95], ['A','B'], default='C')
    return v


def validate(d):
    s, i, o, opening, current, sku = (d[n] for n in ['slots','inbound','outbound','opening_stock','current_stock','skus'])
    for name in ['slots','skus','opening_stock','inbound','outbound','current_stock']:
        check(not d[name].isna().any().any(), f'Null data in {name}')
    for name, col in [('slots','slot_id'),('skus','sku_id'),('opening_stock','pallet_id'),
                      ('inbound','pallet_id'),('outbound','pallet_id'),('current_stock','pallet_id')]:
        check(d[name][col].is_unique, f'Nonunique {col} in {name}')
    check(len(opening)+len(i)-len(o)==len(current), 'Pallet count reconciliation fails')
    check(set(opening.pallet_id).isdisjoint(set(i.pallet_id)), 'Opening and inbound pallet IDs overlap')
    inventory_ids = (set(opening.pallet_id)|set(i.pallet_id))-set(o.pallet_id)
    check(inventory_ids == set(current.pallet_id), 'Closing pallet identities mismatch')
    check(set(o.pallet_id).issubset(set(opening.pallet_id)|set(i.pallet_id)), 'Unknown outbound pallet')
    check(set(i.putaway_slot).issubset(set(s.slot_id)), 'Unknown inbound slot')
    check(set(o.picked_from_slot).issubset(set(s.slot_id)), 'Unknown outbound slot')
    check(set(opening.slot_id).issubset(set(s.slot_id)), 'Unknown opening slot')
    check(set(current.slot_id).issubset(set(s.slot_id)), 'Unknown current slot')
    check(set(i.sku_id).issubset(set(sku.sku_id)) and set(o.sku_id).issubset(set(sku.sku_id)), 'Unknown SKU')
    for name, col in [('inbound','receipt_date'),('outbound','ship_date')]:
        dates=pd.to_datetime(d[name][col]);check(dates.between(START,END).all(),f'{name} outside history')
    check((s.max_pallet_in == s.opening_height_in-4).all(), 'Slot clearance inconsistent')
    for label, df, col in [('opening',opening,'slot_id'),('inbound',i,'putaway_slot'),('outbound',o,'picked_from_slot'),('current',current,'slot_id')]:
        j=df[['sku_id',col]].merge(s[['slot_id','max_pallet_in']],left_on=col,right_on='slot_id').merge(sku[['sku_id','pallet_height_in']],on='sku_id')
        check((j.pallet_height_in<=j.max_pallet_in).all(), f'{label} includes non-fitting pallet')
    # Verify individual pallets retain their SKU and physical slot over time.
    origin=pd.concat([opening[['pallet_id','sku_id','slot_id']].assign(source='opening'),
                      i[['pallet_id','sku_id','putaway_slot']].rename(columns={'putaway_slot':'slot_id'}).assign(source='inbound')])
    for label, frame, col in [('outbound',o,'picked_from_slot'),('current',current,'slot_id')]:
        merged=frame[['pallet_id','sku_id',col]].merge(origin[['pallet_id','sku_id','slot_id']],on='pallet_id',suffixes=('_observed','_original'),validate='many_to_one')
        check(len(merged)==len(frame),f'{label}: unmatched pallet IDs')
        check((merged.sku_id_observed==merged.sku_id_original).all(),f'{label}: SKU mismatch')
        check((merged[col if col != 'slot_id' else 'slot_id_observed']==merged['slot_id_original' if col == 'slot_id' else 'slot_id']).all(),f'{label}: pallet slot mismatch')
    return {'slots':len(s),'opening_pallets':len(opening),'receipts':len(i),'shipments':len(o),'closing_pallets':len(current),'sku_count':len(sku)}


def task2a(d,out):
    s=d['slots'].copy(); docks=d['docks']
    levels=s.groupby('level').agg(slots=('slot_id','size'),aisles=('aisle','nunique'),mean_opening_in=('opening_height_in','mean'),mean_distance_ft=('dist_dock_ft','mean')).reset_index()
    save(levels,out,'2a_slots_by_level')
    aisles=s.groupby('aisle').agg(slots=('slot_id','size'),bays=('bay','nunique'),max_level=('level','max'),mean_distance_ft=('dist_dock_ft','mean'),tall_openings=('opening_height_in',lambda x:(x>=90).sum())).reset_index()
    save(aisles,out,'2a_aisle_layout')
    openings=s.groupby('opening_height_in').size().rename('slots').reset_index();save(openings,out,'2a_opening_heights')
    # Explicit coordinate path check: dock apron + horizontal front-aisle + aisle depth.
    # Facility map uses dock y=-40, and slot y is distance from front aisle; front aisle y=0.
    check(np.allclose(docks.y_ft,-40), 'Dock y coordinates differ from assumed 40-ft apron')
    expected=np.abs(s.aisle_x_ft.to_numpy()[:,None]-docks.x_ft.to_numpy()[None,:]).mean(axis=1)+40+s.y_ft.to_numpy()
    errors=s.dist_dock_ft.to_numpy()-expected
    verification=pd.DataFrame({'slot_id':s.slot_id,'provided_avg_dist_ft':s.dist_dock_ft,'computed_avg_dist_ft':expected,'difference_ft':errors})
    save(verification,out,'2a_distance_verification')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(levels.level.astype(str),levels.slots);ax.set(xlabel='Rack level',ylabel='Number of pallet slots',title='2(a) Rack slots by level');plot(out,'2a_slots_by_level')
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(s.opening_height_in,bins=np.arange(20,136,5),edgecolor='white');ax.set(xlabel='Opening height (in)',ylabel='Slots',title='2(a) Rack opening heights');plot(out,'2a_opening_height_histogram')
    fig,ax=plt.subplots(figsize=(11,5));scatter=ax.scatter(s.x_ft,s.y_ft,c=s.opening_height_in,s=3,cmap='viridis',rasterized=True);fig.colorbar(scatter,ax=ax,label='Opening height (in)');ax.set(xlabel='Warehouse x (ft)',ylabel='Warehouse depth y (ft)',title='2(a) Spatial distribution of rack openings');plot(out,'2a_opening_height_map')
    return {'aisles':int(s.aisle.nunique()),'slots':len(s),'levels':int(s.level.max()),'dock_doors':len(docks),'mean_abs_distance_error_ft':float(np.abs(errors).mean()),'max_abs_distance_error_ft':float(np.abs(errors).max())}


def task2b(d,v,out):
    i=d['inbound'];s=d['slots'];sku=d['skus']
    m=i[['pallet_id','sku_id','receipt_date','putaway_slot','travel_ft','lift_ft']].merge(
        s[['slot_id','aisle','level','opening_height_in','max_pallet_in','dist_dock_ft','elev_ft']],left_on='putaway_slot',right_on='slot_id',validate='many_to_one').merge(
        v[['sku_id','client_id','velocity_class','pallet_height_in','pallets_shipped']],on='sku_id',validate='many_to_one')
    m['unused_height_in']=m.opening_height_in-m.pallet_height_in
    m['excess_above_clearance_in']=m.max_pallet_in-m.pallet_height_in
    check((m.excess_above_clearance_in>=0).all(),'Non-fitting placement')
    by_class=m.groupby('velocity_class').agg(putaways=('pallet_id','size'),avg_travel_ft=('travel_ft','mean'),median_travel_ft=('travel_ft','median'),avg_level=('level','mean'),avg_unused_height_in=('unused_height_in','mean'),avg_excess_clearance_in=('excess_above_clearance_in','mean')).reset_index()
    save(by_class,out,'2b_putaway_by_velocity')
    by_client=m.groupby('client_id').agg(putaways=('pallet_id','size'),avg_travel_ft=('travel_ft','mean'),avg_level=('level','mean'),avg_unused_height_in=('unused_height_in','mean')).reset_index();save(by_client,out,'2b_putaway_by_client')
    client_aisle=pd.crosstab(m.client_id,m.aisle);client_aisle.to_csv(out/'2b_client_aisle_counts.csv')
    shares=client_aisle.div(client_aisle.sum(axis=1),axis=0);shares.to_csv(out/'2b_client_aisle_shares.csv')
    by_level=m.groupby('level').agg(putaways=('pallet_id','size'),avg_travel_ft=('travel_ft','mean'),avg_unused_height_in=('unused_height_in','mean')).reset_index();save(by_level,out,'2b_putaway_by_level')
    # Slot occupancy reconstructed using actual movement records, not independent snapshots.
    # Track daily occupied-slot height slack for a time-weighted measure.
    pallet_height = dict(zip(d['opening_stock'].pallet_id,
                             d['opening_stock'].sku_id.map(sku.set_index('sku_id').pallet_height_in)))
    pallet_height.update(dict(zip(i.pallet_id, i.sku_id.map(sku.set_index('sku_id').pallet_height_in))))
    opening_by_slot = s.set_index('slot_id').opening_height_in.to_dict()
    occupied=dict(zip(d['opening_stock'].slot_id,d['opening_stock'].pallet_id))
    out_by_date={k:g for k,g in d['outbound'].groupby('ship_date',sort=False)}
    in_by_date={k:g for k,g in d['inbound'].groupby('receipt_date',sort=False)}
    occupancy=[]
    daily_height=[]
    for day in pd.date_range(START,END):
        key=day.strftime('%Y-%m-%d')
        for row in out_by_date.get(key,pd.DataFrame(columns=['picked_from_slot','pallet_id'])).itertuples(index=False):
            check(occupied.get(row.picked_from_slot)==row.pallet_id, f'Slot occupancy mismatch on outbound {key}: {row.pallet_id}')
            del occupied[row.picked_from_slot]
        for row in in_by_date.get(key,pd.DataFrame(columns=['putaway_slot','pallet_id'])).itertuples(index=False):
            check(row.putaway_slot not in occupied,f'Putaway to occupied slot on {key}: {row.putaway_slot}')
            occupied[row.putaway_slot]=row.pallet_id
        occupancy.append((key,len(occupied)))
        daily_slack = sum(opening_by_slot[slot]-pallet_height[pid] for slot,pid in occupied.items())
        daily_height.append((key,len(occupied),daily_slack,
                             daily_slack/len(occupied) if occupied else np.nan))
    check(occupied==dict(zip(d['current_stock'].slot_id,d['current_stock'].pallet_id)),'End-of-year slot occupancy mismatch')
    save(pd.DataFrame(occupancy,columns=['date','occupied_slots_end_of_day']),out,'2b_daily_slot_occupancy_validation')
    height_daily = pd.DataFrame(daily_height, columns=['date','occupied_slots','total_unused_opening_height_in','mean_unused_height_in'])
    save(height_daily,out,'2b_daily_occupied_height_slack')
    time_weighted_slack = float(height_daily.total_unused_opening_height_in.sum()/height_daily.occupied_slots.sum())
    # Quantify physical height slack for observed putaways, with and without 4-inch clearance.
    height_hist=m.groupby(pd.cut(m.excess_above_clearance_in,[-.001,10,20,30,40,60,100,200],include_lowest=True),observed=True).size().rename('putaways').reset_index();height_hist.iloc[:,0]=height_hist.iloc[:,0].astype(str);save(height_hist,out,'2b_excess_height_bins')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(by_class.velocity_class,by_class.avg_travel_ft);ax.set(xlabel='Task 1 velocity class',ylabel='Mean putaway travel (ft, one way)',title='2(b) Storage distance by SKU velocity');plot(out,'2b_velocity_distance')
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(by_client.client_id,by_client.avg_travel_ft);ax.set(xlabel='Client',ylabel='Mean putaway travel (ft)',title='2(b) Putaway distance by client');plot(out,'2b_client_distance')
    fig,ax=plt.subplots(figsize=(11,4));ax.imshow(shares.to_numpy(),aspect='auto',cmap='Blues');ax.set_xticks(range(len(shares.columns)),shares.columns,rotation=90,fontsize=7);ax.set_yticks(range(len(shares.index)),shares.index);ax.set(xlabel='Aisle',ylabel='Client',title='2(b) Share of each client’s putaways by aisle');plot(out,'2b_client_aisle_heatmap')
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(m.excess_above_clearance_in,bins=35,edgecolor='white');ax.set(xlabel='Unused height beyond mandatory 4-in clearance (in)',ylabel='Putaway pallets',title='2(b) Vertical space left unused at putaway');plot(out,'2b_unused_height')
    # Actual-door feasible-slot benchmark: sample every 10th putaway, deterministically.
    # Reconstruct occupancy with day picks preceding night receipts. For tied timestamps,
    # pallet_id breaks ties; available-slot comparisons are conditional on this assumption.
    slot_order=s.reset_index(drop=True)
    slot_to_idx={slot:k for k,slot in enumerate(slot_order.slot_id)}
    occupied_mask=np.zeros(len(slot_order),dtype=bool)
    occupied_mask[[slot_to_idx[x] for x in d['opening_stock'].slot_id]]=True
    max_height=slot_order.max_pallet_in.to_numpy()
    aisle=slot_order.aisle.to_numpy()
    opening_height=slot_order.opening_height_in.to_numpy()
    dock_x=d['docks'].set_index('door_id').x_ft.to_dict()
    actual_door_distance={door:40+np.abs(slot_order.aisle_x_ft.to_numpy()-x)+slot_order.y_ft.to_numpy()
                          for door,x in dock_x.items()}
    sku_height=sku.set_index('sku_id').pallet_height_in.to_dict()
    o_days={k:g for k,g in d['outbound'].groupby('ship_date',sort=False)}
    i_days={k:g for k,g in i.groupby('receipt_date',sort=False)}
    samples=[]; movement_number=0
    for day in pd.date_range(START,END):
        key=day.strftime('%Y-%m-%d')
        outs=o_days.get(key)
        if outs is not None:
            for r in outs.itertuples(index=False):
                idx=slot_to_idx[r.picked_from_slot]
                check(occupied_mask[idx],f'Outbound from empty slot {r.picked_from_slot} on {key}')
                occupied_mask[idx]=False
        ins=i_days.get(key)
        if ins is not None:
            for r in ins.sort_values(['putaway_at','pallet_id']).itertuples(index=False):
                idx=slot_to_idx[r.putaway_slot]
                check(not occupied_mask[idx],f'Putaway into occupied slot {r.putaway_slot} on {key}')
                if movement_number % 10 == 0:
                    feasible=(~occupied_mask) & (max_height>=sku_height[r.sku_id])
                    check(feasible[idx],f'Actual putaway is not feasible {r.pallet_id}')
                    distance=actual_door_distance[r.door_id]
                    feasible_dist=distance[feasible]
                    min_dist=float(feasible_dist.min())
                    min_slack=float((opening_height[feasible]-sku_height[r.sku_id]).min())
                    chosen_slack=float(opening_height[idx]-sku_height[r.sku_id])
                    samples.append((r.pallet_id,key,r.sku_id,r.door_id,r.putaway_slot,
                                    int(feasible.sum()),float(distance[idx]),min_dist,
                                    float(distance[idx]-min_dist),
                                    float(np.mean(feasible_dist<=distance[idx])),
                                    chosen_slack,min_slack,chosen_slack-min_slack,
                                    int(np.sum(feasible & (aisle==aisle[idx])))))
                occupied_mask[idx]=True
                movement_number+=1
    check(set(slot_order.slot_id[occupied_mask])==set(d['current_stock'].slot_id),
          'Free-slot reconstruction failed to reconcile closing stock')
    alternatives=pd.DataFrame(samples,columns=['pallet_id','receipt_date','sku_id','door_id','putaway_slot',
        'feasible_free_slots','chosen_actual_door_distance_ft','nearest_feasible_actual_door_distance_ft',
        'extra_distance_vs_nearest_ft','chosen_distance_percentile_among_feasible',
        'chosen_unused_opening_height_in','minimum_feasible_unused_opening_height_in',
        'extra_unused_height_vs_tightest_fit_in','feasible_free_slots_same_aisle'])
    save(alternatives,out,'2b_sampled_feasible_slot_comparison')
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(alternatives.extra_distance_vs_nearest_ft,bins=35,edgecolor='white')
    ax.set(xlabel='Extra one-way distance vs nearest feasible slot from actual door (ft)',
           ylabel='Sampled putaways',title='2(b) Actual choice vs feasible alternatives (sample)')
    plot(out,'2b_feasible_slot_comparison')
    # Correlations are descriptive, not proof of driver intent.
    corr=float(m[['pallets_shipped','travel_ft']].corr(method='spearman').iloc[0,1])
    return {'putaways':len(m),'avg_unused_height_in':float(m.unused_height_in.mean()),'avg_excess_clearance_in':float(m.excess_above_clearance_in.mean()),'pct_excess_clearance_over_20_in':float(100*(m.excess_above_clearance_in>20).mean()),'spearman_sku_volume_vs_putaway_distance':corr,'end_occupancy_validated':True,'time_weighted_avg_unused_opening_height_in':time_weighted_slack,'sampled_putaways_compared':len(alternatives),'sample_mean_extra_distance_ft':float(alternatives.extra_distance_vs_nearest_ft.mean()),'sample_nearest_slot_selection_pct':float(100*np.mean(np.isclose(alternatives.extra_distance_vs_nearest_ft,0))),'availability_sampling_limitation':'Every 10th putaway; same-timestamp putaway order assumed by pallet ID; compare using the actual receiving door'}


def task2c(d,v,out):
    opening=d['opening_stock'];i=d['inbound'];o=d['outbound'].copy();o['picked_time']=pd.to_datetime(o.picked_at)
    # FIFO evaluated by receipt DATE. Equal-date pallets are interchangeable.
    # Since shipping is on the day shift and receiving at night, process all outbound before same-day inbound.
    heaps={sku:[] for sku in d['skus'].sku_id}
    for r in opening.itertuples(index=False):
        heapq.heappush(heaps[r.sku_id],(r.received_date,r.pallet_id))
    inbound_days={key:g for key,g in i.groupby('receipt_date',sort=False)}
    outbound_days={key:g for key,g in o.groupby('ship_date',sort=False)}
    checks=[]
    for day in pd.date_range(START,END):
        key=day.strftime('%Y-%m-%d')
        day_out=outbound_days.get(key)
        if day_out is not None:
            for r in day_out.sort_values(['picked_time','pallet_id']).itertuples(index=False):
                h=heaps[r.sku_id]
                check(bool(h),f'No available pallet for SKU {r.sku_id} at {key}')
                oldest_date=h[0][0]
                compliant=(r.received_date==oldest_date)
                checks.append((r.pallet_id,r.sku_id,r.client_id,key,r.received_date,oldest_date,compliant,int(r.days_in_stock)))
                # Remove actual shipped pallet, even if FIFO violated; heap is kept consistent.
                actual=(r.received_date,r.pallet_id)
                try: idx=h.index(actual)
                except ValueError: raise ValueError(f'Shipped pallet unavailable: {r.pallet_id} on {key}')
                h[idx]=h[-1];h.pop()
                if h: heapq.heapify(h)
        day_in=inbound_days.get(key)
        if day_in is not None:
            for r in day_in.itertuples(index=False):
                heapq.heappush(heaps[r.sku_id],(r.receipt_date,r.pallet_id))
    leftover={p for h in heaps.values() for _,p in h}
    check(leftover==set(d['current_stock'].pallet_id),'FIFO reconstruction closing pallet set differs')
    fifo=pd.DataFrame(checks,columns=['pallet_id','sku_id','client_id','ship_date','actual_received_date','oldest_available_received_date','fifo_compliant','days_in_stock'])
    # Check recorded age against calendar date arithmetic.
    derived=(pd.to_datetime(fifo.ship_date)-pd.to_datetime(fifo.actual_received_date)).dt.days
    check((derived==fifo.days_in_stock).all(),'days_in_stock differs from date subtraction')
    fifo=fifo.merge(v[['sku_id','velocity_class']],on='sku_id',validate='many_to_one')
    save(fifo,out,'2c_fifo_pallet_audit')
    # Daily-batch FIFO: compare the multiset of receipt dates shipped per SKU/day
    # to the multiset of the oldest available pallets before that day's shipments.
    # This does NOT require pallets to be picked in chronological order within day.
    stock=defaultdict(Counter)
    for r in opening.itertuples(index=False): stock[r.sku_id][r.received_date]+=1
    day_groups={day:{sk:g for sk,g in group.groupby('sku_id')}
                for day,group in o.groupby('ship_date')}
    strict_by_day=fifo.loc[~fifo.fifo_compliant].groupby(['ship_date','sku_id']).size().to_dict()
    batch=[]
    for day in pd.date_range(START,END):
        key=day.strftime('%Y-%m-%d')
        for sk,g in day_groups.get(key,{}).items():
            available=stock[sk]; actual=Counter(g.received_date)
            remaining=len(g); ideal=Counter()
            for received_date,count in sorted(available.items()):
                take=min(remaining,count)
                ideal[received_date]+=take
                remaining-=take
                if remaining==0: break
            check(remaining==0,f'Insufficient stock for {sk} on {key}')
            exceptions=sum((actual-ideal).values())
            batch.append((key,sk,len(g),exceptions,int(strict_by_day.get((key,sk),0))))
            for received_date,count in actual.items():
                check(available[received_date]>=count,f'Inventory underflow {sk} on {key}')
                available[received_date]-=count
                if not available[received_date]: del available[received_date]
        day_in=inbound_days.get(key)
        if day_in is not None:
            for r in day_in.itertuples(index=False): stock[r.sku_id][r.receipt_date]+=1
    remaining_stock=Counter({(sk,dt):count for sk,dates in stock.items() for dt,count in dates.items()})
    actual_closing=Counter((r.sku_id,r.received_date) for r in d['current_stock'].itertuples(index=False))
    check(remaining_stock==actual_closing,'Daily-batch audit closing stock mismatch')
    daily=pd.DataFrame(batch,columns=['ship_date','sku_id','shipments','daily_batch_exceptions','strict_picking_order_exceptions'])
    daily['daily_batch_compliant']=daily.daily_batch_exceptions.eq(0)
    save(daily,out,'2c_fifo_daily_batch_audit')
    save(daily.loc[~daily.daily_batch_compliant],out,'2c_fifo_daily_batch_exceptions')
    by_class=fifo.groupby('velocity_class').agg(shipped_pallets=('pallet_id','size'),fifo_rate_pct=('fifo_compliant',lambda x:100*x.mean()),mean_days=('days_in_stock','mean'),median_days=('days_in_stock','median'),p95_days=('days_in_stock',lambda x:x.quantile(.95)),max_days=('days_in_stock','max')).reset_index();save(by_class,out,'2c_residence_by_velocity')
    by_client=fifo.groupby('client_id').agg(shipped_pallets=('pallet_id','size'),fifo_rate_pct=('fifo_compliant',lambda x:100*x.mean()),mean_days=('days_in_stock','mean'),median_days=('days_in_stock','median'),p95_days=('days_in_stock',lambda x:x.quantile(.95)),max_days=('days_in_stock','max')).reset_index();save(by_client,out,'2c_residence_by_client')
    save(fifo.sort_values(['days_in_stock','pallet_id'],ascending=[False,True]).head(25),out,'2c_longest_stored_pallets')
    # Censored ages: pallets still in stock on the final date have not yet shipped.
    closing=d['current_stock'].copy()
    closing['age_days_at_end']=(END-pd.to_datetime(closing.received_date)).dt.days
    check((closing.age_days_at_end>=0).all(),'Closing inventory contains future receipt dates')
    closing=closing.merge(v[['sku_id','velocity_class','client_id']],on='sku_id',validate='many_to_one')
    save(closing.sort_values(['age_days_at_end','pallet_id'],ascending=[False,True]).head(50),
         out,'2c_oldest_pallets_still_in_stock')
    save(closing.groupby('velocity_class').agg(pallets=('pallet_id','size'),
         mean_age_days=('age_days_at_end','mean'),max_age_days=('age_days_at_end','max')).reset_index(),
         out,'2c_closing_stock_aging_by_velocity')
    save(fifo.loc[~fifo.fifo_compliant],out,'2c_strict_picking_order_exceptions')
    fig,ax=plt.subplots(figsize=(9,4));ax.hist(fifo.days_in_stock,bins=55,edgecolor='white');ax.set(xlabel='Days in warehouse before shipment',ylabel='Shipped pallets',title='2(c) Pallet storage duration');plot(out,'2c_residence_histogram')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(by_class.velocity_class,by_class.mean_days);ax.set(xlabel='SKU velocity class',ylabel='Mean days in warehouse',title='2(c) Storage duration by velocity');plot(out,'2c_residence_by_velocity')
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(by_client.client_id,by_client.mean_days);ax.set(xlabel='Client',ylabel='Mean days in warehouse',title='2(c) Storage duration by client');plot(out,'2c_residence_by_client')
    return {'shipped_pallets_audited':len(fifo),'strict_picking_order_compliance_pct':float(100*fifo.fifo_compliant.mean()),'strict_picking_order_exceptions':int((~fifo.fifo_compliant).sum()),'daily_batch_fifo_compliance_pct':float(100*(1-daily.daily_batch_exceptions.sum()/len(fifo))),'daily_batch_fifo_exceptions':int(daily.daily_batch_exceptions.sum()),'sku_day_groups_audited':len(daily),'mean_days_in_stock':float(fifo.days_in_stock.mean()),'median_days_in_stock':float(fifo.days_in_stock.median()),'max_days_in_stock':int(fifo.days_in_stock.max()),'max_age_days_of_unshipped_stock':int(closing.age_days_at_end.max()),'unshipped_pallets_aged':len(closing)}


def df_timedelta(weekdays):
    return pd.to_timedelta(weekdays,unit='D')


def task2d(d,out):
    # Slot movements include round trip and handling. Dock movements are separately counted.
    i=d['inbound'].copy();o=d['outbound'].copy()
    for df,dt,kind in [(i,'receipt_date','night'),(o,'ship_date','day')]:
        df['date']=pd.to_datetime(df[dt]);df['slot_move_min']=2*df.travel_ft/264+2*df.lift_ft/60+1.5
        df['dock_move_min']=1.8
        df['total_move_min']=df.slot_move_min+df.dock_move_min
        check((df.total_move_min>0).all(),f'Nonpositive move time {kind}')
        grouped=df.groupby('date').agg(pallets=('pallet_id','size'),slot_move_hours=('slot_move_min',lambda x:x.sum()/60),dock_move_hours=('dock_move_min',lambda x:x.sum()/60),all_moves_hours=('total_move_min',lambda x:x.sum()/60)).reset_index()
        grouped['shift']=kind
        if kind=='night': night=grouped
        else: day=grouped
    dates=pd.DataFrame({'date':pd.date_range(START,END)})
    full=pd.concat([dates.merge(day,on='date',how='left').assign(shift='day'),dates.merge(night,on='date',how='left').assign(shift='night')],ignore_index=True)
    for col in ['pallets','slot_move_hours','dock_move_hours','all_moves_hours']:full[col]=full[col].fillna(0)
    full['shift_length_hours']=np.where(full['shift']=='day',12.0,7.5)
    full['minimum_avg_forklifts_continuously_busy']=full.all_moves_hours/full.shift_length_hours
    full['idealized_forklifts_ceil']=np.ceil(full.minimum_avg_forklifts_continuously_busy).astype(int)
    full['weekday']=full.date.dt.day_name()
    save(full,out,'2d_daily_shift_workload')
    # Monday-start weeks. Distinguish prescribed slot labor from dock-move workload.
    full['week_start_monday']=full.date-df_timedelta(full.date.dt.weekday)
    weekly=full.groupby(['week_start_monday','shift'],as_index=False).agg(
        pallets=('pallets','sum'),slot_move_labor_hours=('slot_move_hours','sum'),
        dock_move_hours=('dock_move_hours','sum'),total_handling_hours=('all_moves_hours','sum'))
    save(weekly,out,'2d_weekly_shift_labor_hours')
    save(weekly.groupby('week_start_monday',as_index=False)[['pallets','slot_move_labor_hours','dock_move_hours','total_handling_hours']].sum(),
         out,'2d_weekly_total_labor_hours')
    by_shift=full.groupby('shift').agg(total_pallets=('pallets','sum'),annual_slot_hours=('slot_move_hours','sum'),annual_dock_hours=('dock_move_hours','sum'),annual_all_moves_hours=('all_moves_hours','sum'),mean_calendar_day_hours=('all_moves_hours','mean'),peak_shift_hours=('all_moves_hours','max'),p95_shift_hours=('all_moves_hours',lambda x:x.quantile(.95))).reset_index();save(by_shift,out,'2d_shift_summary')
    top=full.sort_values('all_moves_hours',ascending=False).head(30);save(top,out,'2d_peak_shift_days')
    fig,ax=plt.subplots(figsize=(12,4));
    for shift in ['day','night']:
        f=full[full['shift']==shift].sort_values('date');ax.plot(f.date,f.all_moves_hours,label=shift,linewidth=.8,alpha=.8)
    ax.set(xlabel='Date',ylabel='Total forklift work-hours (including dock moves)',title='2(d) Daily forklift workload');ax.legend();plot(out,'2d_daily_forklift_hours')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(by_shift['shift'],by_shift.annual_slot_hours,label='Slot movements');ax.bar(by_shift['shift'],by_shift.annual_dock_hours,bottom=by_shift.annual_slot_hours,label='Dock movements');ax.set(ylabel='Annual forklift work-hours',title='2(d) Annual work by shift and movement type');ax.legend();plot(out,'2d_annual_hours_breakdown')
    # Deliberately blank experiment template: must be filled from real Digital Twin runs.
    template=pd.DataFrame(columns=['historical_date','shift','forklifts','truck_wait_count','mean_truck_delay_min','max_truck_delay_min','night_end_time','queue_observations','bottleneck_explanation','notes'])
    save(template,out,'2d_DIGITAL_TWIN_EXPERIMENT_TEMPLATE')
    return {'annual_slot_move_hours':float(full.slot_move_hours.sum()),'annual_dock_move_hours':float(full.dock_move_hours.sum()),'annual_all_moves_hours':float(full.all_moves_hours.sum()),'peak_day_shift_hours':float(day.all_moves_hours.max()),'peak_night_shift_hours':float(night.all_moves_hours.max()),'weeks_reported':int(weekly.week_start_monday.nunique()),'digital_twin_status':'Not performed; template supplied, requires web Digital Twin'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('input_data'));p.add_argument('--out',type=Path,default=Path('outputs'));p.add_argument('--velocity-file',type=Path,default=None,help='Optional Task 1 SKU velocity CSV');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    d=load(a.data)
    summary={'validation':validate(d)}
    v=velocity_classes(d['skus'],d['outbound'])
    if a.velocity_file is not None:
        check(a.velocity_file.exists(),f'Velocity file not found: {a.velocity_file}')
        supplied=pd.read_csv(a.velocity_file)
        check({'sku_id','velocity_class'}.issubset(supplied.columns),'Velocity file needs sku_id and velocity_class')
        check(supplied.sku_id.is_unique,'Velocity file has duplicate SKUs')
        check(set(supplied.sku_id)==set(v.sku_id),'Velocity file SKU list does not match input data')
        compare=v[['sku_id','velocity_class']].merge(supplied[['sku_id','velocity_class']],on='sku_id',suffixes=('_computed','_task1'))
        check((compare.velocity_class_computed==compare.velocity_class_task1).all(),
              'Task 1 velocity file differs from the Task 2 ABC thresholds; resolve before proceeding')
        v=v.drop(columns=['velocity_class']).merge(supplied[['sku_id','velocity_class']],on='sku_id',validate='one_to_one')
    save(v,a.out,'task1_velocity_classes_reused')
    summary['task2a']=task2a(d,a.out)
    summary['task2b']=task2b(d,v,a.out)
    summary['task2c']=task2c(d,v,a.out)
    summary['task2d']=task2d(d,a.out)
    (a.out/'METHODOLOGY_AND_LIMITATIONS.md').write_text(
        '# Task 2 methodology and limitations\n\n'
        '2(a): rack coordinates, heights and modeled mean distances from all 32 dock doors. '
        'Visually compare spatial plot to the supplied facility_map.png manually.\n\n'
        '2(b): putaway associations are descriptive, not causal proof of drivers’ decision rule. '
        'The feasible-slot benchmark samples every 10th putaway, reconstructs free slots, '
        'and uses actual receiving-door distance. '
        'For equal putaway timestamps, order is assumed by pallet ID, which may change sampled available slots. '
        'Height slack distinguishes total unused opening height from excess beyond required 4-inch clearance; '
        'time-weighted slack is measured at the end of each operating day.\n\n'
        '2(c): strict picking-order FIFO is based on recorded pick timestamps and receipt dates; '
        'equal receipt dates are interchangeable. Daily-batch FIFO asks whether each SKU’s '
        'shipments were the oldest available as a set on that day; this is a different definition. '
        'The data do not establish physical picking order for tied timestamps. '
        'Unshipped pallets are analyzed separately as right-censored inventory ages.\n\n'
        '2(d): forklift slot-movement minutes follow the prescribed formula; 1.8-minute dock '
        'moves are shown separately and included in total handling workload. '
        'The case defines weekly labor hours specifically for slot moves; use that column '
        'when reporting the professor’s labor-hours definition; weekly tables are exported. '
        'Work-hours divided by shift length is only an idealized capacity bound, not a simulation. '
        'Digital Twin runs are not included and must be performed on the website.\n',encoding='utf-8')
    (a.out/'task2_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'\nCompleted. Files saved to {a.out.resolve()}')


if __name__=='__main__':
    main()
