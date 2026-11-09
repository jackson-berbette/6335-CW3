#!/usr/bin/env python3
"""StackBay Casework 3 Phase 1 Task 3: validated baseline, capacity-aware greedy scenario, and relaxed opportunity bound.
Run: python stackbay_task3_analysis.py --data input_data --velocity-file task1_velocity_stock.csv --out task3_outputs
The opportunity analysis is a relaxed, non-operational benchmark; see README.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

START='2025-07-07'; END='2026-07-05'

def require(ok,msg):
    if not bool(ok): raise ValueError(msg)

def write(df,out,name):
    df.to_csv(out/f'{name}.csv',index=False)

def figure(out,name):
    plt.tight_layout();plt.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight');plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('input_data'))
    ap.add_argument('--velocity-file',type=Path,default=Path('task1_velocity_stock.csv'))
    ap.add_argument('--out',type=Path,default=Path('task3_outputs'))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    names=['slots','docks','skus','clients','opening_stock','inbound','outbound','current_stock']
    d={n:pd.read_csv(a.data/f'{n}.csv') for n in names}
    s=d['slots']; sku=d['skus']; ib=d['inbound']; ob=d['outbound']; opening=d['opening_stock']; current=d['current_stock']
    require(a.velocity_file.exists(),f'Missing Task 1 velocity file: {a.velocity_file}')
    v=pd.read_csv(a.velocity_file)
    require(v.sku_id.is_unique and set(v.sku_id)==set(sku.sku_id),'Task 1 velocity SKU coverage mismatch')
    actual=ob.groupby('sku_id').size().rename('observed_pallets')
    vv=v[['sku_id','pallets_shipped','velocity_class','average_stock','peak_stock']].merge(actual,left_on='sku_id',right_index=True,how='left')
    require((vv.pallets_shipped==vv.observed_pallets.fillna(0)).all(),'Task 1 shipped volumes disagree with outbound data')
    require(len(opening)+len(ib)-len(ob)==len(current),'Inventory reconciliation failed')
    require(set(opening.pallet_id).isdisjoint(set(ib.pallet_id)),'Duplicate pallet IDs')
    require((set(opening.pallet_id)|set(ib.pallet_id))-set(ob.pallet_id)==set(current.pallet_id),'Closing inventory pallet mismatch')
    require((s.opening_height_in-s.max_pallet_in==4).all(),'Clearance mismatch')
    require((pd.to_datetime(ib.receipt_date).between(START,END)).all(),'Inbound outside history')
    require((pd.to_datetime(ob.ship_date).between(START,END)).all(),'Outbound outside history')
    # Movement-level labor and travel: one-way travel_ft, round-trip 2x, slot move minutes prescribed.
    movements=[]
    for kind,frame,slotcol,datecol in [('putaway',ib,'putaway_slot','receipt_date'),('pick',ob,'picked_from_slot','ship_date')]:
        f=frame[['pallet_id','sku_id','door_id',slotcol,datecol,'travel_ft','lift_ft']].copy()
        f=f.rename(columns={slotcol:'slot_id',datecol:'date'})
        f['movement_type']=kind
        f['slot_minutes']=2*f.travel_ft/264+2*f.lift_ft/60+1.5
        f['travel_roundtrip_ft']=2*f.travel_ft
        f['travel_minutes']=2*f.travel_ft/264
        f['lift_minutes']=2*f.lift_ft/60
        f['dock_minutes']=1.8
        movements.append(f)
    m=pd.concat(movements,ignore_index=True)
    m=m.merge(s[['slot_id','aisle','level','opening_height_in','max_pallet_in','dist_dock_ft','elev_ft']],on='slot_id',validate='many_to_one')
    m=m.merge(sku[['sku_id','client_id','pallet_height_in']],on='sku_id',validate='many_to_one')
    m=m.merge(vv[['sku_id','velocity_class']],on='sku_id',validate='many_to_one')
    require(m.notna().all().all(),'Missing movement metadata')
    require((m.pallet_height_in<=m.max_pallet_in).all(),'Historical pallet does not fit')
    require(np.allclose(m.lift_ft,m.elev_ft,atol=.11),'Lift mismatch')
    # Actual travel is door-specific and should not be replaced by average dist_dock_ft.
    dock=d['docks']; print('Dock columns:',list(dock.columns))
    door_xcol=next((c for c in ['x_ft','door_x_ft'] if c in dock.columns),None)
    if door_xcol is not None:
        dx=dock.set_index('door_id')[door_xcol]
        aisle_x=m['slot_id'].map(s.set_index('slot_id').aisle_x_ft)
        y=m['slot_id'].map(s.set_index('slot_id').y_ft)
        expected=40+(m.door_id.map(dx)-aisle_x).abs()+y
        require(np.allclose(expected,m.travel_ft,atol=.15),'Travel path inconsistent with door and slot coordinates')
    m['unused_height_in']=m.opening_height_in-m.pallet_height_in
    m['excess_clearance_in']=m.unused_height_in-4
    m['slot_hours']=m.slot_minutes/60
    m['all_moves_hours']=(m.slot_minutes+m.dock_minutes)/60
    write(m,a.out,'3b_movement_level_baseline')
    # inventory: opening stock then each day outbound before inbound; daily closing stock.
    dates=pd.date_range(START,END,freq='D')
    ibc=ib.groupby('receipt_date').size();obc=ob.groupby('ship_date').size()
    daily=pd.DataFrame({'date':dates.strftime('%Y-%m-%d')})
    daily['received']=daily.date.map(ibc).fillna(0).astype(int)
    daily['shipped']=daily.date.map(obc).fillna(0).astype(int)
    daily['opening_pallets']=len(opening)+daily.received.cumsum().shift(fill_value=0)-daily.shipped.cumsum().shift(fill_value=0)
    daily['after_day_picks']=daily.opening_pallets-daily.shipped
    daily['closing_pallets']=daily.after_day_picks+daily.received
    require((daily[['opening_pallets','after_day_picks','closing_pallets']]>=0).all().all(),'Negative aggregate inventory')
    require(int(daily.closing_pallets.iloc[-1])==len(current),'Closing pallet count differs')
    daily['closing_slot_occupancy_pct']=100*daily.closing_pallets/len(s)
    write(daily,a.out,'3b_daily_inventory_occupancy')
    # Time-weighted opening height of occupied pallets at end of each day; process pallet identity events.
    lookup=sku.set_index('sku_id').pallet_height_in.to_dict()
    sloth=s.set_index('slot_id').opening_height_in.to_dict()
    origin=pd.concat([opening[['pallet_id','sku_id','slot_id']],ib[['pallet_id','sku_id','putaway_slot']].rename(columns={'putaway_slot':'slot_id'})],ignore_index=True)
    require(origin.pallet_id.is_unique,'Origin pallet ID duplicate')
    origin['unused_height_in']=origin.apply(lambda r:sloth[r.slot_id]-lookup[r.sku_id],axis=1)
    heights=origin.set_index('pallet_id').unused_height_in.to_dict()
    initial_sum=sum(heights[p] for p in opening.pallet_id)
    ibsum=ib.assign(h=ib.pallet_id.map(heights)).groupby('receipt_date').h.sum()
    obsum=ob.assign(h=ob.pallet_id.map(heights)).groupby('ship_date').h.sum()
    daily['total_unused_height_in']=initial_sum+daily.date.map(ibsum).fillna(0).cumsum()-daily.date.map(obsum).fillna(0).cumsum()
    daily['avg_unused_height_occupied_in']=daily.total_unused_height_in/daily.closing_pallets.replace(0,np.nan)
    write(daily[['date','closing_pallets','closing_slot_occupancy_pct','total_unused_height_in','avg_unused_height_occupied_in']],a.out,'3b_daily_space_fit')
    # Reproducible definitions, including invariants unaffected by storage design.
    definitions=[
      ('one_way_slot_travel_ft','Sum travel_ft for all inbound and outbound pallets','inbound.csv, outbound.csv','Yes: slot placement changes path'),
      ('round_trip_slot_travel_ft','2 * sum travel_ft for all inbound and outbound pallets','inbound.csv, outbound.csv','Yes'),
      ('slot_move_labor_hours','Sum(2*travel_ft/264 + 2*lift_ft/60 + 1.5)/60','inbound.csv, outbound.csv','Yes: travel and lift change'),
      ('dock_move_hours','1.8*(number of inbound + outbound pallets)/60','inbound.csv, outbound.csv','No: fixed flow and handling rule'),
      ('total_handling_hours','slot_move_labor_hours + dock_move_hours','inbound.csv, outbound.csv','Partly: slot component only'),
      ('mean_closing_slot_occupancy_pct','Mean over all 364 calendar days of daily closing pallets / 8365 * 100','opening_stock.csv, inbound.csv, outbound.csv, slots.csv','No: inventory flow fixed'),
      ('peak_closing_slot_occupancy_pct','Maximum daily closing pallets / 8365 * 100','same as above','No: inventory flow fixed'),
      ('mean_unused_height_per_putaway_in','Mean(opening_height_in - pallet_height_in) across inbound pallet placements','inbound.csv, slots.csv, skus.csv','Yes'),
      ('occupied_pallet_day_weighted_unused_height_in','Sum daily closing unused inches / sum daily closing pallet counts','opening_stock.csv, inbound.csv, outbound.csv, slots.csv, skus.csv','Yes'),
      ('annual_pallets_received','Number of inbound.csv rows','inbound.csv','No: receipts fixed'),
      ('annual_pallets_shipped','Number of outbound.csv rows','outbound.csv','No: shipments fixed'),
      ('mean_closing_inventory_pallets','Mean daily closing pallet count','opening_stock.csv, inbound.csv, outbound.csv','No: fixed flow'),
      ('peak_closing_inventory_pallets','Maximum daily closing pallet count','same as above','No: fixed flow'),
      ('mean_shipped_pallet_age_days','Mean days_in_stock for shipped pallets','outbound.csv','No: fixed pallet identities and dates')]
    write(pd.DataFrame(definitions,columns=['metric','calculation','source_files','can_storage_design_change']),a.out,'3a_metric_definitions')
    # Baseline scorecard.
    score={
      'warehouse_slots':len(s),'sku_count':len(sku),'days_in_historical_year':len(daily),
      'pallets_received':len(ib),'pallets_shipped':len(ob),'opening_inventory_pallets':len(opening),
      'closing_inventory_pallets':len(current),
      'one_way_slot_travel_ft':float(m.travel_ft.sum()),
      'round_trip_slot_travel_ft':float(m.travel_roundtrip_ft.sum()),
      'slot_move_labor_hours':float(m.slot_hours.sum()),
      'dock_move_hours':float(m.dock_minutes.sum()/60),
      'total_handling_hours':float(m.all_moves_hours.sum()),
      'mean_closing_inventory_pallets':float(daily.closing_pallets.mean()),
      'peak_closing_inventory_pallets':int(daily.closing_pallets.max()),
      'mean_closing_slot_occupancy_pct':float(daily.closing_slot_occupancy_pct.mean()),
      'peak_closing_slot_occupancy_pct':float(daily.closing_slot_occupancy_pct.max()),
      'mean_unused_height_per_putaway_in':float(m.loc[m.movement_type=='putaway','unused_height_in'].mean()),
      'occupied_pallet_day_weighted_unused_height_in':float(daily.total_unused_height_in.sum()/daily.closing_pallets.sum()),
      'mean_shipped_pallet_age_days':float(ob.days_in_stock.mean())}
    write(pd.DataFrame([{'metric':k,'value':v} for k,v in score.items()]),a.out,'3b_baseline_scorecard')
    (a.out/'3b_baseline_scorecard.json').write_text(json.dumps(score,indent=2))
    # 3c: report by all four requested dimensions and each movement type.
    totaltravel=m.travel_roundtrip_ft.sum(); totalhours=m.slot_hours.sum()
    for dim in ['velocity_class','client_id','level','aisle']:
        for bytype in [False,True]:
            keys=[dim,'movement_type'] if bytype else [dim]
            g=m.groupby(keys,dropna=False).agg(pallet_movements=('pallet_id','size'),one_way_travel_ft=('travel_ft','sum'),round_trip_travel_ft=('travel_roundtrip_ft','sum'),slot_move_labor_hours=('slot_hours','sum'),dock_move_hours=('dock_minutes',lambda x:x.sum()/60),mean_lift_ft=('lift_ft','mean')).reset_index()
            g['share_all_slot_travel_pct']=100*g.round_trip_travel_ft/totaltravel
            g['share_all_slot_labor_pct']=100*g.slot_move_labor_hours/totalhours
            write(g,a.out,'3c_by_'+dim+('_movement_type' if bytype else ''))
    # Fastest 5% SKUs: ceil(5% of all SKUs), rank by shipped pallet count desc then SKU ID asc.
    ranked=vv.sort_values(['pallets_shipped','sku_id'],ascending=[False,True]).reset_index(drop=True)
    k=int(np.ceil(.05*len(ranked)))
    fastest=set(ranked.head(k).sku_id)
    m['fastest_5pct_sku']=m.sku_id.isin(fastest)
    fast=m.groupby('fastest_5pct_sku').agg(pallet_movements=('pallet_id','size'),round_trip_travel_ft=('travel_roundtrip_ft','sum'),slot_move_labor_hours=('slot_hours','sum')).reset_index()
    fast['travel_share_pct']=100*fast.round_trip_travel_ft/totaltravel
    fast['labor_share_pct']=100*fast.slot_move_labor_hours/totalhours
    write(fast,a.out,'3c_fastest_5pct_contribution')
    write(ranked.assign(fastest_5pct_sku=ranked.sku_id.isin(fastest)),a.out,'3c_sku_velocity_rank')
    # 3d: relaxed best-possible compatible slot for every fast-SKU movement, per ACTUAL door.
    # Does not enforce slot uniqueness, contemporaneous occupancy, SKU FIFO, or physical feasibility.
    # Therefore this is an optimistic potential, NOT an implementable warehouse design.
    dockx=dock.set_index('door_id')[door_xcol].to_dict() if door_xcol else None
    require(dockx is not None,'Cannot calculate door-specific benchmark: docks x coordinate missing')
    sf=s[['slot_id','aisle_x_ft','y_ft','elev_ft','max_pallet_in']].copy()
    doors=sorted(m.loc[m.fastest_5pct_sku,'door_id'].unique())
    heights_sorted=sorted(m.loc[m.fastest_5pct_sku,'pallet_height_in'].unique())
    best={}
    for door in doors:
        x=dockx[door]
        dist=40+np.abs(sf.aisle_x_ft.to_numpy()-x)+sf.y_ft.to_numpy()
        mins=2*dist/264+2*sf.elev_ft.to_numpy()/60+1.5
        for height in heights_sorted:
            fit=sf.max_pallet_in.to_numpy()>=height
            require(fit.any(),f'No feasible slot for pallet height {height}')
            # min distance is a travel-only bound; min time may select a different slot.
            best[(door,height)]=(float(dist[fit].min()),float(mins[fit].min()))
    target=m.loc[m.fastest_5pct_sku].copy()
    keys=list(zip(target.door_id,target.pallet_height_in))
    target['relaxed_best_one_way_ft']=[best[key][0] for key in keys]
    target['relaxed_best_slot_minutes']=[best[key][1] for key in keys]
    target['travel_savings_lower_bound_ft']=0.0
    target['optimistic_travel_savings_roundtrip_ft']=np.maximum(0,2*(target.travel_ft-target.relaxed_best_one_way_ft))
    target['optimistic_labor_savings_hours']=np.maximum(0,(target.slot_minutes-target.relaxed_best_slot_minutes)/60)
    require((target.optimistic_travel_savings_roundtrip_ft>=0).all(),'Invalid savings')
    write(target[['movement_type','date','sku_id','pallet_id','door_id','slot_id','pallet_height_in','travel_ft','lift_ft','slot_minutes','relaxed_best_one_way_ft','relaxed_best_slot_minutes','optimistic_travel_savings_roundtrip_ft','optimistic_labor_savings_hours']],a.out,'3d_relaxed_fast_sku_opportunity_per_movement')
    opportunity={
       'fastest_sku_count':k,'fastest_sku_pct_of_all':100*k/len(sku),
       'fastest_sku_pallets_shipped':int(ranked.head(k).pallets_shipped.sum()),
       'fastest_sku_share_of_total_shipments_pct':100*ranked.head(k).pallets_shipped.sum()/len(ob),
       'fastest_sku_share_of_all_roundtrip_slot_travel_pct':float(100*target.travel_roundtrip_ft.sum()/totaltravel),
       'baseline_total_roundtrip_slot_travel_ft':float(totaltravel),
       'baseline_total_slot_labor_hours':float(totalhours),
       'optimistic_upper_bound_travel_savings_ft':float(target.optimistic_travel_savings_roundtrip_ft.sum()),
       'optimistic_upper_bound_travel_savings_pct_of_all':float(100*target.optimistic_travel_savings_roundtrip_ft.sum()/totaltravel),
       'optimistic_upper_bound_slot_labor_savings_hours':float(target.optimistic_labor_savings_hours.sum()),
       'optimistic_upper_bound_slot_labor_savings_pct_of_all':float(100*target.optimistic_labor_savings_hours.sum()/totalhours),
       'benchmark_status':'RELAXED / NOT CAPACITY-FEASIBLE; not a prediction of realizable savings'}
    write(pd.DataFrame([{'metric':k,'value':v} for k,v in opportunity.items()]),a.out,'3d_opportunity_summary')
    (a.out/'3d_opportunity_summary.json').write_text(json.dumps(opportunity,indent=2))

    # Additional reproducible, stakeholder-specific scorecard.
    by_client=m.groupby('client_id').agg(movements=('pallet_id','size'),travel_ft=('travel_roundtrip_ft','sum'),slot_hours=('slot_hours','sum')).reset_index()
    by_client['roundtrip_ft_per_movement']=by_client.travel_ft/by_client.movements
    by_client['slot_minutes_per_movement']=60*by_client.slot_hours/by_client.movements
    write(by_client,a.out,'3b_client_handling_kpis')
    by_date=m.groupby(['date','movement_type']).agg(movements=('pallet_id','size'),slot_hours=('slot_hours','sum'),dock_hours=('dock_minutes',lambda x:x.sum()/60)).reset_index()
    write(by_date,a.out,'3b_daily_handling_kpis')
    stakeholder=[
      ('StackBay','annual_slot_labor_hours',score['slot_move_labor_hours'],'hours','Sum of all slot movement minutes /60','Placement changes travel and lift'),
      ('StackBay','annual_dock_labor_hours',score['dock_move_hours'],'hours','1.8 minutes times all pallet movements /60','Fixed by observed throughput'),
      ('StackBay','mean_daily_slot_occupancy',score['mean_closing_slot_occupancy_pct'],'percent','Average daily closing occupied slots / total slots','Fixed pallet flow; placement does not change count'),
      ('StackBay','occupied_pallet_day_unused_height',score['occupied_pallet_day_weighted_unused_height_in'],'inches','Sum unused opening inches on occupied pallet-days / occupied pallet-days','Changes with placement'),
      ('Clients','mean_shipped_pallet_age',score['mean_shipped_pallet_age_days'],'days','Average shipped pallet days_in_stock','Fixed under identical shipment/receipt identities'),
      ('Clients','total_pallets_shipped',score['pallets_shipped'],'pallets','Number of outbound pallet records','Fixed historical demand'),
      ('Forklift drivers','mean_roundtrip_travel_per_movement',score['round_trip_slot_travel_ft']/len(m),'feet/movement','2*sum travel_ft / total pallet movements','Changes with placement'),
      ('Forklift drivers','mean_slot_minutes_per_movement',60*score['slot_move_labor_hours']/len(m),'minutes/movement','Sum slot minutes / total pallet movements','Changes with placement'),
      ('Forklift drivers','peak_day_slot_hours',float(by_date.loc[by_date.movement_type=='pick','slot_hours'].max()),'hours','Max shipping-day slot work-hours','Changes with placement'),
      ('Forklift drivers','peak_night_slot_hours',float(by_date.loc[by_date.movement_type=='putaway','slot_hours'].max()),'hours','Max receiving-night slot work-hours','Changes with placement')]
    write(pd.DataFrame(stakeholder,columns=['stakeholder','metric','value','units','method','storage_design_effect']),a.out,'3a_3b_stakeholder_scorecard')

    # Validate actual storage events: every pallet has one origin, every shipment frees
    # its original slot, and every receipt enters a vacant slot after daytime picks.
    # Same-timestamp inbound rows are processed in input order; only day-level
    # ordering is needed for physical vacancy validation.
    slot_ids=s.slot_id.to_numpy(); slotidx={z:i for i,z in enumerate(slot_ids)}
    occupied=np.full(len(s),'',dtype=object)
    pallet_slots={}
    for r in opening.itertuples(index=False):
        j=slotidx[r.slot_id]
        require(occupied[j]=='',f'Duplicate opening slot: {r.slot_id}')
        occupied[j]=r.pallet_id;pallet_slots[r.pallet_id]=j
    # Sort by timestamp within each shift, and always pick before putaway on a date.
    out_events=ob.sort_values(['ship_date','picked_at','pallet_id'],kind='stable')
    in_events=ib.sort_values(['receipt_date','putaway_at','pallet_id'],kind='stable')
    for day in dates.strftime('%Y-%m-%d'):
        for r in out_events.loc[out_events.ship_date==day].itertuples(index=False):
            require(r.pallet_id in pallet_slots,f'Pallet picked before receipt: {r.pallet_id}')
            j=pallet_slots.pop(r.pallet_id)
            require(slot_ids[j]==r.picked_from_slot,f'Actual pick slot mismatch: {r.pallet_id}')
            require(occupied[j]==r.pallet_id,f'Actual occupied slot mismatch: {r.pallet_id}')
            occupied[j]=''
        for r in in_events.loc[in_events.receipt_date==day].itertuples(index=False):
            j=slotidx[r.putaway_slot]
            require(occupied[j]=='',f'Actual slot collision on {day}: {r.putaway_slot}')
            require(r.pallet_id not in pallet_slots,f'Duplicate pallet receipt: {r.pallet_id}')
            occupied[j]=r.pallet_id;pallet_slots[r.pallet_id]=j
    require(set(pallet_slots)==set(current.pallet_id),'Pallet identity reconciliation failed at closing')
    require(all(slot_ids[j]==current.set_index('pallet_id').loc[p,'slot_id'] for p,j in pallet_slots.items()),'Closing slot placement mismatch')
    write(pd.DataFrame([{'check':'opening slot uniqueness','passed':True},
                        {'check':'chronological pallet identity and slot vacancy','passed':True},
                        {'check':'closing pallet IDs and slot positions','passed':True},
                        {'check':'pallet height fits actual slot','passed':True}]),a.out,'3b_physical_inventory_validation')

    # Capacity-aware counterfactual: keep opening pallets in place; ship exactly
    # the same pallet IDs on the same dates; reassign only new putaways. Fast SKUs
    # receive closest *currently vacant compatible* slots from their actual doors.
    # Other SKUs retain their historical slot if available, otherwise use closest
    # compatible vacant slot. No relocations; one pallet per slot; same daily order.
    # This is an illustrative greedy policy, NOT an optimized design or simulation.
    heights=s.max_pallet_in.to_numpy(float)
    elev=s.elev_ft.to_numpy(float)
    aisle_x=s.aisle_x_ft.to_numpy(float); y=s.y_ft.to_numpy(float)
    hmap=sku.set_index('sku_id').pallet_height_in.to_dict()
    free=np.ones(len(s),dtype=bool)
    alt_pallet_slot={}
    for r in opening.itertuples(index=False):
        j=slotidx[r.slot_id];free[j]=False;alt_pallet_slot[r.pallet_id]=j
    # Cache sorted feasible indices by (door, pallet height), ordered by travel.
    cache={}
    def candidates(door,height):
        key=(door,float(height))
        if key not in cache:
            dist=40+np.abs(aisle_x-dockx[door])+y
            feasible=np.flatnonzero(heights>=height)
            require(len(feasible)>0,f'No slots fit height {height}')
            order=np.lexsort((feasible,elev[feasible],dist[feasible]))
            cache[key]=feasible[order]
        return cache[key]
    baseline_travel=0.;alternative_travel=0.;baseline_minutes=0.;alternative_minutes=0.
    fast_base=0.;fast_alt=0.;fast_base_minutes=0.;fast_alt_minutes=0.
    substitutions=0;fast_receipts=0;movements_count=0;peak_used=0
    results=[]
    for day in dates.strftime('%Y-%m-%d'):
        for kind,rows in [('pick',out_events.loc[out_events.ship_date==day]),('putaway',in_events.loc[in_events.receipt_date==day])]:
            for r in rows.itertuples(index=False):
                is_fast=r.sku_id in fastest
                if kind=='pick':
                    require(r.pallet_id in alt_pallet_slot,f'Counterfactual missing shipped pallet {r.pallet_id}')
                    j=alt_pallet_slot.pop(r.pallet_id)
                    require(not free[j],f'Counterfactual empty pick slot {r.pallet_id}')
                    free[j]=True
                else:
                    height=hmap[r.sku_id]
                    historical_j=slotidx[r.putaway_slot]
                    if not is_fast and free[historical_j] and heights[historical_j]>=height:
                        j=historical_j
                    else:
                        cand=candidates(r.door_id,height)
                        available=cand[free[cand]]
                        require(len(available)>0,f'Counterfactual has no feasible empty slot on {day} for {r.pallet_id}')
                        j=int(available[0])
                    require(free[j],'Counterfactual double allocation')
                    free[j]=False;alt_pallet_slot[r.pallet_id]=j
                    substitutions+=int(j!=historical_j)
                    fast_receipts+=int(is_fast)
                actual_ft=float(r.travel_ft)
                new_ft=float(40+abs(aisle_x[j]-dockx[r.door_id])+y[j])
                actual_min=2*actual_ft/264+2*float(r.lift_ft)/60+1.5
                new_min=2*new_ft/264+2*elev[j]/60+1.5
                baseline_travel+=2*actual_ft;alternative_travel+=2*new_ft
                baseline_minutes+=actual_min;alternative_minutes+=new_min
                if is_fast:
                    fast_base+=2*actual_ft;fast_alt+=2*new_ft
                    fast_base_minutes+=actual_min;fast_alt_minutes+=new_min
                movements_count+=1
                results.append((day,kind,r.pallet_id,r.sku_id,r.door_id,slot_ids[j],round(2*actual_ft,3),round(2*new_ft,3),round(actual_min,5),round(new_min,5),is_fast))
        peak_used=max(peak_used,int((~free).sum()))
    require(set(alt_pallet_slot)==set(current.pallet_id),'Counterfactual closing inventory identities mismatch')
    require(int((~free).sum())==len(current),'Counterfactual closing occupancy mismatch')
    require(np.isclose(baseline_travel,totaltravel,rtol=1e-8),'Counterfactual baseline travel mismatch')
    require(np.isclose(baseline_minutes/60,totalhours,rtol=1e-8),'Counterfactual baseline labor mismatch')
    write(pd.DataFrame(results,columns=['date','movement_type','pallet_id','sku_id','door_id','counterfactual_slot_id','actual_roundtrip_ft','counterfactual_roundtrip_ft','actual_slot_minutes','counterfactual_slot_minutes','fastest_5pct_sku']),a.out,'3d_capacity_aware_movement_comparison')
    scenario={
       'scenario':'Greedy closest-free-slot for fast SKUs, historical slot preference for other SKUs',
       'initial_placements':'Historical opening stock unchanged',
       'event_sequence':'Daily picks before putaways; chronological within each shift',
       'fastest_sku_count':k,'movements':movements_count,'fast_receipts':fast_receipts,
       'putaways_with_different_slot':substitutions,'peak_end_of_day_occupied_slots':peak_used,
       'baseline_roundtrip_travel_ft':baseline_travel,'scenario_roundtrip_travel_ft':alternative_travel,
       'travel_savings_ft':baseline_travel-alternative_travel,
       'travel_savings_pct':100*(baseline_travel-alternative_travel)/baseline_travel,
       'baseline_slot_labor_hours':baseline_minutes/60,'scenario_slot_labor_hours':alternative_minutes/60,
       'slot_labor_savings_hours':(baseline_minutes-alternative_minutes)/60,
       'slot_labor_savings_pct':100*(baseline_minutes-alternative_minutes)/baseline_minutes,
       'fast_sku_baseline_travel_ft':fast_base,'fast_sku_scenario_travel_ft':fast_alt,
       'fast_sku_baseline_slot_hours':fast_base_minutes/60,'fast_sku_scenario_slot_hours':fast_alt_minutes/60,
       'limitations':'One pallet per slot and daily availability enforced. Greedy, not globally optimized; ignores traffic and shift queuing. Historical picks fixed; actual slot assignments of non-fast SKUs may be displaced. No claim of realized savings.'}
    write(pd.DataFrame([{'metric':k,'value':v} for k,v in scenario.items()]),a.out,'3d_capacity_aware_scenario_summary')
    (a.out/'3d_capacity_aware_scenario_summary.json').write_text(json.dumps(scenario,indent=2))
    # Evidence-based report guidance, including counterexamples and non-comparable assumptions.
    lines=[
       '# Task 3(e): Evidence for the performance judgment',
       '',
       'This is an analytical draft; review it against your team’s interpretation before submission.',
       '',
       f'- Observed average daily closing slot occupancy: {score["mean_closing_slot_occupancy_pct"]:.2f}%; peak {score["peak_closing_slot_occupancy_pct"]:.2f}%.',
       f'- Annual slot travel: {totaltravel:,.0f} round-trip feet; slot labor: {totalhours:,.2f} hours.',
       f'- Fastest {k} SKUs: {opportunity["fastest_sku_share_of_all_roundtrip_slot_travel_pct"]:.2f}% of slot travel.',
       f'- Relaxed (non-feasible) benchmark: {opportunity["optimistic_upper_bound_travel_savings_pct_of_all"]:.2f}% travel and {opportunity["optimistic_upper_bound_slot_labor_savings_pct_of_all"]:.2f}% slot labor.',
       f'- Capacity-aware greedy scenario: {scenario["travel_savings_pct"]:.2f}% travel and {scenario["slot_labor_savings_pct"]:.2f}% slot labor; this is a scenario, not a guarantee.',
       '',
       '## Inherent conditions',
       'Historical demand, pallet dimensions, facility geometry, fixed dock locations, and opening inventory constrain the possible outcome.',
       '',
       '## Controllable choices',
       'Which vacant compatible slot receives each incoming pallet can change travel, lift time, and unused height.',
       '',
       '## Cautions',
       'Do not interpret occupancy below 100% as proof of surplus *usable* capacity: slot-height compatibility matters.',
       'The greedy counterfactual does not simulate forklift interference, service reliability, or truck departures.',
       'Its end-of-year pallet locations need not match historical current_stock.csv; pallet identities and counts do.',
       'The relaxed benchmark is not necessarily an attainable target, and its labor and travel minima can use different slots.',
       'The scorecard does not prove that client service was poor; it only evaluates metrics supported by the files.'
    ]
    (a.out/'3e_evidence_based_assessment.md').write_text('\n'.join(lines)+'\n')
    print('CAPACITY_AWARE_SCENARIO:',json.dumps(scenario,indent=2))

    # Charts
    plt.figure(figsize=(10,4));plt.plot(pd.to_datetime(daily.date),daily.closing_slot_occupancy_pct,lw=1);plt.ylabel('Closing slot occupancy (%)');plt.title('Daily closing occupancy');figure(a.out,'3b_occupancy_timeseries')
    for dim in ['velocity_class','client_id','level','aisle']:
        t=pd.read_csv(a.out/f'3c_by_{dim}.csv')
        plt.figure(figsize=(10 if dim!='aisle' else 13,4));plt.bar(t[dim].astype(str),t.round_trip_travel_ft/1e6);plt.ylabel('Round-trip slot travel (million ft)');plt.title('Travel by '+dim);plt.xticks(rotation=45 if dim in ['aisle','client_id'] else 0);figure(a.out,'3c_travel_by_'+dim)
    print(json.dumps({'scorecard':score,'opportunity':opportunity},indent=2))

if __name__=='__main__': main()
