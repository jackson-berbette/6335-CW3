
"""StackBay Phase 1 Task 1(a-f). Run: python task1.py --data /path/to/csvs --out outputs"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

START=pd.Timestamp('2025-07-07'); END=pd.Timestamp('2026-07-05'); DAYS=pd.date_range(START,END)
FILES=['clients','client_sites','skus','suppliers','slots','docks','opening_stock','inbound','outbound','trucks','current_stock']

def check(condition,message):
    if not bool(condition): raise ValueError('DATA VALIDATION FAILED: '+message)

def write(df,out,name,index=False):
    df.to_csv(out/(name+'.csv'),index=index)

def plot_save(out,name):
    plt.tight_layout();plt.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close()

def main(data,out):
    out.mkdir(parents=True,exist_ok=True)
    d={n:pd.read_csv(data/(n+'.csv')) for n in FILES}
    c,sites,sku,sup,slots,docks,opening,ib,ob,trucks,closing=[d[n] for n in FILES]
    for t,col in [(ib,'receipt_date'),(ob,'ship_date'),(trucks,'ship_date')]:t[col]=pd.to_datetime(t[col])
    for t in (opening,closing):t['received_date']=pd.to_datetime(t['received_date'])
    ob['picked_at']=pd.to_datetime(ob['picked_at']);ob['departs_at']=pd.to_datetime(ob['departs_at']);ib['arrived_at']=pd.to_datetime(ib['arrived_at'])
    check(len(slots)==8365 and slots.slot_id.is_unique,'slots not 8365 unique')
    check(len(sku)==1189 and sku.sku_id.is_unique,'SKUs not 1189 unique')
    check(len(sites)==936 and sites.site_id.is_unique,'sites not 936 unique')
    check(len(docks)==32 and docks.door_id.is_unique,'doors not 32 unique')
    check(c.client_id.is_unique and sup.supplier_id.is_unique,'duplicate master IDs')
    check(trucks.truck_id.is_unique,'duplicate truck IDs')
    check(ib.pallet_id.is_unique and ob.pallet_id.is_unique,'duplicate movement pallet IDs')
    check(opening.pallet_id.is_unique and closing.pallet_id.is_unique,'duplicate stock pallet IDs')
    check(opening.slot_id.is_unique and closing.slot_id.is_unique,'duplicate occupied stock slots')
    check(set(ib.sku_id).issubset(set(sku.sku_id)) and set(ob.sku_id).issubset(set(sku.sku_id)),'unknown movement SKU')
    check(set(ob.site_id).issubset(set(sites.site_id)),'unknown site')
    check(set(ob.truck_id).issubset(set(trucks.truck_id)),'unknown truck')
    check(set(ib.putaway_slot).issubset(set(slots.slot_id)) and set(ob.picked_from_slot).issubset(set(slots.slot_id)),'unknown slot')
    check(set(opening.slot_id).issubset(set(slots.slot_id)) and set(closing.slot_id).issubset(set(slots.slot_id)),'unknown stock slot')
    check(ib.receipt_date.between(START,END).all() and ob.ship_date.between(START,END).all(),'movements outside year')
    check(len(opening)+len(ib)-len(ob)==len(closing),'global inventory balance')
    check(set(opening.pallet_id).isdisjoint(set(ib.pallet_id)),'opening pallet also inbound')
    available=set(opening.pallet_id)|set(ib.pallet_id)
    check(set(ob.pallet_id).issubset(available),'outbound pallet not supplied')
    check((available-set(ob.pallet_id))==set(closing.pallet_id),'closing pallet identities mismatch')
    check(ob.groupby('truck_id').size().reindex(trucks.truck_id).to_numpy().tolist()==trucks.pallets.to_list(),'truck pallets mismatch')
    check((sku.ti*sku.hi==sku.cases_per_pallet).all(),'SKU Ti*Hi mismatch')
    check(c.set_index('client_id').sites.sort_index().equals(sites.groupby('client_id').size().sort_index()),'client store counts mismatch')
    check(c.set_index('client_id').skus.sort_index().equals(sku.groupby('client_id').size().sort_index()),'client SKU counts mismatch')
    check(sup.set_index('supplier_id').skus.sort_index().equals(sku.groupby('supplier_id').size().sort_index()),'supplier SKU counts mismatch')
    check(ob.merge(sku[['sku_id','client_id']],on='sku_id',suffixes=('','_master')).eval('client_id == client_id_master').all(),'outbound client does not match SKU')
    check(ob.merge(sites[['site_id','client_id']],on='site_id',suffixes=('','_site')).eval('client_id == client_id_site').all(),'outbound site does not match client')
    check(ib.merge(sku[['sku_id','supplier_id']],on='sku_id',suffixes=('','_master')).eval('supplier_id == supplier_id_master').all(),'inbound supplier does not match SKU')
    check(ob.merge(trucks[['truck_id','ship_date','door_id','route_id']],on='truck_id',suffixes=('','_truck')).eval('(ship_date == ship_date_truck) & (door_id == door_id_truck) & (route_id == route_id_truck)').all(),'outbound truck metadata mismatch')
    # Slot fit for all historical and stock placements.
    for tbl,slotcol in [(ib,'putaway_slot'),(ob,'picked_from_slot'),(opening,'slot_id'),(closing,'slot_id')]:
        fitted=tbl[['sku_id',slotcol]].merge(sku[['sku_id','pallet_height_in']],on='sku_id').merge(slots[['slot_id','max_pallet_in']],left_on=slotcol,right_on='slot_id')
        check((fitted.pallet_height_in<=fitted.max_pallet_in+1e-6).all(),f'height violation in {slotcol}')
    # 1a client profile
    client=c.merge(sites.groupby('client_id').agg(states=('state','nunique'),cities=('city','nunique')).reset_index(),on='client_id')
    shipped=ob.groupby('client_id').size().rename('pallets_shipped_year')
    client=client.join(shipped,on='client_id').fillna({'pallets_shipped_year':0})
    client['pallets_per_week']=client.pallets_shipped_year/52
    client['share_of_shipped_pct']=100*client.pallets_shipped_year/len(ob)
    client['pallets_per_store_per_week']=client.pallets_per_week/client.sites
    client=client.sort_values('pallets_shipped_year',ascending=False)
    write(client,out,'1a_client_profile')
    geography=sites.groupby(['client_id','state']).size().rename('stores').reset_index().merge(c[['client_id','client_name']],on='client_id')
    write(geography,out,'1a_client_states')
    fig,ax=plt.subplots(figsize=(10,5));ax.barh(client.client_name[::-1],client.pallets_per_week[::-1]);ax.set_xlabel('Pallets shipped per week');ax.set_title('1(a) Client shipping volumes');plot_save(out,'1a_client_volumes')
    piv=geography.pivot_table(index='client_name',columns='state',values='stores',fill_value=0).reindex(client.client_name)
    piv.plot(kind='bar',stacked=True,figsize=(11,5));plt.ylabel('Restaurant sites');plt.title('1(a) Restaurant geography by client and state');plot_save(out,'1a_client_geography')
    # 1b SKU velocity: A first 80% volume, B next 15%, C remaining 5%; boundary assigned by cumulative PRIOR to item, preserving whole SKUs.
    velocity=sku.copy().merge(ob.groupby('sku_id').size().rename('pallets_shipped').reset_index(),on='sku_id',how='left')
    velocity.pallets_shipped=velocity.pallets_shipped.fillna(0).astype(int)
    velocity=velocity.sort_values(['pallets_shipped','sku_id'],ascending=[False,True]).reset_index(drop=True)
    velocity['rank']=np.arange(1,len(velocity)+1)
    velocity['cum_pallet_share']=velocity.pallets_shipped.cumsum()/len(ob)
    velocity['prior_cum_share']=(velocity.pallets_shipped.cumsum()-velocity.pallets_shipped)/len(ob)
    velocity['velocity_class']=np.select([velocity.prior_cum_share<.80,velocity.prior_cum_share<.95],['A','B'],default='C')
    velocity['sku_percentile']=100*velocity['rank']/len(velocity)
    write(velocity,out,'1b_sku_velocity')
    classes=velocity.groupby('velocity_class').agg(skus=('sku_id','size'),pallets=('pallets_shipped','sum'),avg_height_in=('pallet_height_in','mean'),avg_weight_lb=('pallet_weight_lb','mean')).reset_index()
    classes['volume_share_pct']=100*classes.pallets/len(ob);write(classes,out,'1b_velocity_class_summary')
    fig,ax=plt.subplots(figsize=(9,5));ax.plot(np.r_[0,velocity['rank']/len(velocity)*100],np.r_[0,velocity.cum_pallet_share*100]);ax.axhline(80,ls='--',color='gray');ax.axhline(95,ls=':',color='gray');ax.set(xlabel='Top SKUs (%)',ylabel='Cumulative shipped pallets (%)',title='1(b) SKU velocity Pareto curve',xlim=(0,100),ylim=(0,101));plot_save(out,'1b_pareto')
    fig,axs=plt.subplots(1,2,figsize=(11,4));
    for ax,col,title in [(axs[0],'pallet_height_in','Pallet height (in)'),(axs[1],'pallet_weight_lb','Pallet weight (lb)')]:
        for cl in ['A','B','C']:
            v=velocity.loc[velocity.velocity_class==cl,col];ax.hist(v,bins=22,alpha=.45,label=f'{cl} ({len(v)} SKUs)')
        ax.set(xlabel=title,ylabel='Number of SKUs');ax.legend()
    fig.suptitle('1(b) Height and weight by velocity class (SKU counts)');plot_save(out,'1b_heights_weights')
    # 1c shipping by operational week starting Monday; historical year 52 full weeks
ob['week_index']=((ob.ship_date-START).dt.days//7)+1
    weekly=ob.groupby('week_index').size().reindex(range(1,53),fill_value=0).rename('pallets').reset_index()
    weekly['week_start']=START+pd.to_timedelta((weekly.week_index-1)*7,unit='D')
    weekly['week_over_week_pct']=weekly.pallets.pct_change()*100
    write(weekly,out,'1c_weekly_demand')
    ob['weekday']=ob.ship_date.dt.day_name();ob['departure_hour']=ob.departs_at.dt.hour;ob['pick_hour']=ob.picked_at.dt.hour
    daily_ship=ob.groupby('ship_date').size().reindex(DAYS,fill_value=0).rename('pallets')
    daily_ship.rename_axis('date').reset_index().to_csv(out/'1c_daily_demand.csv',index=False)
    weekday=daily_ship.groupby(daily_ship.index.day_name()).agg(['sum','mean','max']).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index().rename(columns={'index':'weekday'})
    write(weekday,out,'1c_weekday_demand')
    hours=ob.groupby('departure_hour').size().reindex(range(24),fill_value=0).rename('pallets').reset_index();write(hours,out,'1c_hourly_departures')
    pickhours=ob.groupby('pick_hour').size().reindex(range(24),fill_value=0).rename('pallets').reset_index();write(pickhours,out,'1c_hourly_picks')
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(weekly.week_start,weekly.pallets);ax.set(xlabel='Week starting',ylabel='Pallets shipped',title='1(c) Weekly outbound volume');plot_save(out,'1c_weekly_demand')
    fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].bar(weekday['weekday'].str[:3],weekday['mean']);axs[0].set(title='Mean pallets per calendar weekday',ylabel='Pallets/day');axs[1].bar(hours.departure_hour,hours.pallets);axs[1].set(title='Pallets by scheduled truck departure hour',xlabel='Hour (24h)',ylabel='Pallets/year');plot_save(out,'1c_weekday_hour')
    # 1d sites, routes, trucks
    sitevol=sites.merge(ob.groupby('site_id').size().rename('pallets_year').reset_index(),on='site_id',how='left').fillna({'pallets_year':0})
    sitevol['pallets_week']=sitevol.pallets_year/52
    sitevol['delivery_weekdays']=sitevol.site_id.map(ob.groupby('site_id').weekday.apply(lambda s:', '.join(sorted(s.unique())))).fillna('none')
    write(sitevol,out,'1d_restaurant_profile')
    truckstats=trucks.copy();truckstats['weekday']=truckstats.ship_date.dt.day_name();truckstats['fill_pct']=100*truckstats.pallets/26
    check((truckstats.pallets<=26).all(),'truck exceeds 26 pallets')
    write(truckstats,out,'1d_truck_profile')
    route=truckstats.groupby(['route_id','weekday']).agg(trucks=('truck_id','size'),avg_pallets=('pallets','mean'),avg_stops=('stops','mean'),min_pallets=('pallets','min'),max_pallets=('pallets','max')).reset_index()
    write(route,out,'1d_route_profile')
    trucksday=truckstats.groupby('ship_date').agg(trucks=('truck_id','size'),pallets=('pallets','sum')).reindex(DAYS,fill_value=0).rename_axis('date').reset_index()
    write(trucksday,out,'1d_trucks_per_day')
    fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].hist(truckstats.pallets,bins=np.arange(.5,27.5,1));axs[0].set(xlabel='Pallets/truck (26 max)',ylabel='Trucks/year',title='1(d) Truck load distribution');axs[1].hist(truckstats.stops,bins=np.arange(.5,truckstats.stops.max()+1.5,1));axs[1].set(xlabel='Stops/truck',ylabel='Trucks/year',title='Stops per truck');plot_save(out,'1d_truck_loads')
    fig,ax=plt.subplots(figsize=(9,4));ax.hist(sitevol.pallets_week,bins=30);ax.set(xlabel='Pallets per restaurant per week',ylabel='Restaurant count',title='1(d) Restaurant demand distribution');plot_save(out,'1d_restaurant_demand')
    # 1e purchase orders and supplier volumes
    pos=ib.groupby(['po_id','supplier_id','receipt_date']).agg(pallets=('pallet_id','size'),distinct_skus=('sku_id','nunique'),doors=('door_id','nunique')).reset_index()
    write(pos,out,'1e_purchase_orders')
    supplier=sup.merge(ib.groupby('supplier_id').agg(pallets_received=('pallet_id','size'),purchase_orders=('po_id','nunique')).reset_index(),on='supplier_id',how='left').fillna({'pallets_received':0,'purchase_orders':0})
    supplier['pallets_received']=supplier.pallets_received.astype(int);supplier['purchase_orders']=supplier.purchase_orders.astype(int)
    supplier=supplier.sort_values('pallets_received',ascending=False);write(supplier,out,'1e_supplier_profile')
    ib['week_index']=((ib.receipt_date-START).dt.days//7)+1
    inweekly=ib.groupby('week_index').agg(pallets=('pallet_id','size'),orders=('po_id','nunique')).reindex(range(1,53),fill_value=0).rename_axis('week_index').reset_index()
    write(inweekly,out,'1e_weekly_receipts')
    night=ib.groupby('receipt_date').agg(pallets=('pallet_id','size'),orders=('po_id','nunique'),suppliers=('supplier_id','nunique')).reindex(DAYS,fill_value=0).rename_axis('date').reset_index()
    night['weekday']=night.date.dt.day_name();write(night,out,'1e_receiving_nights')
    fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].bar(supplier.head(15).supplier_id,supplier.head(15).pallets_received);axs[0].tick_params(axis='x',rotation=70);axs[0].set(title='1(e) Top 15 suppliers',ylabel='Pallets received');axs[1].hist(pos.pallets,bins=35);axs[1].set(title='Pallets per purchase order',xlabel='Pallets/order',ylabel='Orders');plot_save(out,'1e_supplier_orders')
    fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].plot(inweekly.week_index,inweekly.pallets);axs[0].set(title='Inbound pallets per week',xlabel='Week',ylabel='Pallets');recvday=night.groupby('weekday').pallets.mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']);axs[1].bar(recvday.index.str[:3],recvday);axs[1].set(title='Average pallets by receiving weekday',ylabel='Pallets/night');plot_save(out,'1e_receiving_pattern')
    # 1f end-of-day stock by SKU: opening stock exists before day shift on START; shipments before night receipts.
    ids=sku.sku_id.to_list();dates=DAYS
    startcounts=opening.groupby('sku_id').size().reindex(ids,fill_value=0)
    daily_in=ib.groupby(['receipt_date','sku_id']).size().unstack(fill_value=0).reindex(index=dates,columns=ids,fill_value=0)
    daily_out=ob.groupby(['ship_date','sku_id']).size().unstack(fill_value=0).reindex(index=dates,columns=ids,fill_value=0)
    stock=(daily_in-daily_out).cumsum(axis=0).add(startcounts,axis=1)
    check((stock>=0).to_numpy().all(),'negative SKU inventory at end of day')
    check(stock.iloc[-1].equals(closing.groupby('sku_id').size().reindex(ids,fill_value=0)),'SKU closing balance mismatch')
    # Day-shift picks precede night receipts: stock before receipts must cover all daily picks.
    pre_receipts=stock-daily_in
    check((pre_receipts>=0).to_numpy().all(),'daily picks exceed inventory available before night receipts')
    stock.index.name='date';stock.to_csv(out/'1f_daily_stock_by_sku.csv')
    total=stock.sum(axis=1)
    daily=pd.DataFrame({'date':dates,'opening_of_day_pallets':(total-daily_in.sum(axis=1)+daily_out.sum(axis=1)).to_numpy(), 'received_pallets':daily_in.sum(axis=1).to_numpy(),'shipped_pallets':daily_out.sum(axis=1).to_numpy(),'end_of_day_pallets':total.to_numpy()})
    daily['occupancy_pct']=100*daily.end_of_day_pallets/len(slots)
    write(daily,out,'1f_daily_warehouse_inventory')
    stockstats=pd.DataFrame({'sku_id':ids,'average_stock':stock.mean(axis=0).to_numpy(),'peak_stock':stock.max(axis=0).to_numpy(),'opening_stock':startcounts.to_numpy(),'closing_stock':stock.iloc[-1].to_numpy()})
    stockstats=stockstats.merge(velocity[['sku_id','pallets_shipped','velocity_class','rank']],on='sku_id')
    stockstats['days_of_shipments']=np.where(stockstats.pallets_shipped>0,stockstats.average_stock/(stockstats.pallets_shipped/364),np.nan)
    write(stockstats,out,'1f_sku_stock_statistics')
    write(stockstats[['sku_id','pallets_shipped','velocity_class','average_stock','peak_stock']],out,'TASK1_REQUIRED_sku_velocity_stock')
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(dates,total);ax.axhline(len(slots),ls='--',color='gray',label='Slot capacity');ax.set(xlabel='Date',ylabel='End-of-day pallets',title='1(f) Daily warehouse stock');ax.legend();plot_save(out,'1f_inventory_over_time')
    fig,ax=plt.subplots(figsize=(9,4));ax.hist(stockstats.average_stock,bins=40);ax.set(xlabel='Average pallets per SKU',ylabel='Number of SKUs',title='1(f) Distribution of average SKU stock');plot_save(out,'1f_sku_stock_distribution')
    # Extra pallet identity and FIFO checks: these are useful quality checks, not substitutes for Task 2.
    source=pd.concat([opening[['pallet_id','sku_id','received_date']],ib[['pallet_id','sku_id','receipt_date']].rename(columns={'receipt_date':'received_date'})],ignore_index=True)
    match=ob[['pallet_id','sku_id','received_date','ship_date']].merge(source,on='pallet_id',suffixes=('','_source'),validate='one_to_one')
    check(len(match)==len(ob),'missing pallet source')
    check((match.sku_id==match.sku_id_source).all() and (match.received_date==match.received_date_source).all(),'shipped pallet origin mismatch')
    check((match.received_date<match.ship_date).all(),'same-day or future-dated shipment')
    # Summary text
    annual=len(ob);weeks=52
    mean_stock=total.mean();peak_stock=total.max();avg_daily_ship=annual/364
    summary={
      'historical_days':len(DAYS),'weeks':weeks,'sites':len(sites),'skus':len(sku),'slots':len(slots),'clients':len(c),
      'annual_shipped_pallets':annual,'annual_received_pallets':len(ib),'opening_pallets':len(opening),'closing_pallets':len(closing),
      'avg_weekly_shipments':annual/weeks,'weekly_shipments_sd_sample':weekly.pallets.std(ddof=1),
      'weekly_shipments_cv_pct':100*weekly.pallets.std(ddof=1)/weekly.pallets.mean(),
      'peak_week':int(weekly.loc[weekly.pallets.idxmax(),'week_index']),'peak_week_pallets':int(weekly.pallets.max()),
      'peak_shipping_date':str(daily_ship.idxmax().date()),'peak_shipping_day_pallets':int(daily_ship.max()),
      'peak_departure_hour':int(hours.loc[hours.pallets.idxmax(),'departure_hour']),
      'truck_count':len(trucks),'mean_truck_load':trucks.pallets.mean(),'mean_truck_fill_pct':100*trucks.pallets.mean()/26,
      'purchase_orders':ib.po_id.nunique(),'purchase_orders_per_week':ib.po_id.nunique()/52,
      'mean_skus_per_po':pos.distinct_skus.mean(),'mean_pallets_per_po':pos.pallets.mean(),
      'mean_end_of_day_stock':mean_stock,'peak_end_of_day_stock':int(peak_stock),
      'mean_occupancy_pct':100*mean_stock/len(slots),'peak_occupancy_pct':100*peak_stock/len(slots),
      'mean_stock_days_of_shipments':mean_stock/avg_daily_ship,
      'mean_stock_days_using_shipping_days':mean_stock/(annual/(52*6)),
      'peak_receiving_date':str(night.loc[night.pallets.idxmax(),'date'].date()),'peak_receiving_pallets':int(night.pallets.max()),
    }
    pd.DataFrame([summary]).to_csv(out/'TASK1_summary_metrics.csv',index=False)
    with (out/'TASK1_findings.md').open('w') as f:
        f.write('# StackBay Task 1: computational findings\n\n')
        f.write('**Method:** 52 Monday–Sunday weeks, July 7 2025–July 5 2026. All weekly averages divide by 52. Inventory is end-of-calendar-day, after day picks and night receipts; average stock is the mean of 364 daily end balances. A=SKUs up to approximately 80% of cumulative shipped volume; B=next approximately 15%; C=remainder. Entire SKUs are assigned by the cumulative share *before* each SKU, so boundaries can slightly exceed 80% and 95%. Ties sorted by SKU ID. Hourly outbound demand is measured both at truck departure and at recorded pick time. No seasonality is assumed without statistical evidence.\n\n')
        for k,v in summary.items():f.write(f'- **{k}:** {v:,.3f}\n' if isinstance(v,(float,np.floating)) else f'- **{k}:** {v}\n')
        f.write('\n## 1(a) Clients ranked by annual shipments\n\n'+client[['client_name','sites','skus','pallets_shipped_year','pallets_per_week','share_of_shipped_pct']].to_markdown(index=False,floatfmt='.2f')+'\n')
        f.write('\n## 1(b) Velocity classes\n\n'+classes.to_markdown(index=False,floatfmt='.2f')+'\n')
        f.write('\n## 1(e) Leading suppliers\n\n'+supplier[['supplier_id','skus','lead_time_days','pallets_received','purchase_orders']].head(12).to_markdown(index=False)+'\n')
        f.write('\n## Notes and limitations\n\n- `days_of_shipments` for a SKU is undefined when annual shipments are zero; blank values are intentional.\n- Average and peak stock refer to daily **end-of-day** stock, not within-day peaks. Since outbound precedes inbound, nightly post-receipt stock is the end-of-day high point for that date.\n- The inventory coverage denominator uses all 364 calendar days; a separate 6-shipping-days/week measure is reported.\n- The distribution of truck loads is not proof of why a particular truck was underfilled; examine fixed route schedules, stop demand, and 26-pallet capacity before drawing causal conclusions.\n- The input data are synthetic; avoid inferring a seasonal pattern solely from weekly fluctuations.\n- This file provides computed evidence, not a finished three-page narrative.\n')
    with (out/'validation.txt').open('w') as f:f.write('PASS: all structural, cross-file, slot fit, pallet identity, inventory and truck reconciliations implemented in task1_analysis.py\n')
    print('SUCCESS: Task 1(a-f) completed. Output:',out)
    print(pd.Series(summary).to_string())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('outputs'));a=ap.parse_args();main(a.data,a.out)







