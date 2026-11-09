#!/usr/bin/env python3
#to run in vs code : python -m pip install pandas numpy matplotlib tabulate  (to install libraries)
#to run the code : python stackbay_task1_complete.py --data input_data --out outputs
"""StackBay Phase 1 Task 1(a-f). Run: python task1_analysis.py --data /path/to/csvs --out outputs"""
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
    # REPLACEMENT FOR TASKS 1(c), 1(d), 1(e) -- paste inside main().
    # Uses existing ob, trucks, sites, ib, sup, out, START, DAYS, write, plot_save.
    DATES = DAYS
    WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    CAPACITY = 26
    save = lambda df, output, name: write(df, output, name)
    chart = plot_save
    check(trucks.pallets.between(1, CAPACITY).all(), 'truck capacity violation')
    check(ib.groupby('po_id').supplier_id.nunique().max() == 1, 'PO maps to multiple suppliers')
    check(ib.groupby('po_id').receipt_date.nunique().max() == 1, 'PO appears on multiple receipt dates')

    def task_1c(ob, out):
        """Demand: weekly trend, 4-week smoothing, week-to-week variability, weekday and hourly peaks."""
        weekly = ob.groupby(((ob.ship_date-START).dt.days//7)+1).size().reindex(range(1,53),fill_value=0)
        wk = pd.DataFrame({'week':np.arange(1,53),'week_start':START+pd.to_timedelta(np.arange(52)*7,unit='D'),'pallets':weekly.to_numpy()})
        wk['moving_average_4wk'] = wk.pallets.rolling(4,min_periods=4).mean()
        slope, intercept = np.polyfit(wk.week, wk.pallets, 1)
        wk['linear_trend'] = intercept + slope*wk.week
        wk['week_over_week_change_pct'] = wk.pallets.pct_change()*100
        save(wk,out,'1c_weekly_trend')
        daily = ob.groupby('ship_date').size().reindex(DATES,fill_value=0).rename_axis('date').reset_index(name='pallets')
        daily['weekday']=daily.date.dt.day_name()
        daily['week']=((daily.date-START).dt.days//7)+1
        save(daily,out,'1c_daily_demand')
        weekdays=daily.groupby('weekday').pallets.agg(days='size',total='sum',mean='mean',peak='max',sd='std').reindex(WEEKDAYS).reset_index()
        save(weekdays,out,'1c_weekday_statistics')
        for col,name in [('picked_at','pick'),('departs_at','departure')]:
            hourly=ob.groupby(ob[col].dt.hour).size().reindex(range(24),fill_value=0).rename_axis('hour').reset_index(name='pallets')
            save(hourly,out,f'1c_hourly_{name}')
            if name=='pick': pickhour=hourly
            else: dephour=hourly
        # Week-of-year position is a diagnostic only: one year cannot prove annual seasonality.
        mid = 26
        first_half=float(wk.pallets.iloc[:mid].mean()); second_half=float(wk.pallets.iloc[mid:].mean())
        mean=float(wk.pallets.mean()); sd=float(wk.pallets.std(ddof=1)); cv=sd/mean
        fit=np.polyval([slope,intercept],wk.week)
        ss_total=((wk.pallets-mean)**2).sum()
        r2=1-((wk.pallets-fit)**2).sum()/ss_total if ss_total else np.nan
        metrics={'annual_pallets':len(ob),'weeks':52,'weekly_mean':mean,'weekly_sd':sd,'weekly_cv_pct':100*cv,
                 'weekly_min':int(wk.pallets.min()),'weekly_max':int(wk.pallets.max()),'peak_week':int(wk.loc[wk.pallets.idxmax(),'week']),
                 'linear_trend_pallets_per_week':float(slope),'linear_trend_r_squared':float(r2),
                 'first_26_weeks_mean':first_half,'last_26_weeks_mean':second_half,
                 'peak_shipping_date':str(daily.loc[daily.pallets.idxmax(),'date'].date()),'peak_shipping_day_pallets':int(daily.pallets.max()),
                 'peak_pick_hour':int(pickhour.loc[pickhour.pallets.idxmax(),'hour']),
                 'peak_departure_hour':int(dephour.loc[dephour.pallets.idxmax(),'hour'])}
        save(pd.DataFrame([metrics]),out,'1c_summary')
        fig,ax=plt.subplots(figsize=(11,5))
        ax.plot(wk.week_start,wk.pallets,label='Actual weekly pallets',alpha=.6)
        ax.plot(wk.week_start,wk.moving_average_4wk,label='4-week moving average',lw=2)
        ax.plot(wk.week_start,wk.linear_trend,label=f'Linear trend ({slope:+.2f} pallets/week)',ls='--')
        ax.set(xlabel='Week starting',ylabel='Pallets',title='1(c) Weekly demand, smoothing, and trend');ax.legend();chart(out,'1c_weekly_trend')
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        axs[0].bar(weekdays.weekday.str[:3],weekdays['mean']);axs[0].set(title='Average shipments per calendar weekday',ylabel='Pallets/day')
        axs[1].bar(pickhour.hour,pickhour.pallets,label='Picked',alpha=.7)
        axs[1].plot(dephour.hour,dephour.pallets,marker='o',label='Truck departures')
        axs[1].set(xlabel='Hour of day',ylabel='Annual pallets',title='Pallets by pick/departure hour');axs[1].legend()
        chart(out,'1c_weekday_and_hour')
        return metrics

    def task_1d(ob,tr,sites,out):
        """Routes/trucks: classify fill, link actual route and restaurant demand to low loads."""
        tr=tr.copy();tr['weekday']=tr.ship_date.dt.day_name()
        tr['fill_pct']=100*tr.pallets/CAPACITY
        tr['load_group']=np.select([tr.pallets<=13,tr.pallets>=23],['Low (1-13)','High (23-26)'],default='Middle (14-22)')
        # Actual unique restaurants served per truck, rather than assuming stop count equals site count.
        truck_sites=ob.groupby('truck_id').site_id.nunique().rename('unique_sites')
        tr=tr.join(truck_sites,on='truck_id')
        save(tr,out,'1d_truck_diagnostics')
        site=sites.copy().merge(ob.groupby('site_id').agg(annual_pallets=('pallet_id','size'),delivery_dates=('ship_date','nunique'),distinct_routes=('route_id','nunique')).reset_index(),on='site_id',how='left')
        site[['annual_pallets','delivery_dates','distinct_routes']]=site[['annual_pallets','delivery_dates','distinct_routes']].fillna(0).astype(int)
        site['pallets_per_week']=site.annual_pallets/52
        site['deliveries_per_week']=site.delivery_dates/52
        daymap=ob.assign(weekday=ob.ship_date.dt.day_name()).groupby('site_id').weekday.unique().map(lambda days:', '.join(x for x in WEEKDAYS if x in days))
        site['delivery_weekdays']=site.site_id.map(daymap).fillna('None')
        save(site,out,'1d_restaurant_diagnostics')
        route=tr.groupby(['route_id','weekday']).agg(truck_departures=('truck_id','size'),mean_pallets=('pallets','mean'),median_pallets=('pallets','median'),min_pallets=('pallets','min'),max_pallets=('pallets','max'),mean_stops=('stops','mean'),mean_unique_sites=('unique_sites','mean'),low_load_trucks=('load_group',lambda s:(s=='Low (1-13)').sum())).reset_index()
        route['low_load_share_pct']=100*route.low_load_trucks/route.truck_departures
        save(route.sort_values(['low_load_share_pct','truck_departures'],ascending=[False,False]),out,'1d_route_low_load_comparison')
        groups=tr.groupby('load_group').agg(trucks=('truck_id','size'),mean_pallets=('pallets','mean'),mean_stops=('stops','mean'),mean_sites=('unique_sites','mean'),mean_fill_pct=('fill_pct','mean')).reindex(['Low (1-13)','Middle (14-22)','High (23-26)']).reset_index()
        save(groups,out,'1d_load_group_comparison')
        day=tr.groupby('ship_date').agg(trucks=('truck_id','size'),pallets=('pallets','sum'),stops=('stops','sum')).reindex(DATES,fill_value=0).rename_axis('date').reset_index()
        day['weekday']=day.date.dt.day_name();save(day,out,'1d_daily_trucks')
        weekdays=day.groupby('weekday').agg(mean_trucks=('trucks','mean'),max_trucks=('trucks','max'),mean_pallets=('pallets','mean')).reindex(WEEKDAYS).reset_index()
        save(weekdays,out,'1d_trucks_by_weekday')
        # Explicitly distinguish observed associations from causal explanations.
        low=tr[tr.pallets<=13]; high=tr[tr.pallets>=23]
        metrics={'trucks':len(tr),'annual_pallets':int(tr.pallets.sum()),'mean_pallets_per_truck':float(tr.pallets.mean()),
                 'mean_fill_pct':float(tr.fill_pct.mean()),'mean_stops_per_truck':float(tr.stops.mean()),
                 'low_load_trucks_13_or_less':len(low),'low_load_share_pct':100*len(low)/len(tr),
                 'high_load_trucks_23_or_more':len(high),'high_load_share_pct':100*len(high)/len(tr),
                 'mean_stops_low_load':float(low.stops.mean()) if len(low) else np.nan,
                 'mean_stops_high_load':float(high.stops.mean()) if len(high) else np.nan,
                 'mean_sites_low_load':float(low.unique_sites.mean()) if len(low) else np.nan,
                 'mean_sites_high_load':float(high.unique_sites.mean()) if len(high) else np.nan,
                 'peak_trucks_date':str(day.loc[day.trucks.idxmax(),'date'].date()),'peak_trucks_per_day':int(day.trucks.max())}
        save(pd.DataFrame([metrics]),out,'1d_summary')
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        axs[0].hist(tr.pallets,bins=np.arange(.5,27.5,1));axs[0].axvline(13.5,ls='--',color='gray');axs[0].axvline(22.5,ls='--',color='gray')
        axs[0].set(title='Truck loads (maximum 26)',xlabel='Pallets/truck',ylabel='Number of trucks')
        axs[1].scatter(tr.stops,tr.pallets,alpha=.16,s=12)
        axs[1].set(title='Truck load vs stops',xlabel='Stops/truck',ylabel='Pallets/truck');chart(out,'1d_truck_load_vs_stops')
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        axs[0].bar(groups.load_group,groups.mean_stops);axs[0].tick_params(axis='x',rotation=15)
        axs[0].set(title='Mean stops by load group',ylabel='Stops/truck')
        axs[1].bar(weekdays.weekday.str[:3],weekdays.mean_trucks);axs[1].set(title='Average trucks by weekday',ylabel='Trucks/day')
        chart(out,'1d_route_load_context')
        return metrics

    def task_1e(ib,su,out):
        """Supply: PO distributions, suppliers, weekday receipts, peak-night putaway workload."""
        po=ib.groupby('po_id').agg(supplier_id=('supplier_id','first'),receipt_date=('receipt_date','first'),pallets=('pallet_id','size'),distinct_skus=('sku_id','nunique'),doors=('door_id','nunique')).reset_index()
        save(po,out,'1e_purchase_order_diagnostics')
        sup=su.merge(ib.groupby('supplier_id').agg(received_pallets=('pallet_id','size'),purchase_orders=('po_id','nunique'),distinct_receiving_days=('receipt_date','nunique')).reset_index(),on='supplier_id',how='left')
        for col in ['received_pallets','purchase_orders','distinct_receiving_days']:
            sup[col]=sup[col].fillna(0).astype(int)
        sup['volume_share_pct']=100*sup.received_pallets/len(ib)
        sup['pallets_per_week']=sup.received_pallets/52
        sup=sup.sort_values('received_pallets',ascending=False)
        save(sup,out,'1e_supplier_diagnostics')
        ib=ib.copy();ib['putaway_minutes']=2*ib.travel_ft/264 + 2*ib.lift_ft/60 + 1.5
        # Workload includes putaway slot moves + inbound trailer-to-staging moves.
        # Slot-move time follows casework formula; dock move = 1.8 min/pallet.
        night=ib.groupby('receipt_date').agg(pallets=('pallet_id','size'),orders=('po_id','nunique'),suppliers=('supplier_id','nunique'),slot_move_minutes=('putaway_minutes','sum')).reindex(DATES,fill_value=0).rename_axis('date').reset_index()
        night['weekday']=night.date.dt.day_name()
        night['putaway_hours']=night.slot_move_minutes/60
        night['unload_hours']=night.pallets*1.8/60
        night['combined_forklift_work_hours']=night.putaway_hours+night.unload_hours
        night['minimum_average_forklifts_for_7_5h_shift']=night.combined_forklift_work_hours/7.5
        save(night,out,'1e_receiving_night_workload')
        recv_week=ib.groupby(((ib.receipt_date-START).dt.days//7)+1).agg(pallets=('pallet_id','size'),orders=('po_id','nunique')).reindex(range(1,53),fill_value=0).rename_axis('week').reset_index()
        save(recv_week,out,'1e_weekly_receiving')
        weekdays=night.groupby('weekday').agg(calendar_days=('date','size'),receiving_nights=('pallets',lambda s:(s>0).sum()),mean_pallets=('pallets','mean'),peak_pallets=('pallets','max'),mean_forklift_work_hours=('combined_forklift_work_hours','mean')).reindex(WEEKDAYS).reset_index()
        save(weekdays,out,'1e_receiving_weekday_comparison')
        peak=night.loc[night.pallets.idxmax()]
        metrics={'annual_receipts':len(ib),'purchase_orders':len(po),'orders_per_week':len(po)/52,
                 'mean_pallets_per_order':float(po.pallets.mean()),'median_pallets_per_order':float(po.pallets.median()),
                 'mean_skus_per_order':float(po.distinct_skus.mean()),'median_skus_per_order':float(po.distinct_skus.median()),
                 'mean_receipts_per_week':float(recv_week.pallets.mean()),
                 'peak_receiving_date':str(peak.date.date()),'peak_receiving_pallets':int(peak.pallets),
                 'peak_night_combined_forklift_work_hours':float(peak.combined_forklift_work_hours),
                 'peak_night_minimum_average_forklifts_for_7_5h_shift':float(peak.minimum_average_forklifts_for_7_5h_shift),
                 'largest_supplier_id':str(sup.iloc[0].supplier_id),
                 'top_10_supplier_volume_share_pct':float(sup.head(10).volume_share_pct.sum()),
                 'supplier_lead_time_min_days':int(su.lead_time_days.min()),
                 'supplier_lead_time_max_days':int(su.lead_time_days.max())}
        save(pd.DataFrame([metrics]),out,'1e_summary')
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        axs[0].bar(sup.head(15).supplier_id,sup.head(15).received_pallets)
        axs[0].tick_params(axis='x',rotation=70);axs[0].set(title='Largest suppliers by receipts',ylabel='Pallets/year')
        axs[1].hist(po.pallets,bins=30);axs[1].set(title='Purchase order size distribution',xlabel='Pallets/PO',ylabel='PO count')
        chart(out,'1e_suppliers_and_orders')
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        axs[0].plot(night.date,night.pallets,alpha=.75);axs[0].set(title='Daily receiving volume',xlabel='Date',ylabel='Pallets/night')
        axs[1].bar(weekdays.weekday.str[:3],weekdays.mean_forklift_work_hours)
        axs[1].set(title='Mean inbound forklift work by weekday',ylabel='Forklift-hours/night')
        chart(out,'1e_receiving_night_and_workload')
        return metrics

    # Execute all three sections and retain variables needed by the original
    # Task 1 summary (below Task 1(f)).
    task_1c(ob, out)
    task_1d(ob, trucks, sites, out)
    task_1e(ib, sup, out)
    weekly = pd.read_csv(out / '1c_weekly_trend.csv').rename(columns={'week':'week_index'})
    daily_ship = ob.groupby('ship_date').size().reindex(DAYS, fill_value=0).rename('pallets')
    hours = pd.read_csv(out / '1c_hourly_departure.csv').rename(columns={'hour':'departure_hour'})
    pos = pd.read_csv(out / '1e_purchase_order_diagnostics.csv')
    night = pd.read_csv(out / '1e_receiving_night_workload.csv', parse_dates=['date'])
    supplier = pd.read_csv(out / '1e_supplier_diagnostics.csv').rename(columns={'received_pallets':'pallets_received'})

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
