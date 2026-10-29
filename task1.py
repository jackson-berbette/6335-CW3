"""
Casework 3 - StackBay Warehousing - Phase 1, Task 1: Characterize the business
================================================================================
Run from the project folder:      python task1.py            (data in ./data)
                         or:      python task1.py path/to/data
Outputs (tables .csv and charts .png) are written to ./outputs_task1/

Every row of outbound.csv / inbound.csv is ONE pallet (pallet_id is unique).
Historical year: 2025-07-07 (Mon) to 2026-07-05 (Sun) = 364 days = 52 weeks.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = "outputs_task1"
os.makedirs(OUT, exist_ok=True)

N_SLOTS = 8365            # from slots.csv (checked below)
TRUCK_CAP = 26            # brief: "each on its own route, with up to 26 pallets"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Velocity classes on pallets shipped per week (see 1b for the justification)
CLASS_CUTS = [("A", 5.0), ("B", 1.0), ("C", 0.2), ("D", 0.0)]   # lower bound, pallets/week


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150)
    plt.close(fig)


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string())


# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
out = pd.read_csv(f"{DATA}/outbound.csv",
                  parse_dates=["ship_date", "picked_at", "departs_at", "received_date"])
inb = pd.read_csv(f"{DATA}/inbound.csv", parse_dates=["receipt_date", "arrived_at", "putaway_at"])
trucks = pd.read_csv(f"{DATA}/trucks.csv", parse_dates=["ship_date", "departs_at"])
skus = pd.read_csv(f"{DATA}/skus.csv")
clients = pd.read_csv(f"{DATA}/clients.csv")
sites = pd.read_csv(f"{DATA}/client_sites.csv")
suppliers = pd.read_csv(f"{DATA}/suppliers.csv")
opening = pd.read_csv(f"{DATA}/opening_stock.csv", parse_dates=["received_date"])
current = pd.read_csv(f"{DATA}/current_stock.csv", parse_dates=["received_date"])
slots = pd.read_csv(f"{DATA}/slots.csv")

START, END = pd.Timestamp("2025-07-07"), pd.Timestamp("2026-07-05")
DAYS = pd.date_range(START, END, freq="D")
N_WEEKS = len(DAYS) / 7                                   # 52.0
assert len(slots) == N_SLOTS, "slot count differs from the brief"
assert out.pallet_id.is_unique and inb.pallet_id.is_unique
# Stock balance check: opening + receipts - shipments = closing stock
assert len(opening) + len(inb) - len(out) == len(current)

out["week"] = (out.ship_date - START).dt.days // 7 + 1
inb["week"] = (inb.receipt_date - START).dt.days // 7 + 1
out = out.merge(skus[["sku_id", "pallet_height_in", "pallet_weight_lb", "category"]], on="sku_id", how="left")
out = out.merge(sites[["site_id", "city", "state"]], on="site_id", how="left")

# ---------------------------------------------------------------------------
# 1a. Clients
# ---------------------------------------------------------------------------
cl = (out.groupby("client_id")
      .agg(pallets=("pallet_id", "size"),
           sites_served=("site_id", "nunique"),
           skus_shipped=("sku_id", "nunique"),
           states=("state", "nunique"))
      .join(clients.set_index("client_id")[["client_name", "sites", "skus"]]))
cl["pallets_per_week"] = cl.pallets / N_WEEKS
cl["pallets_per_site_week"] = cl.pallets_per_week / cl.sites
cl["share_pallets_%"] = 100 * cl.pallets / cl.pallets.sum()
cl = cl.sort_values("pallets", ascending=False)
cl["cum_share_%"] = cl["share_pallets_%"].cumsum()
stock_by_client = (pd.concat([opening, current]).merge(skus[["sku_id", "client_id"]], on="sku_id")
                   .groupby("client_id").size() / 2).rename("avg_stock_pallets_open_close")
cl = cl.join(stock_by_client)
cl = cl[["client_name", "sites", "skus", "skus_shipped", "states", "pallets", "pallets_per_week",
         "pallets_per_site_week", "share_pallets_%", "cum_share_%", "avg_stock_pallets_open_close"]]
cl.round(2).to_csv(f"{OUT}/1a_clients.csv")
show("1a Clients", cl.round(1))

geo = sites.groupby(["client_id", "state"]).size().unstack(fill_value=0)
geo.to_csv(f"{OUT}/1a_sites_by_state.csv")
show("1a Restaurants by state", geo)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(cl.client_name.str.replace(" Restaurants", ""), cl.pallets_per_week, color="#2563eb")
for i, v in enumerate(cl.pallets_per_week):
    ax.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=8)
ax.set_ylabel("Pallets shipped per week")
ax.set_title("1a. Pallets shipped per week by client")
save(fig, "1a_pallets_per_week_by_client")

fig, ax = plt.subplots(figsize=(7, 6))
for c, g in sites.groupby("client_id"):
    jit = np.random.default_rng(0).normal(0, 0.03, (len(g), 2))
    ax.scatter(g.lon + jit[:, 0], g.lat + jit[:, 1], s=8, alpha=0.6,
               label=clients.set_index("client_id").client_name[c].replace(" Restaurants", ""))
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("1a. Restaurant locations by client (nearest city)")
ax.legend(fontsize=7, markerscale=2)
save(fig, "1a_restaurant_map")

# ---------------------------------------------------------------------------
# 1b. SKUs: ranking, cumulative curve, velocity classes, heights and weights
# ---------------------------------------------------------------------------
sk = skus.set_index("sku_id").copy()
sk["pallets_shipped"] = out.groupby("sku_id").size().reindex(sk.index, fill_value=0)
sk["pallets_per_week"] = sk.pallets_shipped / N_WEEKS
sk = sk.sort_values("pallets_shipped", ascending=False)
sk["rank"] = np.arange(1, len(sk) + 1)
sk["cum_share_pallets"] = sk.pallets_shipped.cumsum() / sk.pallets_shipped.sum()


def velocity_class(ppw):
    for name, lo in CLASS_CUTS:
        if ppw >= lo:
            return name
    return CLASS_CUTS[-1][0]


sk["velocity_class"] = sk.pallets_per_week.apply(velocity_class)
sk["top5pct"] = sk["rank"] <= np.ceil(0.05 * len(sk))       # fastest 5% (needed in Task 3c)

cls = (sk.groupby("velocity_class")
       .agg(skus=("rank", "size"), pallets=("pallets_shipped", "sum"),
            min_ppw=("pallets_per_week", "min"), max_ppw=("pallets_per_week", "max"),
            mean_height_in=("pallet_height_in", "mean"), mean_weight_lb=("pallet_weight_lb", "mean")))
cls["share_skus_%"] = 100 * cls.skus / cls.skus.sum()
cls["share_pallets_%"] = 100 * cls.pallets / cls.pallets.sum()
cls.round(2).to_csv(f"{OUT}/1b_velocity_classes.csv")
show("1b Velocity classes (cut-offs on pallets/week: A>=5, B 1-5, C 0.2-1, D<0.2)", cls.round(2))

pts = {q: int((sk.cum_share_pallets <= q).sum()) for q in [0.5, 0.8, 0.95]}
print("\nSKUs needed for 50% / 80% / 95% of pallets:", pts,
      "| SKUs never shipped:", int((sk.pallets_shipped == 0).sum()))

fig, ax = plt.subplots(figsize=(8, 5))
x = 100 * sk["rank"] / len(sk)
ax.plot(x, 100 * sk.cum_share_pallets, color="#1e3a5f", lw=2)
colors = {"A": "#dc2626", "B": "#f59e0b", "C": "#10b981", "D": "#64748b"}
for c, g in sk.groupby("velocity_class"):
    ax.axvspan(100 * (g["rank"].min() - 1) / len(sk), 100 * g["rank"].max() / len(sk),
               color=colors[c], alpha=0.12, label=f"Class {c}: {len(g)} SKUs, "
               f"{100 * g.pallets_shipped.sum() / sk.pallets_shipped.sum():.0f}% of pallets")
ax.set_xlabel("% of SKUs (ranked by pallets shipped)")
ax.set_ylabel("Cumulative % of pallets shipped")
ax.set_title("1b. Cumulative pallets shipped vs SKUs, with velocity classes")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
save(fig, "1b_cumulative_curve")

# Heights and weights, counted per pallet shipped (what forklifts actually move)
hw = out.merge(sk[["velocity_class"]], left_on="sku_id", right_index=True)
hw_tab = hw.groupby("velocity_class")[["pallet_height_in", "pallet_weight_lb"]].describe(
    percentiles=[0.1, 0.5, 0.9]).round(0)
hw_tab.to_csv(f"{OUT}/1b_height_weight_by_class.csv")
show("1b Pallet height and weight per shipped pallet, by class", hw_tab)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for c in ["A", "B", "C", "D"]:
    g = hw[hw.velocity_class == c]
    axes[0].hist(g.pallet_height_in, bins=range(24, 92, 4), alpha=0.5, label=c, color=colors[c])
    axes[1].hist(g.pallet_weight_lb, bins=range(0, 2500, 100), alpha=0.5, label=c, color=colors[c])
axes[0].set_xlabel("Pallet height (in)"); axes[1].set_xlabel("Pallet weight (lb)")
for a in axes:
    a.set_ylabel("Pallets shipped"); a.legend(title="Class")
fig.suptitle("1b. Height and weight of shipped pallets by velocity class")
save(fig, "1b_height_weight")

# ---------------------------------------------------------------------------
# 1c. Demand over time
# ---------------------------------------------------------------------------
wk = out.groupby("week").size().reindex(range(1, 53), fill_value=0)
slope, intercept = np.polyfit(wk.index, wk.values, 1)
cv = wk.std() / wk.mean()
wow = wk.diff().abs()
print(f"\n=== 1c Weekly pallets: mean {wk.mean():.0f}, sd {wk.std():.0f}, CV {cv:.3f}, "
      f"min {wk.min()} (wk {wk.idxmin()}), max {wk.max()} (wk {wk.idxmax()}), "
      f"trend {slope:+.2f} pallets/week per week, mean |week-to-week change| {wow.mean():.0f} "
      f"(max {wow.max():.0f}), peak factor p95/mean {wk.quantile(0.95) / wk.mean():.2f}")
mon = out.groupby(out.ship_date.dt.to_period("M")).size()
ship_days = out.groupby(out.ship_date.dt.to_period("M")).ship_date.nunique()
mon_idx = (mon / ship_days) / (len(out) / out.ship_date.nunique())
show("1c Seasonality index by month (pallets per shipping day / annual average)", mon_idx.round(3).to_frame("index"))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wk.index, wk.values, marker="o", ms=3, label="Pallets shipped")
ax.plot(wk.index, wk.rolling(4, center=True).mean(), lw=2, label="4-week moving average")
ax.plot(wk.index, intercept + slope * wk.index, "--", label=f"Linear trend ({slope:+.1f}/week)")
ax.set_xlabel("Week of the historical year (week 1 starts Mon 7 Jul 2025)")
ax.set_ylabel("Pallets shipped"); ax.set_title("1c. Pallets shipped per week")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "1c_weekly")

daily = out.groupby("ship_date").size()
dow = daily.groupby(daily.index.day_name()).agg(["mean", "std", "max", "count"]).reindex(WEEKDAYS)
dow["share_%"] = 100 * out.ship_date.dt.day_name().value_counts().reindex(WEEKDAYS) / len(out)
show("1c Pallets shipped per day of week", dow.round(1))
rdaily = inb.groupby("receipt_date").size()
rdow = rdaily.groupby(rdaily.index.day_name()).agg(["mean", "std", "max", "count"]).reindex(WEEKDAYS)
show("1c Pallets received per night by day of week", rdow.round(1))

hr_out = out.groupby(out.picked_at.dt.hour).size() / out.ship_date.nunique()
hr_in = inb.groupby(inb.putaway_at.dt.hour).size() / inb.receipt_date.nunique()
hr_dep = trucks.groupby(trucks.departs_at.dt.hour).size() / trucks.ship_date.nunique()
show("1c Average pallets picked per hour (day shift)", hr_out.round(1).to_frame("pallets/h"))
show("1c Average pallets put away per hour (night shift)", hr_in.round(1).to_frame("pallets/h"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(dow.index.str[:3], dow["mean"], yerr=dow["std"], color="#2563eb", capsize=3)
axes[0].bar(rdow.index.str[:3], -rdow["mean"].fillna(0), color="#f59e0b")
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_ylabel("Pallets per day (shipped up, received down)")
axes[0].set_title("Day of week")
axes[1].bar(hr_out.index, hr_out.values, color="#2563eb", label="picked (day shift)")
axes[1].bar(hr_in.index, hr_in.values, color="#f59e0b", label="put away (night shift)")
axes[1].set_xlabel("Hour of day"); axes[1].set_ylabel("Average pallets per hour")
axes[1].set_title("Hour of day"); axes[1].legend()
fig.suptitle("1c. Flows by day of week and hour of day")
save(fig, "1c_day_hour")

# ---------------------------------------------------------------------------
# 1d. Restaurants, routes and trucks
# ---------------------------------------------------------------------------
site = out.groupby("site_id").agg(client_id=("client_id", "first"), pallets=("pallet_id", "size"),
                                  delivery_days=("ship_date", "nunique"),
                                  weekdays=("ship_date", lambda d: d.dt.day_name().nunique()))
site["pallets_per_week"] = site.pallets / N_WEEKS
site["deliveries_per_week"] = site.delivery_days / N_WEEKS
site["pallets_per_delivery"] = site.pallets / site.delivery_days
site.round(2).to_csv(f"{OUT}/1d_restaurants.csv")
show("1d Restaurants: distribution", site[["pallets_per_week", "deliveries_per_week", "pallets_per_delivery",
                                           "weekdays"]].describe().round(2))
show("1d Restaurants by number of delivery weekdays", site.weekdays.value_counts().sort_index().to_frame("sites"))
show("1d Restaurants per client: pallets per week (median)",
     site.groupby("client_id").pallets_per_week.describe()[["count", "50%", "mean", "max"]].round(2))

trucks["weekday"] = trucks.ship_date.dt.day_name()
trucks["fill_%"] = 100 * trucks.pallets / TRUCK_CAP
tpd = trucks.groupby("ship_date").size()
tdow = trucks.groupby("weekday").agg(trucks_per_day=("truck_id", "size"), pallets=("pallets", "mean"),
                                     stops=("stops", "mean"), fill=("fill_%", "mean")).reindex(WEEKDAYS)
tdow["trucks_per_day"] = tdow.trucks_per_day / trucks.groupby("weekday").ship_date.nunique().reindex(WEEKDAYS)
show("1d Trucks by weekday", tdow.round(1))
print(f"Routes: {trucks.route_id.nunique()} | trucks/day mean {tpd.mean():.1f} (min {tpd.min()}, max {tpd.max()})"
      f" | pallets/truck mean {trucks.pallets.mean():.1f} | stops/truck mean {trucks.stops.mean():.1f}"
      f" | average fill {trucks['fill_%'].mean():.0f}% | trucks <50% full: {100 * (trucks['fill_%'] < 50).mean():.1f}%")

# Why are some trucks nearly empty? Compare low vs high fill, and route-level stability.
per_truck = out.groupby("truck_id").agg(clients=("client_id", "nunique"), states=("state", "nunique"),
                                        sites=("site_id", "nunique"))
tt = trucks.set_index("truck_id").join(per_truck)
tt["fill_band"] = pd.cut(tt["fill_%"], [0, 50, 85, 100], labels=["<50%", "50-85%", ">85%"], include_lowest=True)
band = tt.groupby("fill_band", observed=True).agg(trucks=("pallets", "size"), pallets=("pallets", "mean"),
                                                  stops=("stops", "mean"), clients=("clients", "mean"),
                                                  states=("states", "mean"))
show("1d Low-fill vs full trucks", band.round(2))
rt = tt.groupby("route_id").agg(weekday=("weekday", "first"), runs=("pallets", "size"),
                                mean_pallets=("pallets", "mean"), sd_pallets=("pallets", "std"),
                                mean_stops=("stops", "mean"), max_pallets=("pallets", "max"))
rt["mean_fill_%"] = 100 * rt.mean_pallets / TRUCK_CAP
rt["sites_on_route"] = out.groupby("route_id").site_id.nunique()
rt["share_of_route_sites_per_run_%"] = 100 * rt.mean_stops / rt.sites_on_route
print(f"\nRoutes visit on average {rt.sites_on_route.mean():.1f} distinct restaurants over the year but "
      f"{rt.mean_stops.mean():.1f} stops per run ({rt['share_of_route_sites_per_run_%'].mean():.0f}%): "
      f"a restaurant on a route is not served every week, so the load of a fixed route varies week to week.")
rt.round(2).sort_values("mean_fill_%").to_csv(f"{OUT}/1d_routes.csv")
show("1d Emptiest and fullest routes", pd.concat([rt.sort_values("mean_fill_%").head(5),
                                                  rt.sort_values("mean_fill_%").tail(5)]).round(1))
print(f"Correlation pallets vs stops per truck: {tt.pallets.corr(tt.stops):.2f}")
print(f"Share of fill variance explained by route (between-route): "
      f"{1 - (tt.groupby('route_id').pallets.transform('var').mean() / tt.pallets.var()):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(trucks.pallets, bins=range(0, 28), color="#2563eb")
axes[0].axvline(TRUCK_CAP, color="r", ls="--", label="capacity 26")
axes[0].set_xlabel("Pallets per truck"); axes[0].set_ylabel("Trucks"); axes[0].legend()
axes[1].hist(trucks.stops, bins=range(0, 18), color="#10b981")
axes[1].set_xlabel("Stops per truck")
axes[2].scatter(tt.stops + np.random.default_rng(1).normal(0, 0.15, len(tt)), tt.pallets, s=4, alpha=0.3)
axes[2].set_xlabel("Stops per truck"); axes[2].set_ylabel("Pallets per truck")
fig.suptitle("1d. Outbound trucks: load, stops and their relationship")
save(fig, "1d_trucks")

# ---------------------------------------------------------------------------
# 1e. Supply
# ---------------------------------------------------------------------------
po = inb.groupby("po_id").agg(supplier_id=("supplier_id", "first"), date=("receipt_date", "first"),
                              skus=("sku_id", "nunique"), pallets=("pallet_id", "size"),
                              arrived=("arrived_at", "first"))
po["week"] = (po.date - START).dt.days // 7 + 1
pow_ = po.groupby("week").size()
print(f"\n=== 1e Purchase orders: {len(po)} in the year, {pow_.mean():.1f} per week (min {pow_.min()}, "
      f"max {pow_.max()}); SKUs per PO mean {po.skus.mean():.2f} (median {po.skus.median():.0f}); "
      f"pallets per PO mean {po.pallets.mean():.1f} (median {po.pallets.median():.0f}, max {po.pallets.max()})")
sup = (inb.groupby("supplier_id").agg(pallets=("pallet_id", "size"), pos=("po_id", "nunique"),
                                      skus_received=("sku_id", "nunique"),
                                      delivery_weekdays=("receipt_date", lambda d: d.dt.day_name().nunique()))
       .join(suppliers.set_index("supplier_id")).sort_values("pallets", ascending=False))
sup["share_%"] = 100 * sup.pallets / sup.pallets.sum()
sup["cum_share_%"] = sup["share_%"].cumsum()
sup.round(2).to_csv(f"{OUT}/1e_suppliers.csv")
show("1e Top 15 suppliers by pallets received", sup.head(15).round(1))
print(f"Suppliers for 50% / 80% of pallets: {(sup['cum_share_%'] <= 50).sum() + 1} / "
      f"{(sup['cum_share_%'] <= 80).sum() + 1} of {len(sup)}")
lt = np.repeat(sup.lead_time_days.values, sup.pallets.values)
print(f"Lead time (days): suppliers mean {suppliers.lead_time_days.mean():.1f}, range "
      f"{suppliers.lead_time_days.min()}-{suppliers.lead_time_days.max()}; pallet-weighted mean {lt.mean():.1f}")

night = inb.groupby("receipt_date").agg(pallets=("pallet_id", "size"), pos=("po_id", "nunique"),
                                        trucks=("door_id", "nunique"), first_arrival=("arrived_at", "min"),
                                        last_putaway=("putaway_at", "max"))
night["weekday"] = night.index.day_name()
night["finish_hours_after_16h"] = (night.last_putaway - night.index - pd.Timedelta(hours=16)).dt.total_seconds() / 3600
# Night-shift forklift hours from the modeling rules (put-away slot move + 1.8 min unload dock move)
inb["move_min"] = 2 * inb.travel_ft / 264 + 2 * inb.lift_ft / 60 + 1.5
night["putaway_hours"] = inb.groupby("receipt_date").move_min.sum() / 60
night["forklift_hours_incl_unload"] = night.putaway_hours + night.pallets * 1.8 / 60
night.round({c: 2 for c in night.select_dtypes("number").columns}).to_csv(f"{OUT}/1e_receiving_nights.csv")
show("1e Receiving nights by weekday", night.groupby("weekday")[["pallets", "pos", "putaway_hours",
                                                                 "forklift_hours_incl_unload",
                                                                 "finish_hours_after_16h"]]
     .agg(["mean", "max"]).reindex([d for d in WEEKDAYS if d in night.weekday.values]).round(1))
print(f"Pallets per night: mean {night.pallets.mean():.0f}, p95 {night.pallets.quantile(.95):.0f}, "
      f"max {night.pallets.max()} | forklift hours per night (put-away + unload): mean "
      f"{night.forklift_hours_incl_unload.mean():.1f}, max {night.forklift_hours_incl_unload.max():.1f} "
      f"-> forklifts needed over a 7.5 h shift (16:30-24:00), peak night: "
      f"{night.forklift_hours_incl_unload.max() / 7.5:.1f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(100 * np.arange(1, len(sup) + 1) / len(sup), sup["cum_share_%"].values, lw=2)
axes[0].set_xlabel("% of suppliers"); axes[0].set_ylabel("Cumulative % of pallets received")
axes[0].set_title("Supplier concentration"); axes[0].grid(alpha=0.3)
axes[1].hist(po.pallets, bins=40, color="#f59e0b")
axes[1].set_xlabel("Pallets per purchase order"); axes[1].set_ylabel("POs"); axes[1].set_title("PO size")
nd = night.groupby("weekday").pallets.mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
axes[2].bar(nd.index.str[:3], nd.values, color="#f59e0b")
axes[2].set_ylabel("Pallets received per night"); axes[2].set_title("Receiving nights")
fig.suptitle("1e. Supply")
save(fig, "1e_supply")

# ---------------------------------------------------------------------------
# 1f. Daily stock reconstruction (end of day, i.e. after that night's receipts)
#     S(t) = opening + receipts(date <= t) - shipments(date <= t)
# ---------------------------------------------------------------------------
sku_ids = skus.sku_id
rec = inb.pivot_table(index="receipt_date", columns="sku_id", values="pallet_id", aggfunc="size") \
         .reindex(index=DAYS, columns=sku_ids, fill_value=0).fillna(0)
shp = out.pivot_table(index="ship_date", columns="sku_id", values="pallet_id", aggfunc="size") \
         .reindex(index=DAYS, columns=sku_ids, fill_value=0).fillna(0)
open_s = opening.groupby("sku_id").size().reindex(sku_ids, fill_value=0)
stock = open_s + (rec - shp).cumsum()                                  # days x SKUs, end of day

neg = (stock < 0).sum().sum()
print(f"\n=== 1f Daily stock: negative SKU-days = {neg} (must be 0)")
close_check = (stock.iloc[-1] - current.groupby("sku_id").size().reindex(sku_ids, fill_value=0)).abs().sum()
print(f"Closing stock matches current_stock.csv: {close_check == 0}")

total = stock.sum(axis=1)
avg_ship_shipday = len(out) / out.ship_date.nunique()
avg_ship_calday = len(out) / len(DAYS)
summary = pd.DataFrame({
    "pallets": [total.mean(), total.max(), total.min()],
    "share_of_slots_%": [100 * total.mean() / N_SLOTS, 100 * total.max() / N_SLOTS, 100 * total.min() / N_SLOTS],
    "days_of_shipments_(per shipping day)": [total.mean() / avg_ship_shipday, total.max() / avg_ship_shipday,
                                            total.min() / avg_ship_shipday],
    "days_of_shipments_(per calendar day)": [total.mean() / avg_ship_calday, total.max() / avg_ship_calday,
                                            total.min() / avg_ship_calday]},
    index=["average", f"peak ({total.idxmax().date()})", f"minimum ({total.idxmin().date()})"])
summary.round(2).to_csv(f"{OUT}/1f_stock_summary.csv")
show("1f Total stock in the building (end of day)", summary.round(1))
print(f"Shipments per shipping day {avg_ship_shipday:.1f}, per calendar day {avg_ship_calday:.1f}")
total.to_frame("pallets_in_stock").to_csv(f"{OUT}/1f_daily_stock_total.csv")
stock.to_csv(f"{OUT}/1f_daily_stock_by_sku.csv")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(total.index, total.values, lw=1.5)
ax.axhline(N_SLOTS, color="r", ls="--", label=f"slots ({N_SLOTS})")
ax.axhline(total.mean(), color="gray", ls=":", label=f"average ({total.mean():.0f})")
ax.set_ylabel("Pallets in the building"); ax.set_ylim(0, N_SLOTS * 1.05)
ax.set_title("1f. Pallets in stock at the end of each day"); ax.legend()
save(fig, "1f_daily_stock")

cls_stock = stock.T.groupby(sk.velocity_class.reindex(stock.columns)).sum().T
fig, ax = plt.subplots(figsize=(10, 4))
ax.stackplot(cls_stock.index, [cls_stock[c] for c in ["A", "B", "C", "D"]],
             labels=[f"Class {c}" for c in "ABCD"], colors=[colors[c] for c in "ABCD"], alpha=0.8)
ax.set_ylabel("Pallets in stock"); ax.set_title("1f. Stock by velocity class"); ax.legend(loc="upper left")
save(fig, "1f_stock_by_class")
show("1f Average stock by class (pallets, % of stock)",
     pd.DataFrame({"avg_pallets": cls_stock.mean(), "share_%": 100 * cls_stock.mean() / total.mean()}).round(1))

# ---------------------------------------------------------------------------
# Required deliverable: velocity class and stock file, one row per SKU
# ---------------------------------------------------------------------------
vfile = pd.DataFrame({
    "sku_id": sk.index,
    "client_id": sk.client_id,
    "pallets_shipped": sk.pallets_shipped,
    "pallets_per_week": sk.pallets_per_week.round(3),
    "rank": sk["rank"],
    "velocity_class": sk.velocity_class,
    "top5pct_fastest": sk.top5pct,
    "avg_stock_pallets": stock.mean().reindex(sk.index).round(2),
    "peak_stock_pallets": stock.max().reindex(sk.index),
}).reset_index(drop=True)
vfile.to_csv(f"{OUT}/velocity_class_and_stock.csv", index=False)
print(f"\nWrote {OUT}/velocity_class_and_stock.csv ({len(vfile)} SKUs). "
      f"Sum of SKU peaks = {vfile.peak_stock_pallets.sum():.0f} pallets "
      f"({100 * vfile.peak_stock_pallets.sum() / N_SLOTS:.0f}% of slots) vs building peak {total.max():.0f}")
print("Done. All tables and charts are in", OUT)
