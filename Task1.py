import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


# read relevant input files
outbound_df = pd.read_csv("Data/outbound.csv")
clients_df = pd.read_csv("Data/clients.csv")
client_sites_df = pd.read_csv("Data/client_sites.csv")

WEEKS = 52 

# -------- Part a --------
# pallets shipped per client over the year
client_pallets = (
    outbound_df.groupby("client_id")
    .size()
    .reset_index(name="pallets_yr")
)

clients_table_df = (
    clients_df.rename(columns={"sites": "stores"})
    .merge(client_pallets, on="client_id", how="left")
)

clients_table_df["client"] = clients_table_df["client_name"].str.replace(" Restaurants", "")
clients_table_df["pallets_wk"] = clients_table_df["pallets_yr"] / WEEKS
clients_table_df["share"] = clients_table_df["pallets_wk"] / clients_table_df["pallets_wk"].sum()
clients_table_df["pallets_per_store_wk"] = clients_table_df["pallets_wk"] / clients_table_df["stores"]


# where the restaurants are: sattes holding at least 5% of the client's stores, with store counts
def main_states(states): 
    counts = states.value_counts()
    return ", ".join(
        f"{state} {n}" for state, n in counts.items() if n >= 0.05 * len(states)
    )

client_states = (
    client_sites_df.groupby("client_id")["state"]
    .agg(main_states)
    .reset_index(name="states")
)

clients_table_df = (
    clients_table_df.merge(client_states, on="client_id", how="left")
    .sort_values("pallets_wk", ascending=False)
    .reset_index(drop=True)
)

clients_table_df = clients_table_df[
    ["client_id", "client", "stores", "skus", "pallets_wk", "share", "pallets_per_store_wk", "states"]
]

# total row
total_row = pd.DataFrame([{
    "client_id": "Total",
    "client": "Total",
    "stores": clients_table_df["stores"].sum(),
    "skus": clients_table_df["skus"].sum(),
    "pallets_wk": clients_table_df["pallets_wk"].sum(),
    "share": 1.0,
    "pallets_per_store_wk": clients_table_df["pallets_wk"].sum() / clients_table_df["stores"].sum(),
    "states": f"{client_sites_df['state'].nunique()} states"
}])

clients_table_df = pd.concat([clients_table_df, total_row], ignore_index=True)

# validate
assert len(clients_df) == 9
assert np.isclose(clients_table_df["pallets_wk"].iloc[:-1].sum() * WEEKS, len(outbound_df))

display(clients_table_df.round(3))

top4_share = clients_table_df["share"].iloc[:4].sum()
bottom3_share = clients_table_df["share"].iloc[6:9].sum()
bottom3_skus = clients_table_df["skus"].iloc[6:9].sum()

print(f"Top 4 clients ship {top4_share:.1%} of pallets")
print(f"Bottom 3 clients ship {bottom3_share:.1%} with {bottom3_skus} SKUs")

clients_table_df.to_csv("Output/Task1a_Clients.csv", index=False)