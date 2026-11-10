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

clients_table_df["client"] = clients_table_df["client_name"].str.replace("Restaurants", "")
clients_table_df["pallets_wk"] = clients_table_df["pallets_yr"] / WEEKS