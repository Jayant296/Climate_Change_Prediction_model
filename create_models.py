import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

# Make sure models/ directory exists
os.makedirs("models", exist_ok=True)

# =====================================================
# LOAD NASA TEMPERATURE
# =====================================================
def load_temperature():
    url = "https://raw.githubusercontent.com/datasets/global-temp/master/data/annual.csv"
    df = pd.read_csv(url)
    df = df[["Year", "Mean"]]
    df.columns = ["Year", "Temp"]
    return df

# =====================================================
# LOAD CO₂
# =====================================================
def load_co2():
    url = "https://raw.githubusercontent.com/datasets/co2-ppm/master/data/co2-annmean-mlo.csv"
    df = pd.read_csv(url)
    df = df[["Year", "Mean"]]
    df.columns = ["Year", "CO2"]
    return df

# =====================================================
# LOAD SEA LEVEL
# =====================================================
def load_sea_level():
    url = "https://raw.githubusercontent.com/datasets/sea-level-rise/master/data/epa-sea-level.csv"
    df = pd.read_csv(url)
    df = df[["Year", "CSIRO Adjusted Sea Level"]]
    df.columns = ["Year", "SeaLevel"]
    return df

print("Loading datasets...")
temp = load_temperature()
co2 = load_co2()
sea = load_sea_level()

print("Merging...")
climate = temp.merge(co2, on="Year", how="outer")
climate = climate.merge(sea, on="Year", how="outer")
climate = climate.dropna()

def train_and_save(label, filename):
    X = climate[["Year"]]
    y = climate[label]
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    model.fit(X, y)
    joblib.dump(model, f"models/{filename}")
    print(f"Saved: models/{filename}")

train_and_save("Temp", "temp_model.pkl")
train_and_save("CO2", "co2_model.pkl")
train_and_save("SeaLevel", "sea_model.pkl")

print("\nAll models saved successfully!")
