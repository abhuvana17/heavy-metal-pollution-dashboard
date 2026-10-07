"""
generate_dataset.py
────────────────────────────────────────────────────────────────────
Synthetic dataset generator for the Heavy Metal Pollution Dashboard.
Generates exactly 10,000 records matching the schema in utils.py.

Run from inside pollution_dashboard/:
    python generate_dataset.py

Outputs:
    heavy_metal_pollution_dataset.csv   (in pollution_dashboard/)
────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N = 10_000

# ─── Geography: Indian states + districts ────────────────────────────────────
STATE_DISTRICTS = {
    "Maharashtra":   ["Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad", "Thane"],
    "Gujarat":       ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"],
    "Rajasthan":     ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner"],
    "Madhya Pradesh":["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Allahabad"],
    "West Bengal":   ["Kolkata", "Howrah", "Asansol", "Siliguri", "Durgapur"],
    "Tamil Nadu":    ["Chennai", "Coimbatore", "Madurai", "Trichy", "Salem"],
    "Karnataka":     ["Bengaluru", "Mysuru", "Hubli", "Mangaluru", "Belagavi"],
    "Andhra Pradesh":["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Tirupati"],
    "Telangana":     ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad", "Khammam"],
    "Jharkhand":     ["Dhanbad", "Ranchi", "Jamshedpur", "Bokaro", "Deoghar"],
    "Odisha":        ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur"],
    "Chhattisgarh":  ["Raipur", "Bhilai", "Bilaspur", "Korba", "Raigarh"],
    "Punjab":        ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
    "Haryana":       ["Gurugram", "Faridabad", "Ambala", "Panipat", "Rohtak"],
    "Bihar":         ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
    "Assam":         ["Guwahati", "Dibrugarh", "Silchar", "Jorhat", "Tinsukia"],
    "Kerala":        ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
    "Himachal Pradesh":["Shimla", "Manali", "Dharamsala", "Solan", "Mandi"],
    "Uttarakhand":   ["Dehradun", "Haridwar", "Nainital", "Mussoorie", "Rishikesh"],
}

states    = list(STATE_DISTRICTS.keys())
mediums   = ["Soil", "Water", "Air"]
land_uses = ["Industrial", "Agricultural", "Residential", "Mining", "Forest"]
pol_srcs  = ["Industrial Discharge", "Agricultural Runoff", "Mining Operations",
             "Vehicular Emissions", "Natural Weathering"]

# ─── Latitude / Longitude ranges per state (approximate) ─────────────────────
STATE_LATLON = {
    "Maharashtra":    (15.8, 22.0, 72.6, 80.9),
    "Gujarat":        (20.1, 24.7, 68.1, 74.5),
    "Rajasthan":      (23.0, 30.2, 69.4, 78.3),
    "Madhya Pradesh": (21.1, 26.9, 74.0, 82.8),
    "Uttar Pradesh":  (23.9, 30.4, 77.1, 84.6),
    "West Bengal":    (21.5, 27.2, 85.8, 89.9),
    "Tamil Nadu":     ( 8.1, 13.6, 77.0, 80.3),
    "Karnataka":      (11.6, 18.5, 74.1, 78.6),
    "Andhra Pradesh": (12.6, 19.9, 76.8, 84.7),
    "Telangana":      (15.8, 19.9, 77.3, 81.3),
    "Jharkhand":      (21.9, 25.3, 83.3, 87.9),
    "Odisha":         (17.8, 22.6, 81.4, 87.5),
    "Chhattisgarh":   (17.8, 24.1, 80.2, 84.4),
    "Punjab":         (29.5, 32.5, 73.9, 76.9),
    "Haryana":        (27.7, 30.9, 74.5, 77.6),
    "Bihar":          (24.3, 27.5, 83.3, 88.3),
    "Assam":          (24.1, 28.2, 89.7, 96.0),
    "Kerala":         ( 8.2, 12.8, 74.9, 77.4),
    "Himachal Pradesh":(30.4, 33.2, 75.6, 79.0),
    "Uttarakhand":    (28.7, 31.5, 77.6, 81.0),
}

rows = []
start_date = datetime(2020, 1, 1)
end_date   = datetime(2024, 12, 31)
date_range = (end_date - start_date).days

# Reference concentrations (background / permissible threshold) — project-defined
REF = {
    "Pb": 35.0,   # mg/kg
    "Hg": 0.30,   # mg/kg
    "As": 15.0,   # mg/kg
    "Cd": 1.00,   # mg/kg
    "Cr": 60.0,   # mg/kg
}

for i in range(N):
    state = random.choice(states)
    district = random.choice(STATE_DISTRICTS[state])
    lat_min, lat_max, lon_min, lon_max = STATE_LATLON[state]
    lat = round(random.uniform(lat_min, lat_max), 5)
    lon = round(random.uniform(lon_min, lon_max), 5)

    medium   = random.choice(mediums)
    land_use = random.choice(land_uses)
    pol_src  = random.choice(pol_srcs)

    # Date
    date = start_date + timedelta(days=random.randint(0, date_range))
    year = date.year

    # Environmental parameters (correlated with land-use/medium)
    dist_ind    = round(np.random.exponential(scale=25), 2)   # km
    temp_c      = round(np.random.normal(28, 8), 1)
    rainfall_mm = round(max(0, np.random.normal(800, 400)), 1)
    humidity    = round(np.clip(np.random.normal(65, 15), 10, 100), 1)
    soil_ph     = round(np.clip(np.random.normal(6.5, 1.2), 3.5, 9.5), 2)
    soil_moist  = round(np.clip(np.random.normal(35, 12), 5, 90), 1)

    # Pollution scaling factor driven by land-use + proximity to industry
    land_factor = {"Industrial": 2.5, "Mining": 2.8, "Agricultural": 1.5,
                   "Residential": 1.2, "Forest": 0.6}[land_use]
    dist_factor = max(0.3, 1 - dist_ind / 100)   # closer → more pollution
    base_scale  = land_factor * dist_factor

    # Heavy metal concentrations (mg/kg or µg/m³ for Air)
    pb  = round(max(0.1, np.random.lognormal(np.log(REF["Pb"]  * base_scale), 0.6)), 4)
    hg  = round(max(0.001, np.random.lognormal(np.log(REF["Hg"] * base_scale), 0.7)), 4)
    ars = round(max(0.1, np.random.lognormal(np.log(REF["As"]  * base_scale), 0.6)), 4)
    cd  = round(max(0.01, np.random.lognormal(np.log(REF["Cd"] * base_scale), 0.7)), 4)
    cr  = round(max(0.1, np.random.lognormal(np.log(REF["Cr"]  * base_scale), 0.6)), 4)

    # Contamination Factors (CF = conc / reference)
    cf_pb = pb / REF["Pb"]
    cf_hg = hg / REF["Hg"]
    cf_as = ars / REF["As"]
    cf_cd = cd / REF["Cd"]
    cf_cr = cr / REF["Cr"]

    # Indices
    hei = round(cf_pb + cf_hg + cf_as + cf_cd + cf_cr, 6)
    pli = round((cf_pb * cf_hg * cf_as * cf_cd * cf_cr) ** (1/5), 6)

    # HPI — weighted (project-defined weights = inverse of reference)
    weights  = [1/REF["Pb"], 1/REF["Hg"], 1/REF["As"], 1/REF["Cd"], 1/REF["Cr"]]
    concs_v  = [pb, hg, ars, cd, cr]
    total_w  = sum(weights)
    hpi      = round(sum(w * c for w, c in zip(weights, concs_v)) / total_w * 100, 6)

    # Risk level (PLI-based, project-defined)
    if pli < 1:
        risk = "Low"
    elif pli < 2:
        risk = "Moderate"
    elif pli < 3:
        risk = "High"
    else:
        risk = "Critical"

    rec_map = {
        "Low":      "Continue Monitoring",
        "Moderate": "Increase Monitoring Frequency",
        "High":     "Soil and Water Remediation",
        "Critical": "Immediate Government Intervention",
    }
    recommendation = rec_map[risk]

    rows.append({
        "Sample_ID":               f"SMP{i+1:06d}",
        "Date":                    date.strftime("%Y-%m-%d"),
        "State":                   state,
        "District":                district,
        "Latitude":                lat,
        "Longitude":               lon,
        "Medium":                  medium,
        "Land_Use":                land_use,
        "Distance_to_Industry_km": dist_ind,
        "Temperature_C":           temp_c,
        "Rainfall_mm":             rainfall_mm,
        "Humidity_percent":        humidity,
        "Soil_pH":                 soil_ph,
        "Soil_Moisture_percent":   soil_moist,
        "Lead_Pb_mgkg":            pb,
        "Mercury_Hg_mgkg":         hg,
        "Arsenic_As_mgkg":         ars,
        "Cadmium_Cd_mgkg":         cd,
        "Chromium_Cr_mgkg":        cr,
        "Pollution_Source":        pol_src,
        "HPI":                     hpi,
        "HEI":                     hei,
        "PLI":                     pli,
        "Risk_Level":              risk,
        "Recommendation":          recommendation,
        "Year":                    year,
        "Lead_Pb_CF":              round(cf_pb, 6),
        "Mercury_Hg_CF":           round(cf_hg, 6),
        "Arsenic_As_CF":           round(cf_as, 6),
        "Cadmium_Cd_CF":           round(cf_cd, 6),
        "Chromium_Cr_CF":          round(cf_cr, 6),
    })

df = pd.DataFrame(rows)
out = "heavy_metal_pollution_dataset.csv"
df.to_csv(out, index=False)

print(f"✅ Dataset generated: {out}")
print(f"   Records   : {len(df):,}")
print(f"   Columns   : {list(df.columns)}")
print(f"\nRisk distribution:")
print(df["Risk_Level"].value_counts())
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
