"""
SONAR — Supply-chain Optimization and Network Analysis for Resilience.
Central configuration constants.
All magic strings, mappings, and numeric constants live here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DIR, "model_artifacts")

RAW_FILES = {
    "transport_cost":  os.path.join(RAW_DATA_DIR, "transport_cost_by_product.csv"),
    "bilateral_lsci":  os.path.join(RAW_DATA_DIR, "bilateral_shipping_connectivity_index.csv"),
    "country_lsci":    os.path.join(RAW_DATA_DIR, "country_shipping_connectivity_index.csv"),
    "port_throughput": os.path.join(RAW_DATA_DIR, "container_port_throughput.csv"),
    "merchant_fleet":  os.path.join(RAW_DATA_DIR, "merchant_fleet.csv"),
    "seaborne_trade":  os.path.join(RAW_DATA_DIR, "seaborne_trade.csv"),
    "vessel_pct":      os.path.join(RAW_DATA_DIR, "vessel_percent_of_global_fleet.csv"),
}

FEATURES_PATH     = os.path.join(PROCESSED_DIR, "features_long.parquet")
EDGES_PATH        = os.path.join(PROCESSED_DIR, "graph_edges_full.parquet")
MODEL_PATH        = os.path.join(ARTIFACTS_DIR, "xgb_model.json")
EXPLAINER_PATH    = os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl")
CONSTANTS_PATH    = os.path.join(ARTIFACTS_DIR, "normalization_constants.pkl")

# ─── Products ─────────────────────────────────────────────────────────────────
PRODUCT_CODES = [8517, 2106, 3304, 9404, 6109]

PRODUCT_NAMES = {
    8517: "Telephones & Electronics",
    2106: "Dried Food Preparations",
    3304: "Cosmetics & Toiletries",
    9404: "Mattresses & Household",
    6109: "Clothing (T-shirts etc.)",
}

YEARS = list(range(2016, 2022))          # 2016–2021 inclusive
LATEST_YEAR = 2021                       # single year used by the dashboard
TRAIN_YEARS = [2016, 2017, 2018, 2019]
VAL_YEARS   = [2020]
TEST_YEARS  = [2021]

# ─── Top trade corridors (pre-defined by seaborne trade importance) ────────────
# Used by Resilience Analysis heatmap and pre-computed at startup.
TOP_CORRIDORS = [
    ("China",               "United States"),
    ("China",               "Germany"),
    ("China",               "United Kingdom"),
    ("China",               "Japan"),
    ("China",               "Republic of Korea"),
    ("China",               "Australia"),
    ("China",               "India"),
    ("Japan",               "United States"),
    ("Republic of Korea",   "United States"),
    ("Germany",             "United States"),
    ("India",               "United States"),
    ("India",               "Germany"),
    ("United States",       "Germany"),
    ("United States",       "Japan"),
    ("Brazil",              "China"),
    ("Brazil",              "United States"),
    ("Australia",           "China"),
    ("Singapore",           "United States"),
    ("United Arab Emirates","India"),
    ("Saudi Arabia",        "China"),
]

# ─── Maritime waypoints for realistic ship-route visualization ────────────────
# (lat, lon) coordinates for key straits, canals, and ocean crossing points.
# Used by globe.py to draw routes that follow actual sea lanes instead of
# straight point-to-point lines across land.
MARITIME_WAYPOINTS = {
    # Southeast Asia
    "S_CHINA_SEA":   (12.0,  115.0),
    "MALACCA":       ( 1.5,  103.5),   # Strait of Malacca
    "JAVA_SEA":      (-6.0,  111.0),
    "LOMBOK":        (-8.5,  115.7),   # Lombok Strait (alt. to Malacca)
    "TIMOR_SEA":     (-11.0, 128.0),
    "CORAL_SEA":     (-18.0, 155.0),   # NE Australia
    "TASMAN_SEA":    (-37.0, 157.0),
    # Pacific
    "W_PACIFIC":     (25.0,  148.0),   # Western Pacific
    "NW_PACIFIC":    (40.0,  155.0),   # NW Pacific (Japan lanes)
    "N_PACIFIC":     (45.0, -170.0),   # North Pacific crossing
    "E_PACIFIC":     (10.0, -120.0),   # Eastern Pacific (US West Coast)
    "S_PACIFIC":     (-30.0,-130.0),
    "CAPE_HORN":     (-56.0, -68.0),
    # Indian Ocean
    "IND_OCEAN_N":   ( 5.0,   72.0),   # North Indian Ocean
    "IND_OCEAN_S":   (-25.0,  78.0),   # South Indian Ocean
    "ARABIAN_SEA":   (15.0,   62.0),
    "HORMUZ":        (26.5,   56.5),   # Strait of Hormuz
    "GULF_ADEN":     (12.0,   49.0),
    "BAB_EL":        (12.6,   43.5),   # Bab el-Mandeb
    "RED_SEA":       (20.0,   38.5),
    "SUEZ_S":        (29.9,   32.6),   # Suez (Red Sea end)
    "SUEZ_N":        (31.3,   32.3),   # Suez (Mediterranean end)
    "CAPE_GOOD_HOPE":(-34.4,  18.5),
    # Atlantic
    "S_ATLANTIC":    (-25.0,  -5.0),
    "N_ATLANTIC":    ( 35.0, -40.0),
    "CARIBBEAN":     ( 15.0, -68.0),
    "PANAMA_ATL":    (  9.4, -79.9),   # Panama Canal (Atlantic side)
    "PANAMA_PAC":    (  8.9, -79.5),   # Panama Canal (Pacific side)
    # Mediterranean & Northern Europe
    "GIBRALTAR":     (35.9,   -5.4),
    "MED_W":         (38.0,    5.0),   # Western Mediterranean
    "MED_E":         (35.5,   24.0),   # Eastern Mediterranean
    "ENGLISH_CH":    (50.0,   -2.0),   # English Channel
    "NORTH_SEA":     (55.0,    4.0),
    "BALTIC_SEA":    (57.0,   17.0),
}

# Undirected edges connecting maritime waypoints.
MARITIME_EDGES = [
    # Southeast Asia
    ("S_CHINA_SEA", "MALACCA"),
    ("S_CHINA_SEA", "W_PACIFIC"),
    ("S_CHINA_SEA", "JAVA_SEA"),
    ("JAVA_SEA",    "MALACCA"),
    ("JAVA_SEA",    "LOMBOK"),
    ("LOMBOK",      "IND_OCEAN_N"),
    ("LOMBOK",      "TIMOR_SEA"),
    ("TIMOR_SEA",   "CORAL_SEA"),
    ("TIMOR_SEA",   "IND_OCEAN_N"),
    ("CORAL_SEA",   "TASMAN_SEA"),
    ("CORAL_SEA",   "W_PACIFIC"),
    # Pacific
    ("W_PACIFIC",   "NW_PACIFIC"),
    ("NW_PACIFIC",  "N_PACIFIC"),
    ("N_PACIFIC",   "E_PACIFIC"),
    ("W_PACIFIC",   "E_PACIFIC"),       # equatorial Trans-Pacific
    ("E_PACIFIC",   "PANAMA_PAC"),
    ("S_PACIFIC",   "E_PACIFIC"),
    ("S_PACIFIC",   "TASMAN_SEA"),
    ("S_PACIFIC",   "CAPE_HORN"),
    ("CAPE_HORN",   "S_ATLANTIC"),
    # Indian Ocean
    ("MALACCA",     "IND_OCEAN_N"),
    ("IND_OCEAN_N", "ARABIAN_SEA"),
    ("IND_OCEAN_N", "GULF_ADEN"),
    ("IND_OCEAN_N", "IND_OCEAN_S"),
    ("IND_OCEAN_S", "CAPE_GOOD_HOPE"),
    ("ARABIAN_SEA", "HORMUZ"),
    ("ARABIAN_SEA", "GULF_ADEN"),
    ("GULF_ADEN",   "BAB_EL"),
    ("BAB_EL",      "RED_SEA"),
    ("RED_SEA",     "SUEZ_S"),
    ("SUEZ_S",      "SUEZ_N"),          # Suez Canal
    # Atlantic
    ("CAPE_GOOD_HOPE", "S_ATLANTIC"),
    ("S_ATLANTIC",  "N_ATLANTIC"),
    ("N_ATLANTIC",  "CARIBBEAN"),
    ("N_ATLANTIC",  "GIBRALTAR"),
    ("N_ATLANTIC",  "ENGLISH_CH"),
    ("CARIBBEAN",   "PANAMA_ATL"),
    ("PANAMA_ATL",  "PANAMA_PAC"),      # Panama Canal
    # Mediterranean & Europe
    ("SUEZ_N",      "MED_E"),
    ("MED_E",       "MED_W"),
    ("MED_W",       "GIBRALTAR"),
    ("MED_W",       "ENGLISH_CH"),      # Bay of Biscay
    ("GIBRALTAR",   "N_ATLANTIC"),
    ("ENGLISH_CH",  "NORTH_SEA"),
    ("NORTH_SEA",   "BALTIC_SEA"),
]

# Country → nearest maritime entry waypoint.
# Used to anchor each country in the waypoint graph for route rendering.
COUNTRY_PORT_WAYPOINT = {
    # East Asia
    "China":                        "S_CHINA_SEA",
    "Japan":                        "NW_PACIFIC",
    "Republic of Korea":            "NW_PACIFIC",
    "Taiwan":                       "S_CHINA_SEA",
    "Hong Kong, China":             "S_CHINA_SEA",
    "Macao, China":                 "S_CHINA_SEA",
    # Southeast Asia
    "Singapore":                    "MALACCA",
    "Malaysia":                     "MALACCA",
    "Indonesia":                    "JAVA_SEA",
    "Thailand":                     "MALACCA",
    "Viet Nam":                     "S_CHINA_SEA",
    "Philippines":                  "S_CHINA_SEA",
    "Cambodia":                     "S_CHINA_SEA",
    "Myanmar":                      "MALACCA",
    "Brunei Darussalam":            "S_CHINA_SEA",
    # South Asia
    "India":                        "ARABIAN_SEA",
    "Sri Lanka":                    "IND_OCEAN_N",
    "Bangladesh":                   "IND_OCEAN_N",
    "Pakistan":                     "ARABIAN_SEA",
    # Middle East
    "United Arab Emirates":         "HORMUZ",
    "Saudi Arabia":                 "RED_SEA",
    "Iran (Islamic Republic of)":   "HORMUZ",
    "Oman":                         "ARABIAN_SEA",
    "Kuwait":                       "HORMUZ",
    "Qatar":                        "HORMUZ",
    "Bahrain":                      "HORMUZ",
    "Iraq":                         "HORMUZ",
    "Yemen":                        "BAB_EL",
    "Djibouti":                     "BAB_EL",
    "Israel":                       "MED_E",
    "Jordan":                       "RED_SEA",
    "Egypt":                        "SUEZ_N",
    # East & Southern Africa
    "Kenya":                        "IND_OCEAN_N",
    "United Republic of Tanzania":  "IND_OCEAN_N",
    "Mozambique":                   "IND_OCEAN_S",
    "Madagascar":                   "IND_OCEAN_S",
    "Somalia":                      "GULF_ADEN",
    "Eritrea":                      "RED_SEA",
    "Sudan":                        "RED_SEA",
    "South Africa":                 "CAPE_GOOD_HOPE",
    "Namibia":                      "S_ATLANTIC",
    "Angola":                       "S_ATLANTIC",
    # West Africa
    "Nigeria":                      "S_ATLANTIC",
    "Ghana":                        "S_ATLANTIC",
    "Morocco":                      "GIBRALTAR",
    "Algeria":                      "MED_W",
    "Tunisia":                      "MED_W",
    "Libya":                        "MED_W",
    # Mediterranean Europe
    "Italy":                        "MED_W",
    "Spain":                        "GIBRALTAR",
    "Portugal":                     "GIBRALTAR",
    "France":                       "MED_W",
    "Greece":                       "MED_E",
    "Turkiye":                      "MED_E",
    "Malta":                        "MED_W",
    "Cyprus":                       "MED_E",
    # Northern Europe
    "Germany":                      "NORTH_SEA",
    "Netherlands (Kingdom of the)": "NORTH_SEA",
    "Belgium":                      "NORTH_SEA",
    "United Kingdom":               "ENGLISH_CH",
    "Denmark":                      "NORTH_SEA",
    "Sweden":                       "BALTIC_SEA",
    "Norway":                       "NORTH_SEA",
    "Finland":                      "BALTIC_SEA",
    "Poland":                       "BALTIC_SEA",
    "Latvia":                       "BALTIC_SEA",
    "Lithuania":                    "BALTIC_SEA",
    "Estonia":                      "BALTIC_SEA",
    "Ireland":                      "ENGLISH_CH",
    # Americas
    "United States":                "N_ATLANTIC",
    "Canada":                       "N_ATLANTIC",
    "Mexico":                       "CARIBBEAN",
    "Panama":                       "PANAMA_ATL",
    "Colombia":                     "CARIBBEAN",
    "Venezuela":                    "CARIBBEAN",
    "Cuba":                         "CARIBBEAN",
    "Jamaica":                      "CARIBBEAN",
    "Brazil":                       "S_ATLANTIC",
    "Argentina":                    "S_ATLANTIC",
    "Uruguay":                      "S_ATLANTIC",
    "Chile":                        "E_PACIFIC",
    "Peru":                         "E_PACIFIC",
    "Ecuador":                      "E_PACIFIC",
    # Pacific
    "Australia":                    "CORAL_SEA",
    "New Zealand":                  "TASMAN_SEA",
    "Papua New Guinea":             "CORAL_SEA",
}

# ─── Country name normalization (Transport-Cost → Bilateral canonical) ─────────
# Keys = names found in transport_cost_by_product.csv
# Values = matching names in bilateral_shipping_connectivity_index.csv
NAME_CANONICAL = {
    # Different ordering / phrasing
    "Korea, Republic of":              "Republic of Korea",
    "Korea, Dem. People's Rep. of":    "Dem. People's Rep. of Korea",
    "Congo, Dem. Rep. of the":         "Dem. Rep. of the Congo",
    "Moldova, Republic of":            "Republic of Moldova",
    "Tanzania, United Republic of":    "United Republic of Tanzania",
    "United States of America":        "United States",
    # The following are the same in both datasets, kept here for clarity
    "Iran (Islamic Republic of)":      "Iran (Islamic Republic of)",
    "Viet Nam":                        "Viet Nam",
    "Turkiye":                         "Turkiye",
}

# ─── Chokepoints ──────────────────────────────────────────────────────────────
# Maps display name → list of country node(s) to remove from graph when blocked
CHOKEPOINTS = {
    "Suez Canal":         ["Egypt"],
    "Panama Canal":       ["Panama"],
    "Strait of Hormuz":   ["Iran (Islamic Republic of)", "Oman"],
    "Strait of Malacca":  ["Singapore", "Malaysia", "Indonesia"],
}

# Maps display name → maritime waypoint key(s) for globe visualization.
# Used to pin the "blocked" X marker at the actual strait/canal location
# rather than at a country centroid.
CHOKEPOINT_WAYPOINTS = {
    "Suez Canal":        ["SUEZ_S", "SUEZ_N"],
    "Panama Canal":      ["PANAMA_ATL", "PANAMA_PAC"],
    "Strait of Hormuz":  ["HORMUZ"],
    "Strait of Malacca": ["MALACCA"],
}

# All chokepoint countries flattened (used by Resilience Score)
ALL_CHOKEPOINT_COUNTRIES = sorted(
    {c for countries in CHOKEPOINTS.values() for c in countries}
)  # 7 countries: Egypt, Indonesia, Iran (Islamic Republic of), Malaysia, Oman, Panama, Singapore

# ─── Tariff regions ───────────────────────────────────────────────────────────
TARIFF_REGIONS = {
    "United States":  ["United States"],
    "European Union": [
        "Austria", "Belgium", "Bulgaria", "Croatia", "Czechia", "Denmark",
        "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
        "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Netherlands (Kingdom of the)", "Poland", "Portugal", "Romania",
        "Slovakia", "Slovenia", "Spain", "Sweden",
    ],
    "China": ["China"],
    "ASEAN": [
        "Brunei Darussalam", "Cambodia", "Indonesia", "Lao People's Dem. Rep.",
        "Malaysia", "Myanmar", "Philippines", "Singapore", "Thailand",
        "Viet Nam",
    ],
}

# ─── Routing ──────────────────────────────────────────────────────────────────
# Real-world container shipping: direct (1 hop) or single transshipment hub
# (2 hops) covers >95% of all containerised trade lanes (UNCTAD 2023).
# Three or more intermediate ports are rare and only used for remote/niche
# destinations — so we try ≤2 hops first and fall back to ≤3 if needed.
MAX_HOPS          = 2     # preferred hop limit (direct or 1 transshipment hub)
MAX_HOPS_FALLBACK = 3     # fallback if no route found within MAX_HOPS
K_ROUTES          = 3     # number of alternative routes to return
MAX_WEIGHT        = 999.0 # sentinel weight for non-existent edges (not added to graph)

# ─── Resilience Score weights ─────────────────────────────────────────────────
# Derived via Analytic Hierarchy Process (AHP, Saaty 1980).
# Pairwise comparison matrix (scale 1–9, rows dominate columns):
#
#         Alt   Chk   Bil   Fleet
#  Alt  [  1     2     3     5  ]   Route redundancy is the primary resilience
#  Chk  [ 1/2    1     2     4  ]   driver (UNCTAD 2021, 2023 maritime reviews).
#  Bil  [ 1/3   1/2    1     3  ]   A corridor with no alternatives cannot
#  Fleet[ 1/5   1/4   1/3    1  ]   recover from a single disruption event.
#
# Chokepoint exposure (Chk) is second: 40 % of global container trade transits
# the Strait of Malacca; Suez handles ~15 % (UNCTAD, 2023).  Bilateral LSCI
# captures service quality/frequency.  Fleet is most fungible — vessels can be
# globally re-chartered — so it carries least weight.
#
# Consistency Ratio CR = 0.019 < 0.10  →  judgements are consistent (Saaty).
RS_WEIGHT_ALT   = 0.47   # route redundancy      (AHP priority vector)
RS_WEIGHT_CHK   = 0.28   # chokepoint avoidance
RS_WEIGHT_BIL   = 0.17   # bilateral connectivity
RS_WEIGHT_FLEET = 0.07   # fleet availability    (most fungible factor)
# Sum = 1.00 ✓

# ─── Country coordinates (lat/lon centroids for Plotly globe) ─────────────────
# Subset of ~200 countries; used by viz/globe.py
COUNTRY_COORDS = {
    "Afghanistan": (33.93911, 67.709953),
    "Albania": (41.153332, 20.168331),
    "Algeria": (28.033886, 1.659626),
    "Angola": (-11.202692, 17.873887),
    "Argentina": (-38.416097, -63.616672),
    "Armenia": (40.069099, 45.038189),
    "Australia": (-25.274398, 133.775136),
    "Austria": (47.516231, 14.550072),
    "Azerbaijan": (40.143105, 47.576927),
    "Bahamas": (25.03428, -77.39628),
    "Bahrain": (25.930414, 50.637772),
    "Bangladesh": (23.684994, 90.356331),
    "Belarus": (53.709807, 27.953389),
    "Belgium": (50.503887, 4.469936),
    "Belize": (17.189877, -88.49765),
    "Benin": (9.30769, 2.315834),
    "Bhutan": (27.514162, 90.433601),
    "Bolivia (Plurinational State of)": (-16.290154, -63.588653),
    "Bosnia and Herzegovina": (43.915886, 17.679076),
    "Botswana": (-22.328474, 24.684866),
    "Brazil": (-14.235004, -51.92528),
    "Brunei Darussalam": (4.535277, 114.727669),
    "Bulgaria": (42.733883, 25.48583),
    "Burkina Faso": (12.364566, -1.561593),
    "Burundi": (-3.373056, 29.918886),
    "Cambodia": (12.565679, 104.990963),
    "Cameroon": (3.848033, 11.502075),
    "Canada": (56.130366, -106.346771),
    "Central African Republic": (6.611111, 20.939444),
    "Chad": (15.454166, 18.732207),
    "Chile": (-35.675147, -71.542969),
    "China": (35.86166, 104.195397),
    "China, Hong Kong SAR": (22.319304, 114.169361),
    "China, Macao SAR": (22.198745, 113.543873),
    "Colombia": (4.570868, -74.297333),
    "Congo": (-0.228021, 15.827659),
    "Costa Rica": (9.748917, -83.753428),
    "Croatia": (45.1, 15.2),
    "Cuba": (21.521757, -77.781167),
    "Cyprus": (35.126413, 33.429859),
    "Czechia": (49.817492, 15.472962),
    "Dem. Rep. of the Congo": (-4.038333, 21.758664),
    "Dem. People's Rep. of Korea": (40.339852, 127.510093),
    "Denmark": (56.26392, 9.501785),
    "Djibouti": (11.825138, 42.590275),
    "Dominican Republic": (18.735693, -70.162651),
    "Ecuador": (-1.831239, -78.183406),
    "Egypt": (26.820553, 30.802498),
    "El Salvador": (13.794185, -88.89653),
    "Equatorial Guinea": (1.650801, 10.267895),
    "Estonia": (58.595272, 25.013607),
    "Eswatini": (-26.522503, 31.465866),
    "Ethiopia": (9.145, 40.489673),
    "Fiji": (-16.578193, 179.414413),
    "Finland": (61.92411, 25.748151),
    "France": (46.227638, 2.213749),
    "Gabon": (-0.803689, 11.609444),
    "Germany": (51.165691, 10.451526),
    "Ghana": (7.946527, -1.023194),
    "Greece": (39.074208, 21.824312),
    "Guatemala": (15.783471, -90.230759),
    "Guinea": (9.945587, -9.696645),
    "Guyana": (4.860416, -58.93018),
    "Haiti": (18.971187, -72.285215),
    "Honduras": (15.199999, -86.241905),
    "Hungary": (47.162494, 19.503304),
    "Iceland": (64.963051, -19.020835),
    "India": (20.593684, 78.96288),
    "Indonesia": (-0.789275, 113.921327),
    "Iran (Islamic Republic of)": (32.427908, 53.688046),
    "Iraq": (33.223191, 43.679291),
    "Ireland": (53.41291, -8.24389),
    "Israel": (31.046051, 34.851612),
    "Italy": (41.87194, 12.56738),
    "Jamaica": (18.109581, -77.297508),
    "Japan": (36.204824, 138.252924),
    "Jordan": (30.585164, 36.238414),
    "Kazakhstan": (48.019573, 66.923684),
    "Kenya": (-0.023559, 37.906193),
    "Kuwait": (29.31166, 47.481766),
    "Latvia": (56.879635, 24.603189),
    "Lebanon": (33.854721, 35.862285),
    "Lesotho": (-29.609988, 28.233608),
    "Libya": (26.3351, 17.228331),
    "Lithuania": (55.169438, 23.881275),
    "Luxembourg": (49.815273, 6.129583),
    "Madagascar": (-18.766947, 46.869107),
    "Malawi": (-13.254308, 34.301525),
    "Malaysia": (4.210484, 101.975766),
    "Mali": (17.570692, -3.996166),
    "Malta": (35.937496, 14.375416),
    "Mauritius": (-20.348404, 57.552152),
    "Mexico": (23.634501, -102.552784),
    "Moldova, Republic of": (47.411631, 28.369885),
    "Morocco": (31.791702, -7.09262),
    "Mozambique": (-18.665695, 35.529562),
    "Myanmar": (21.913965, 95.956223),
    "Namibia": (-22.95764, 18.49041),
    "Nepal": (28.394857, 84.124008),
    "Netherlands (Kingdom of the)": (52.132633, 5.291266),
    "New Zealand": (-40.900557, 174.885971),
    "Nicaragua": (12.865416, -85.207229),
    "Nigeria": (9.081999, 8.675277),
    "Norway": (60.472024, 8.468946),
    "Oman": (21.512583, 55.923255),
    "Pakistan": (30.375321, 69.345116),
    "Panama": (8.537981, -80.782127),
    "Papua New Guinea": (-6.314993, 143.95555),
    "Paraguay": (-23.442503, -58.443832),
    "Peru": (-9.189967, -75.015152),
    "Philippines": (12.879721, 121.774017),
    "Poland": (51.919438, 19.145136),
    "Portugal": (39.399872, -8.224454),
    "Qatar": (25.354826, 51.183884),
    "Republic of Korea": (35.907757, 127.766922),
    "Republic of Moldova": (47.411631, 28.369885),
    "Romania": (45.943161, 24.96676),
    "Russian Federation": (61.52401, 105.318756),
    "Rwanda": (-1.940278, 29.873888),
    "Saudi Arabia": (23.885942, 45.079162),
    "Senegal": (14.497401, 14.452362),
    "Serbia": (44.016521, 21.005859),
    "Sierra Leone": (8.460555, -11.779889),
    "Singapore": (1.352083, 103.819836),
    "Slovakia": (48.669026, 19.699024),
    "Slovenia": (46.151241, 14.995463),
    "Somalia": (5.152149, 46.199616),
    "South Africa": (-30.559482, 22.937506),
    "South Sudan": (6.876991, 31.306978),
    "Spain": (40.463667, -3.74922),
    "Sri Lanka": (7.873054, 80.771797),
    "Sudan": (12.862807, 30.217636),
    "Sweden": (60.128161, 18.643501),
    "Switzerland, Liechtenstein": (46.818188, 8.227512),
    "Syrian Arab Republic": (34.802075, 38.996815),
    "Tajikistan": (38.861034, 71.276093),
    "Tanzania, United Republic of": (-6.369028, 34.888822),
    "Thailand": (15.870032, 100.992541),
    "Togo": (8.619543, 0.824782),
    "Trinidad and Tobago": (10.691803, -61.222503),
    "Tunisia": (33.886917, 9.537499),
    "Turkiye": (38.963745, 35.243322),
    "Uganda": (1.373333, 32.290275),
    "Ukraine": (48.379433, 31.16558),
    "United Arab Emirates": (23.424076, 53.847818),
    "United Kingdom": (55.378051, -3.435973),
    "United Republic of Tanzania": (-6.369028, 34.888822),
    "United States": (37.09024, -95.712891),
    "United States of America": (37.09024, -95.712891),
    "Uruguay": (-32.522779, -55.765835),
    "Uzbekistan": (41.377491, 64.585262),
    "Venezuela (Bolivarian Rep. of)": (6.42375, -66.58973),
    "Viet Nam": (14.058324, 108.277199),
    "Yemen": (15.552727, 48.516388),
    "Zambia": (-13.133897, 27.849332),
    "Zimbabwe": (-19.015438, 29.154857),
}

# ─── ML feature list ──────────────────────────────────────────────────────────
ML_FEATURES = [
    "origin_lsci", "dest_lsci", "bilateral_lsci",
    "origin_teu", "dest_teu", "teu_log_product",
    "origin_fleet_pct", "dest_fleet_pct", "fleet_supply",
    "origin_loaded_kt", "dest_loaded_kt",
    "origin_discharged_kt", "dest_discharged_kt",
    "trade_imbalance", "lsci_asymmetry",
    "origin_vessel_pct", "dest_vessel_pct",
    "historical_mean_rate",
    "year_int", "product_cat", "post_covid",
    "bilateral_lsci_is_imputed", "origin_teu_is_imputed",
    "dest_teu_is_imputed", "historical_mean_rate_is_imputed",
]
