# Graph experiment: direct matching vs graph-assisted routes

Relationships discovered by the system:

| left | right | join_kind | confidence |
|---|---|---|---|
| accounts | contacts | entity_key_merge | 0.9036 |
| accounts | orders | lookup | 0.7942 |

Dijkstra (cost = −log c): orders → accounts → contacts, reliability 0.718; direct A–C confidence: None.
Linear cost (1 − c): orders → accounts → contacts, reliability 0.718.

Linking each order to the correct contact:

| approach | linkable_orders | linked | correct | wrong | precision | recall |
|---|---|---|---|---|---|---|
| Direct A↔C (shared abbreviated names) | 3576 | 4000 | 0 | 4000 | 0.000 | 0.000 |
| Graph route A→B→C (system) | 3576 | 3402 | 3402 | 0 | 1.000 | 0.951 |

## Real-world counterpart (NYC)

No direct trips ↔ census relationship exists (direct confidence: None). Route found: yellow_tripdata_sample → taxi_zone_lookup → nta_demographics → census_acs_nyc_counties (reliability 0.491).

- yellow_tripdata_sample → taxi_zone_lookup: lookup (0.87) via taxi_zone_lookup::LocationID ↔ yellow_tripdata_sample::DOLocationID, taxi_zone_lookup::LocationID ↔ yellow_tripdata_sample::PULocationID
- taxi_zone_lookup → nta_demographics: aggregate_lookup (0.74) via nta_demographics::geographic_area_borough ↔ taxi_zone_lookup::Borough
- nta_demographics → census_acs_nyc_counties: lookup (0.76) via census_acs_nyc_counties::county ↔ nta_demographics::geographic_area_2010_census_fips_county_code