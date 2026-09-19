# Unseen public schemas

Published foreign keys are the ground truth; nothing in the system was tuned on these databases.

| database | tables | rows | published FKs found | extra links | merge root | root is a fact table | grain kept | lookup match rates correct | validation | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| Northwind | 8 | 3202 | 6/7 | 1 | order-details | yes | yes | 6/6 | passed | 4.2 |
| Chinook | 11 | 15607 | 9/10 | 1 | InvoiceLine | yes | yes | 8/8 | passed | 3.1 |
| Olist | 9 | 1550922 | 7/7 | 2 | order_items | yes | yes | 4/4 | passed | 70.0 |

## Northwind

* missed: orders.shipVia -> shippers.shipperID
* extra links: shippers - suppliers (entity_resolution)

## Chinook

* missed: Customer.SupportRepId -> Employee.EmployeeId
* extra links: Genre - Playlist (entity_resolution)

## Olist

* missed: none
* extra links: customers - geolocation (aggregate_lookup), geolocation - sellers (aggregate_lookup)
