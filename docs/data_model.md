# Charging Session Data Model

## Purpose

This project uses session-level charging data as the foundation for forecasting demand and planning future charging-site capacity.

Each row represents one charging session.

## Core session fields

- session_id
- site_id
- charger_id
- connector_id
- start_time
- end_time
- energy_kwh
- max_power_kw
- charger_type

## Site-level fields

- site_id
- city
- latitude
- longitude
- number_of_connectors
- charger_power_kw
- grid_capacity_kw
- site_type
- commissioning_date

## Future external features

- temperature
- precipitation
- holiday
- traffic intensity
- nearby POIs
- nearby chargers
- competitor density
- population density
- motorway distance-