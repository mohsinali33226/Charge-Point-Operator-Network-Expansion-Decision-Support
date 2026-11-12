# German CPO Network Expansion Decision Support — Market Alignment Baseline

**Research cut-off:** 24 October 2026  
**Purpose:** Requirements and source-backed assumptions for an operator-configurable network expansion decision-support prototype. **Not** a claim of production readiness, technical approval, or validated financial returns.

## 1. Working rule: evidence before assumptions

Every consequential input must carry a source category, an as-of date, and validation status:

- **LAW_STANDARD:** Current binding legal/regulatory/technical requirement, with application conditions and effective dates.
- **VENDOR_SPEC:** Actual supported charger/power-unit/connector capabilities from manufacturer specifications.
- **PUBLIC_BENCHMARK:** Published market observation. Appropriate for plausibility checks or sensitivity bounds, *not* automatically an operator's actual value.
- **OPERATOR_DATA:** Site-specific operational, commercial, grid-connection and financial records.
- **ESTIMATE_QUOTE:** Named supplier or DSO quote, date, scope, currency and exclusions.
- **DEMO_ASSUMPTION:** Explicitly simulated or hypothesized value. Never present an investment recommendation based solely on this as verified.

Missing data must be recorded as **UNKNOWN**, not filled by a plausible-looking constant. Preserve source links, retrieval date, units and gross/net/VAT basis for all values.

## 2. October 2026 verified market anchors

| Topic | Verified information | Source |
|---|---|---|
| Public charging stock | BNetzA reports 157,762 normal and 56,572 fast public charging points as of 1 September 2026. Charge points are not synonymous with physical charger cabinets. | https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/start.html |
| Public end-user tariffs | UBA 2026 market assessment quotes €0.45–€0.70/kWh contract-based and €0.58–€0.79/kWh ad-hoc, including VAT, for its assessed providers. These are retail prices, **not CPO contribution margins**. | https://www.umweltbundesamt.de/system/files/medien/11850/publikationen/2026-06/P26_ZSA_3.Auflage.pdf |
| Advertised operator prices | IONITY's 12 February 2026 release listed €0.69/kWh Direct and €0.66/kWh Go at that time in Germany; tariffs change by date/channel. | https://www.ionity.eu/de/ionity/pressemitteilungen/ionity-senkt-preise-fur-spontanes-laden |
| Real-world usage dispersion | Q3 2026 status-derived analysis: ~24.5 million charging sessions at >175,000 observed German public points; large differences by location/operator and AC vs DC. Status transitions are not measured kWh and must not be treated as energy sales. | https://vision-mobility.de/en/news/chargepointradar-charging-infrastructure-enbw-dominates-but-ionity-uses-its-charging-points-more-intensively-394937.html |
| Charger architecture | Alpitronic HYC400 Series 2: configurable modular output, 2 DC connectors, dynamic power allocation; **actual connector limits/configuration are device-specific**. | https://www.alpitronic.it/en/hypercharger/hyc-400-series-2/ |
| Central power architecture | Kempower distributed charging: one power unit can allocate output dynamically among multiple satellites/outlets, hardware-configuration dependent. | https://kempower.com/solution/distributed-charging-systems/ |
| Grid connection | German connection and upgrade costs/terms depend on the grid operator and the connection; technical rules include VDE-AR-N 4110 for applicable medium-voltage connection. | https://www.bdew.de/energie/vnb-anwendungshilfe-ii-netzanschluss-ladesaeulen-mittelspannung/ ; https://www.vde.com/de/fnn/themen/tar/tar-mittelspannung-vde-ar-n-4110 |
| Metered electricity and bills | Eichrecht obligations for metering, checking measurements, verification and transparent billing; published 24 June 2026 guidance. | https://nationale-leitstelle.de/aktuelles-service/newshub/orientierungshilfe-zum-mess-und-eichrecht-in-der-ladeinfrastruktur/ |
| AFIR / data | Rules cover ad-hoc payment, transparency, and public static/dynamic charging information; national access point reporting and DATEX-II format from April 2026. Apply site- and date-specific triggers. | https://nationale-leitstelle.de/bestand-ausbau/afir/ ; https://www.bundesnetzagentur.de/DE/Vportal/Energie/E_Mobilitaet/_faq/OeffentlicheLadepunkte_table.html |
| Charge-management connectivity | OCPP 1.6 is widely deployed; OCPP 2.0.1 and 2.1 exist. OCPI 2.3.0 is current, but installed operators may use older versions. Protocol compatibility is *per vendor/integration*. | https://openchargealliance.org/protocols/open-charge-point-protocol/ ; https://ocpi-protocol.com/ |
| Demand planning | Germany's Nationale Leitstelle StandortTOOL provides geographic future-demand planning sources; use scenario-driven location-specific projections, not arbitrary fixed growth percentages. | https://nationale-leitstelle.de/tools-analysen/toolbox/ |

## 3. Real CPO network-expansion workflow to implement

1. **Network/site intake:** site identifiers, charger/EVSE/outlet mapping, geometry, site type, contracts and relevant DSO.
2. **Operational truth:** session CDRs, energy delivered, metered site power, charging curves, queue/abandonment where measurable, EVSE downtime/failed sessions, tariffs/roaming settlement. Distinguish measured values from inferences.
3. **Site demand and service KPIs:** completed sessions, turnaways or proxy bounds, customer wait, connector occupancy, energy utilization, peak-period congestion, charging speed, reliability, capacity constrained minutes; compare with peer sites.
4. **Future demand:** calibrated site-level forecasts with P50/P95, EV stock/local traffic, competition, seasonality, new nearby chargers and sensitivity cases. Do not simply multiply all demand by 15% every year.
5. **Technology-specific scenarios:** increase ports, cabinets, power modules, onsite storage, or supply; accept mixed vendors/capacities and true concurrent outlet limits. Expansion candidates must also respect physical site space and tenant permissions.
6. **Grid/constructability screening:** contracted import power (AC), conversion efficiency, existing protection/transformer/switchgear, DSO connection feasibility/offer, construction scope, power charges and expected timelines. Avoid treating nominal grid power as charger DC output or all costs as a universal €/kW.
7. **Customer impact simulation:** same arrivals for baseline and each scenario; time-dependent charging/vehicle acceptance and possible queue; count rejected, incomplete, delayed and successfully served separately; check power budgets at site, power-unit and outlet levels.
8. **Business case:** tariffs by channel net of VAT, roaming/eMSP share, energy procurement, losses, grid energy/demand charges, maintenance, availability, customer service, rent, land/parking, payment fees, hardware/civil works, service connection and financing. Date/site/source every quote and benchmark.
9. **Investment governance:** output expand / monitor / obtain quote / feasibility-blocked; financial sensitivity and lead time; site priorities across the portfolio. Do not equate positive simulation results with investment approval.
10. **Regulatory/construction gates:** AFIR, Eichrecht, BNetzA registrations, DSO technical conditions and planning/building permissions, accessibility and TEN-T obligations *when applicable*.

## 4. Data model (not hardcoded charger count)

- `network`: CPO/entity and reporting region.
- `site`: coordinates, usage class, opening hours, land/lease rights, grid contract, metering point, DSO, local constraints.
- `power_unit`: manufacturer/model, DC total output, AC input, module count, efficiency/derating and allocation capabilities.
- `evse/outlet`: physical location, concurrency, connector type, maximum DC power, current/voltage bounds; linkage to power unit.
- `meter`: grid and billed energy, signed values, time grain and provenance.
- `session`: start/end, site/outlet, metered kWh, connected vs charging time, billing method, failure status and observed peak power.
- `commercial_tariff`: ad-hoc/contract/roaming prices, VAT, effective dates, operator net settlement and fees.
- `site_cost`: actual quote lines, fixed/variable OPEX, grid charges, rent, CAPEX, construction costs and timelines.
- `demand_projection`: model version, location, horizon, quantiles, forecast backtest and scenario assumptions.
- `expansion_option`: add/replace equipment, associated new power modules, grid application/offer status and expected operational benefit.
- `decision`: KPIs, constraints, NPV/payback under sourced inputs, uncertainty and readiness status.

## 5. Existing project: specific gaps to fix before financial claims

- `generate_network_scenarios.py` defaults to +25%/+50%, minimum +2 chargers, assumed 95% efficiency, and grid upgrades rounded in 100 kW steps: **planning placeholders**, not universal German practice. Replace with configurable candidate strategies and DSO-validated possibilities.
- `simulate_expansion.py` requires exactly 2 concurrent connectors per charger; site charging power assumes identical chargers and flat car-requested kW. Vehicle energy acceptance, recharge curves, actual queues, defects and real power-unit constraints are not modeled. The model estimates rejected-car requested power and max stay, which are not measured for refused customers.
- `build_network_business_case.py` uses €0.20/kWh margin and €90k/charger + €30k site works + grid upgrade €50k + €150/kW, with €2k maintenance and 8% discount. These are *hypothetical*. Never describe its negative NPVs as evidence against real German investments.
- `evaluate_future_growth.py` assumes 15% yearly demand growth; **not** a sourced Germany market forecast or a substitute for location-specific forecasts.
- 5 synthetic sites are a *functional demonstration*; scalability requires empirical performance tests and commercial validity requires actual CPO data.

## 6. Decision quality gates

- **DEMO ONLY**: generated sessions/unsupported hardware or commercial assumptions.
- **TECHNICALLY SCREENED**: verified site hardware and electrical configuration; documented uncertainties in demand and operations.
- **FINANCIALLY SCREENED**: all material revenues/costs and terms sourced to operator data, publicly defensible benchmarks or real site quotes, with sensitivity analysis.
- **DSO / PERMIT PENDING**: grid or construction feasibility is not yet approved.
- **INVESTMENT READY FOR HUMAN REVIEW**: technical, financial, legal and DSO/permit gates passed; model provides evidence but does not grant approval.

**Implementation order:** (1) input schema + assumption source register + readiness gates; (2) configurable power-unit and EVSE topology simulation; (3) calibrated site demand + competition/real usage; (4) source-qualified finance and network ranking; (5) dashboard; (6) network-scale benchmarks and documentation.
