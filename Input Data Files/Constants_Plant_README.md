# Constants_Plant_README.md

This file documents the parameters in Constants_Plant.csv used for the energy system simulation.

| Parameter                  | Description                                                      | Example Value         |
|----------------------------|------------------------------------------------------------------|----------------------|
| PERIOD_START               | Start of simulation period (YYYYMMDDHHMM, 24h format)            | 202405060000         |
| PERIOD_END                 | End of simulation period (YYYYMMDDHHMM, 24h format)              | 202405130000         |
| PV_OLD                     | Installed capacity of old PV system (kWp)                        | 1310                 |
| PV_NEW                     | Installed capacity of new PV system (kWp)                        | 1218                 |
| BESS_Capacity              | Battery energy storage system capacity (kWh)                      | 4472                 |
| BESS_limit                 | Minimum allowed SOC as fraction of capacity (unitless)            | 0.05                 |
| SOC_Initial                | Initial state of charge of BESS (kWh)                             | 2236                 |
| BESS_Power_Limit           | Maximum charge/discharge power of BESS (kW)                       | 1000                 |
| BESS_Efficiency_Charge     | BESS charge efficiency (fraction)                                 | 0.984                |
| BESS_Efficiency_Discharge  | BESS discharge efficiency (fraction)                              | 0.984                |
| LCOE_PV                    | Levelized cost of electricity for PV (Euro/kWh)                   | 0.05262              |
| LCOE_BESS                  | Levelized cost of electricity for BESS (Euro/kWh)                 | 0.10442              |
| BIDDING_ZONE               | ENTSO-E bidding zone code                                         | 10YCH-SWISSGRIDZ     |
| ENTSOE_TOKEN               | ENTSO-E API access token                                          | <token>              |
| TIMEZONE_OFFSET            | Local timezone offset from UTC (hours)                            | 2                    |
| Consumer_Price             | Price paid by consumer for electricity (Euro/kWh)                 | 0.12                 |

**Note:**
- All parameter names must match exactly between this file and the code.
- If you add new parameters, update this README and ensure the code references them correctly.
- Do not use inline comments in the CSV file; document here instead.
