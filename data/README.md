# Input data and provenance

Required CSV: `data/input/market_prices.csv` with columns `date`, `nifty50`, `bank_nifty`, `india_vix`. Dates must be unique and prices/index levels positive where observed. These are daily closing observations, not intraday prices.

The accompanying input is a local aligned extract attributed in its coverage notes to:

- NIFTY Indices historical index data: https://www.niftyindices.com/reports/historical-data
- NSE historical India VIX: https://www.nseindia.com/reports-indices-historical-vix

The extract is restricted to dates through 31 December 2025. Its checksum and input coverage are recorded in the results manifest. Source attribution comes from the supplied coverage notes; original exchange downloads are not independently reverified in this analysis.

No gold or government-bond inputs enter this project. India VIX is an implied volatility index; it is not the same quantity as historical volatility calculated from returns.

Missing values are not forward-filled. Union-calendar rows without NIFTY quotes are removed; missing auxiliary quotes remove the affected feature observations. HMM transitions therefore represent consecutive valid feature observations, not calendar days. The pipeline reports all these exclusions.

Input price files are excluded from Git. The downloadable project ZIP includes the local input for reproducibility. A clone of the Git bundle requires placing a suitable CSV at the path above. No price-data redistribution license is asserted by the code license.
