# Data Files

This folder contains training data for the hot water demand forecasting challenge.

## Files

| File | Size | Description |
|------|------|-------------|
| `sensors.zip` | 17 MB | Sensor time series data |
| `weather_forecasts.zip` | 28 MB | Weather forecast data |
| `weather_observations.zip` | 94 MB | Weather observation data |

---

## sensors.zip

Contains `sensor_timeseries.csv` - hourly hot water flow readings from 45 sensors.

**Date range**: 2020-01-01 to 2024-12-31

**Columns**:

| Column | Description |
|--------|-------------|
| `CTime` | Timestamp (hourly) |
| `M01` - `M44` | Individual sensor readings (flow in m³/h) |
| `FRAMRENNSLI_TOTAL` | Total network flow (m³/h) |

**Notes**:
- 43,848 hourly observations
- 45 sensor columns total
- Values represent flow rates in cubic meters per hour

---

## weather_forecasts.zip

Contains `weather_forecasts.csv` - weather forecasts from multiple stations.

**Date range**: 2022-06-08 to 2024-12-31

**Columns**:

| Column | Description |
|--------|-------------|
| `date_time` | Forecast target time (what time is being predicted) |
| `station_id` | Weather station identifier |
| `temperature` | Temperature forecast (°C) |
| `windspeed` | Wind speed forecast (m/s) |
| `cloud_coverage` | Cloud coverage (%) |
| `gust` | Wind gust forecast (m/s) |
| `humidity` | Relative humidity (%) |
| `winddirection` | Wind direction (compass) |
| `dewpoint` | Dew point temperature (°C) |
| `rain_accumulated` | Accumulated rainfall (mm) |
| `value_date` | When the forecast was issued |

**Notes**:
- ~6 million forecast records
- Multiple weather stations in the Reykjavik area
- Forecasts range from 1 to 240 hours ahead (up to 10 days)
- `date_time` is the time being forecasted, `value_date` is when the forecast was made

---

## weather_observations.zip

Contains `weather_observations.csv` - actual weather observations from multiple stations.

**Date range**: 2020-01-01 to 2024-12-31

**Columns**:

| Column | Description |
|--------|-------------|
| `stod` | Weather station identifier |
| `timi` | Observation timestamp |
| `f` | Wind speed (m/s) |
| `fg` | Wind gust (m/s) |
| `fsdev` | Wind speed standard deviation |
| `d` | Wind direction (degrees) |
| `dsdev` | Wind direction standard deviation |
| `t` | Temperature (°C) |
| `tx` | Maximum temperature (°C) |
| `tn` | Minimum temperature (°C) |
| `rh` | Relative humidity (%) |
| `td` | Dew point temperature (°C) |
| `p` | Atmospheric pressure (hPa) |
| `r` | Precipitation (mm) |
| `tg` | Ground temperature (°C) |
| `tng` | Minimum ground temperature (°C) |

**Notes**:
- ~4 million observation records
- Multiple weather stations
- Observations typically every 10 minutes to 1 hour depending on station

---

## Usage

```python
import pandas as pd
import zipfile

# Extract and load sensor data
with zipfile.ZipFile('sensors.zip', 'r') as z:
    with z.open('sensor_timeseries.csv') as f:
        sensors = pd.read_csv(f, index_col=0, parse_dates=True)

# Extract and load weather forecasts
with zipfile.ZipFile('weather_forecasts.zip', 'r') as z:
    with z.open('weather_forecasts.csv') as f:
        forecasts = pd.read_csv(f)
        forecasts['date_time'] = pd.to_datetime(forecasts['date_time'])

# Extract and load weather observations
with zipfile.ZipFile('weather_observations.zip', 'r') as z:
    with z.open('weather_observations.csv') as f:
        observations = pd.read_csv(f)
        observations['timi'] = pd.to_datetime(observations['timi'])
```
