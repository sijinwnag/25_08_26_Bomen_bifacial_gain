# Weather Data Loading Module

This directory contains weather data and the associated loading module for the Bomen Solar Farm bifacial gain analysis project.

## Data Files

### `Bomen_weather_2021.csv`
- **Source**: Extracted from PVsyst comparison analysis project
- **Resolution**: 5-minute intervals
- **Period**: 2021-01-01 to 2021-12-31
- **Location**: Bomen Solar Farm, NSW, Australia
- **Variables**:
  - Air Temperature mean (Avg): Ambient temperature (°C)
  - Albedo Irradiance mean (Avg): Ground reflected irradiance (W/m²)
  - GHI Irradiance mean (Avg): Global Horizontal Irradiance (W/m²)
  - POA Irradiance mean (Avg): Plane of Array irradiance (W/m²)

## Usage

### Basic Loading
```python
from Code.weather_data_loader import load_bomen_weather_data

# Load complete weather dataset
weather_df = load_bomen_weather_data()
print(f"Loaded {len(weather_df)} records")
```

### Date Range Filtering
```python
# Load winter months only (June-August in Australia)
winter_df = load_bomen_weather_data(
    start_date='2021-06-01', 
    end_date='2021-08-31'
)
```

### CSV Export
```python
from Code.weather_data_loader import export_weather_data_csv

# Export processed weather data to CSV
export_path = export_weather_data_csv(weather_df, filename='weather_analysis.csv')
print(f"Exported to: {export_path}")

# Export with auto-generated filename based on date range
export_path = export_weather_data_csv(winter_df)  # Creates 'bomen_weather_data_2021-06-01_to_2021-08-31.csv'
```

### Load and Export in One Step
```python
from Code.weather_data_loader import load_and_export_weather_data

# Load specific date range and export to CSV simultaneously
weather_df, export_path = load_and_export_weather_data(
    start_date='2021-12-01',
    end_date='2021-12-31',
    filename='december_weather_2021.csv'
)
print(f"Loaded {len(weather_df)} records and exported to {export_path}")
```

### Data Validation
```python
from Code.weather_data_loader import validate_weather_data

validation = validate_weather_data(weather_df)
print(f"Validation status: {validation['status']}")
```

### Complete Example
```python
from Code.weather_data_loader import load_bomen_weather_data, validate_weather_data, print_weather_summary, export_weather_data_csv

# Load weather data
weather_data = load_bomen_weather_data()

# Print summary (matches original script output)
print_weather_summary(weather_data)

# Validate physical reasonableness
validation = validate_weather_data(weather_data)
if validation['status'] == 'PASS':
    print("Data ready for PV analysis!")

# Export to CSV for further analysis
export_path = export_weather_data_csv(weather_data)
print(f"Weather data exported to: {export_path}")
```

## Data Quality Notes

- **Completeness**: 92.4% (typical for real-world solar data)
- **Negative values**: Represent nighttime measurements (standard convention)
- **POA/GHI ratio**: 1.34 average (confirms single-axis tracking system)
- **Temperature range**: -2.9°C to 39.2°C (realistic for Australian conditions)
- **Peak irradiance**: GHI 1387 W/m², POA 1426 W/m² (within expected ranges)

## Physical Validation Results

The data has been validated by PV engineering analysis:
- ✓ Temperature data: Physically reasonable for Australian conditions
- ✓ Irradiance data: Within expected ranges for solar PV systems
- ✓ POA/GHI relationship: Consistent with single-axis tracking system behavior
- ✓ Overall assessment: Suitable for solar PV performance analysis

## Integration with Project

This weather data loading module integrates seamlessly with the existing bifacial gain analysis workflow:

1. **Maintenance filtering**: Use `../Code/maintenance_filter.py` to identify maintenance-free days
2. **Weather data loading**: Use this module to load corresponding weather measurements
3. **Performance analysis**: Combine with PV system data for bifacial gain analysis