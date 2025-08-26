# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a **photovoltaic (PV) system maintenance analysis tool** designed for research applications at UNSW, specifically focused on the Bomen solar farm bifacial gain analysis. The project centers around extracting maintenance-free operational days from solar system logbooks for accurate performance analysis.

## Essential Commands

### Main Script Execution (run from Code directory)
```bash
# Navigate to Code directory first
cd Code

# Basic usage - extract maintenance-free days for a specific year
python maintenance_filter.py --year 2021

# Custom Excel file and output location (paths relative to project root)
python maintenance_filter.py --year 2021 --excel_file "../Data/Faults 1.xlsx" --output "../Results/maintenance_free_days_2021.txt"

# Using different input file formats (Excel or CSV)
python maintenance_filter.py --year 2022 --excel_file "../Data/custom_maintenance_log.csv"
```

### Data Visualization Workflow (run from Code directory)
```bash
# Launch Jupyter for interactive data analysis
jupyter lab

# Or launch classic Jupyter notebook
jupyter notebook 25_08_26_Data_visualiser.ipynb

# Key notebook cells for data loading:
# - Electrical power data: full_site_pow_5min.pkl
# - Weather data: dtml_raw_5min.pkl  
# - Individual inverter data: full_inv_pow_5min.pkl
# - Meteorological data: met_pow_5min.pkl
```

### Weather Data Loading Workflow (run from Code directory)
```bash
# Load and validate Bomen weather data
python weather_data_loader.py

# Use as Python module for custom analysis
python -c "from weather_data_loader import load_bomen_weather_data; df = load_bomen_weather_data(); print(df.head())"

# Load specific date ranges for seasonal analysis
python -c "from weather_data_loader import load_bomen_weather_data; summer_df = load_bomen_weather_data(start_date='2021-12-01', end_date='2021-02-28')"
```

### Environment Setup
```bash
# Complete environment setup (recommended - includes all scientific computing dependencies)
pip install -r requirements.txt

# Minimal setup for maintenance filtering only
pip install pandas numpy python-dateutil openpyxl xlrd

# Jupyter environment for data visualization
pip install jupyter jupyterlab matplotlib seaborn scikit-learn

# Key scientific computing packages for solar analysis
pip install pvlib torch numpy pandas matplotlib scikit-learn
```

## Core Architecture

### Hybrid Analysis Architecture
The project implements a **dual-mode analysis framework** combining command-line tools for data preprocessing and Jupyter notebooks for interactive visualization and analysis. This hybrid approach supports both automated batch processing and exploratory data analysis workflows common in photovoltaic research.

### Component Architecture

**Maintenance Filtering (CLI Component)**
- Follows a monolithic Python script design in `maintenance_filter.py`
- Implements robust date parsing and validation for maintenance logbooks
- Designed for automated batch processing and integration into larger workflows

**Data Visualization (Interactive Component)**  
- Jupyter notebook-based analysis in `25_08_26_Data_visualiser.ipynb`
- Supports interactive exploration of electrical power and weather data
- Implements comprehensive scientific computing stack for advanced analysis

### Key Components

**DateFormatDetector Class**
- Handles automatic date format detection using multiple parsing strategies
- Implements fallback mechanisms: pandas auto-detection → dateutil parser → manual format matching → regex-based parsing
- Supports 8 common date formats with intelligent success rate thresholds (80% minimum)

**MaintenanceFilter Class**  
- Main orchestrator for the maintenance day filtering process
- Manages Excel/CSV file loading with automatic column detection
- Implements inclusive date range filtering (maintenance periods include start and end dates)
- Generates leap year-aware calendar calculations

### Data Processing Pipeline
```
Excel/CSV Input → Date Format Detection → Maintenance Event Parsing → Year Calendar Generation → Maintenance Day Filtering → Text File Output
```

### File Structure & Data Architecture
- **Code/**: Contains analysis scripts and Jupyter notebooks
  - `maintenance_filter.py`: CLI tool for maintenance day filtering
  - `weather_data_loader.py`: Standalone weather data loading module
  - `25_08_26_Data_visualiser.ipynb`: Interactive data visualization and analysis
- **Data/**: Contains both maintenance logbooks and time-series datasets
  - **Maintenance Data**: Excel/CSV files with "Start Date" and "End Date" columns
    - Default input: `Data/Faults 1.xlsx`
  - **Weather Data**: Raw weather measurements from Bomen Solar Farm
    - `weather data/Bomen_weather_2021.csv`: 5-minute resolution weather data
    - Air temperature, GHI, POA irradiance, albedo measurements (2021)
  - **Time-Series Data**: Pickle files containing pre-processed solar farm data
    - `full_site_pow_5min.pkl`: Complete site electrical power (5-minute resolution)
    - `dtml_raw_5min.pkl`: Weather data (POA, Temperature, Wind, Power)  
    - `full_inv_pow_5min.pkl`: Individual inverter power data
    - `met_pow_5min.pkl`: Meteorological measurements with power correlation
    - `wet_5min.pkl`: Additional weather measurements
- **Results/**: Directory for analysis outputs (auto-created if missing)
- **Documentary/**: Project documentation and research materials

### Scientific Computing Stack
The project leverages a comprehensive scientific computing environment with **370+ packages** including:
- **Core Analysis**: pandas, numpy, scipy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: torch, scikit-learn, optuna, xgboost
- **Solar Energy**: pvlib (photovoltaic modeling library)
- **Time Series**: various DTW and clustering algorithms
- **Notebook Environment**: jupyter, jupyterlab with interactive widgets

### Error Handling Strategy
The script implements **comprehensive error handling** with multiple validation layers:
- File existence validation before processing
- Date parsing with multiple fallback strategies  
- Null value handling and data quality reporting
- Detailed logging with configurable severity levels
- Graceful degradation when date parsing fails

### Date Processing Logic
**Critical Implementation Detail**: The maintenance filtering uses inclusive date ranges, meaning both start and end dates of maintenance events are considered as maintenance days. The year boundary logic correctly handles maintenance events that span across multiple years by calculating overlaps with the target year.

## Development Patterns

### Logging Integration
All operations use Python's logging module with timestamp and severity level formatting. Log levels provide operational transparency for debugging data quality issues common in research environments.

### Command-Line Interface
The script uses argparse for professional CLI interaction with sensible defaults:
- Year parameter is required (primary filter criteria)
- Excel file defaults to `Data/Faults 1.xlsx`
- Output file auto-generates as `maintenance_free_days_{year}.txt`

### Data Quality Handling
The architecture anticipates **real-world data quality issues**:
- Multiple date format detection strategies
- Success rate thresholds for parsing validation  
- Sample data logging for manual inspection when automatic parsing fails
- Comprehensive error messages for troubleshooting

## Workflow Integration Patterns

### Two-Phase Analysis Workflow
The architecture supports a **sequential analysis workflow** optimized for solar energy research:

**Phase 1: Data Preparation (CLI)**
- Filter maintenance periods using `maintenance_filter.py`
- Generate clean datasets with maintenance-free operational days
- Automated batch processing suitable for multiple years/datasets

**Phase 2: Interactive Analysis (Jupyter)**
- Load pre-processed time-series data from pickle files
- Perform exploratory data analysis with rich visualizations
- Apply machine learning models for pattern recognition and anomaly detection
- Export results and generate research-quality plots

### Data Loading Patterns
**Pickle-Based Data Management**: All time-series data is stored in optimized pickle format for fast loading and analysis:
```python
# Standard data loading pattern in notebooks
df = pd.read_pickle("../Data/full_site_pow_5min.pkl")
print(df.head())  # Verify data structure
df.plot(x='Timestamp', y='Power', figsize=(12, 6))  # Quick visualization
```

### Jupyter Notebook Execution Patterns
**Interactive Analysis Setup**:
- Uses `%matplotlib ipympl` for interactive plotting capabilities
- Implements standardized plot styling (axis_label_size=15, title_size=22)
- Supports both static and animated visualizations for time-series analysis
- Integrates comprehensive ML pipeline with sklearn ensemble methods

This hybrid tool design enables **research reproducibility** where data provenance and processing transparency are essential for scientific validation of solar system performance analysis, while supporting both automated processing and interactive exploration workflows.