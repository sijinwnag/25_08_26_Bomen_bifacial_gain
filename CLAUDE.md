# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a **photovoltaic (PV) system maintenance analysis toolkit** for the Bomen Solar Farm bifacial gain research project at UNSW. The primary focus is on extracting maintenance-free operational days from system logbooks and processing weather data for accurate solar performance analysis.

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
The project implements a **dual-mode analysis framework** combining command-line tools for data preprocessing and Jupyter notebooks for interactive visualization. This approach supports both automated batch processing and exploratory data analysis workflows typical in solar energy research.

### Component Architecture

**Maintenance Filtering (`maintenance_filter.py`)**
- Command-line tool for extracting maintenance-free operational days
- Implements robust date parsing with multiple fallback strategies
- Features automatic column detection and inclusive date range filtering
- Designed for integration with larger solar performance analysis workflows

**Weather Data Processing (`weather_data_loader.py`)**
- Standalone module for loading and processing Bomen Solar Farm weather data
- Provides data validation and physical reasonableness checks
- Supports flexible date range filtering and CSV export functionality
- Compatible with existing solar PV analysis workflows

**Interactive Analysis (`25_08_26_Data_visualiser.ipynb`)**
- Jupyter notebook for exploratory data analysis and visualization
- Loads pre-processed solar farm data from optimized pickle files
- Supports both static and interactive plotting with comprehensive ML pipeline

### Key Components

**ProjectStructureManager Class** (`maintenance_filter.py`)
- Manages project directory structure and path resolution
- Automatically creates Results directory if missing
- Ensures consistent file paths across different execution contexts

**DateFormatDetector Class** (`maintenance_filter.py`)
- Handles automatic date format detection using multiple parsing strategies
- Implements fallback mechanisms: pandas → dateutil → manual format matching → regex parsing
- Supports 8 common date formats with 80% minimum success rate threshold

**MaintenanceFilter Class** (`maintenance_filter.py`)
- Main orchestrator for maintenance day filtering process
- Manages Excel/CSV file loading with automatic column detection
- Implements inclusive date range filtering (maintenance periods include start/end dates)
- Generates leap year-aware calendar calculations

### Data Processing Pipeline
```
Excel/CSV Input → Date Format Detection → Maintenance Event Parsing → Year Calendar Generation → Maintenance Day Filtering → Text File Output
```

### File Structure & Data Architecture
- **Code/**: Contains analysis scripts and Jupyter notebooks
  - `maintenance_filter.py`: CLI tool for maintenance day filtering from Excel/CSV logbooks
  - `weather_data_loader.py`: Weather data loading module with validation and export functionality
  - `25_08_26_Data_visualiser.ipynb`: Interactive data visualization and analysis notebook
- **Data/**: Contains maintenance logbooks and time-series datasets
  - **Maintenance Data**: Excel/CSV files with "Start Date" and "End Date" columns
    - `Faults 1.xlsx`: Default maintenance logbook input file
  - **Weather Data**: Raw weather measurements from Bomen Solar Farm
    - `weather data/Bomen_weather_2021.csv`: 5-minute resolution weather data (2021)
    - Contains air temperature, GHI, POA irradiance, and albedo measurements
  - **Time-Series Data**: Pre-processed pickle files for fast loading
    - `full_site_pow_5min.pkl`: Complete site electrical power (5-minute resolution)
    - `dtml_raw_5min.pkl`: Weather data (POA, Temperature, Wind, Power)
    - `full_inv_pow_5min.pkl`: Individual inverter power measurements
    - `met_pow_5min.pkl`: Meteorological measurements with power correlation
    - `wet_5min.pkl`: Additional weather measurements
- **Results/**: Analysis outputs (auto-created by scripts)
  - `maintenance_free_days_{year}.txt`: Lists of maintenance-free days
  - `bomen_weather_data_{date_range}.csv`: Processed weather data exports
- **Documentary/**: Project documentation and research materials
- **src/**: Contains claude-document-mcp server (development tool, not part of main analysis)

### Scientific Computing Stack
The project leverages a comprehensive scientific computing environment including:
- **Core Analysis**: pandas (2.2.3), numpy (2.2.0), scipy (1.15.2)
- **Visualization**: matplotlib (3.10.5), seaborn (0.12.2), plotly (5.22.0)
- **Machine Learning**: scikit-learn (1.6.1), torch (2.7.0), optuna (3.6.1), xgboost (2.0.3)
- **Solar Energy**: pvlib (0.11.1) - photovoltaic modeling library
- **Data Processing**: openpyxl (3.1.2), xlrd (2.0.1) for Excel file handling
- **Date/Time**: python-dateutil (2.9.0.post0) for flexible date parsing
- **Notebook Environment**: jupyter (1.1.1), jupyterlab (4.4.4), ipympl (0.9.7) for interactive plotting

### Error Handling Strategy
Both scripts implement **comprehensive error handling** with multiple validation layers:
- **File existence validation**: Checks for input files before processing
- **Date parsing resilience**: Multiple fallback strategies for different date formats
- **Data quality reporting**: Detailed logging of parsing success rates and sample data
- **Physical validation**: Weather data validation against expected solar energy ranges
- **Graceful degradation**: Continues processing even when some dates fail to parse

### Critical Implementation Details

**Maintenance Date Processing**: The maintenance filtering uses **inclusive date ranges** - both start and end dates of maintenance events are considered maintenance days. Year boundary logic correctly handles maintenance events spanning multiple years.

**Weather Data Processing**: The weather data loader filters to average measurements only (columns containing "(Avg )") to match PVsyst analysis workflows and provides physical reasonableness validation for solar energy applications.

## Development Patterns

### Logging Integration
Both Python modules use structured logging with timestamp and severity level formatting, providing operational transparency for debugging data quality issues typical in solar energy research environments.

### Command-Line Interface Design
The maintenance filter script uses argparse with sensible defaults:
- `--year` parameter is required (primary filter criteria)  
- `--excel_file` defaults to `../Data/Faults 1.xlsx` (relative to Code directory)
- `--output` auto-generates as `../Results/maintenance_free_days_{year}.txt`

### Data Quality Handling
The architecture anticipates real-world solar data quality issues:
- **Multiple parsing strategies**: Handles various Excel date formats automatically
- **Success rate validation**: Reports parsing effectiveness with sample data logging
- **Physical validation**: Weather data checked against solar energy physics constraints
- **Graceful degradation**: Processing continues even with partial parsing failures

## Workflow Integration Patterns

### Three-Phase Analysis Workflow
The architecture supports a **sequential analysis workflow** optimized for solar performance research:

**Phase 1: Maintenance Filtering**
```bash
cd Code
python maintenance_filter.py --year 2021
```
- Identifies maintenance-free operational days from logbooks
- Generates clean date lists for subsequent analysis
- Handles multiple years and custom logbook files

**Phase 2: Weather Data Processing** 
```bash
python weather_data_loader.py
# or import as module for programmatic use
```
- Loads and validates Bomen Solar Farm weather measurements
- Applies physical reasonableness checks and data quality validation
- Exports processed data for integration with other analysis tools

**Phase 3: Interactive Analysis**
```bash
jupyter lab 25_08_26_Data_visualiser.ipynb
```
- Loads pre-processed time-series data from optimized pickle files
- Performs exploratory data analysis with interactive visualizations
- Applies machine learning models for pattern recognition and anomaly detection

### Data Loading Patterns

**Pickle-Based Time Series Management**: All electrical power and weather data stored in optimized pickle format:
```python
# Standard data loading pattern in notebooks
site_power_df = pd.read_pickle("../Data/full_site_pow_5min.pkl")
weather_df = pd.read_pickle("../Data/dtml_raw_5min.pkl")
print(f"Loaded {len(site_power_df)} power records and {len(weather_df)} weather records")
```

**Weather Data Integration**: Weather data loader provides standardized interface:
```python
from weather_data_loader import load_bomen_weather_data, validate_weather_data
weather_data = load_bomen_weather_data(start_date='2021-06-01', end_date='2021-08-31')
validation = validate_weather_data(weather_data)
```

### Jupyter Notebook Analysis Patterns
**Interactive Solar Analysis Setup**:
- Uses `%matplotlib ipympl` for interactive time-series plotting
- Implements standardized plot styling optimized for solar data visualization
- Supports comprehensive ML pipeline integration with scikit-learn
- Designed for research reproducibility with clear data provenance tracking

This modular design enables **research reproducibility** essential for solar performance analysis, where data quality validation and processing transparency are critical for scientific validation of bifacial gain measurements and solar system performance assessment.