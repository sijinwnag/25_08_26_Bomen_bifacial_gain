# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Project Overview

This repository contains a **photovoltaic (PV) system maintenance analysis tool**. It is designed for research at UNSW, focusing on the Bomen solar farm bifacial gain analysis. The main purpose is to extract maintenance-free operational days from solar system logbooks for accurate performance analysis.

## Building and Running

The project is a single Python script and doesn't require a build process. To run the script, you need to have Python and the required dependencies installed.

### Dependencies

- pandas
- numpy
- python-dateutil
- openpyxl
- xlrd (for older .xls files)

### Installation

```bash
pip install pandas numpy python-dateutil openpyxl xlrd
```

### Running the script

The main script is `maintenance_filter.py` located in the `Code` directory.

#### Basic usage

```bash
cd Code
python maintenance_filter.py --year 2021
```

#### Custom file paths

```bash
cd Code
python maintenance_filter.py --year 2021 --excel_file "../Data/Faults 1.xlsx" --output "../Results/maintenance_free_days_2021.txt"
```

## Development Conventions

- The project follows a **monolithic Python script architecture**.
- The main script is `maintenance_filter.py`.
- It uses Python's `logging` module for logging.
- It uses `argparse` for command-line arguments.
- The code is designed to handle real-world data quality issues with features like automatic date format detection and fallback mechanisms.
- The maintenance filtering uses inclusive date ranges.
