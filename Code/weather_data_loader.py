#!/usr/bin/env python3
"""
Weather Data Loading Module for Bomen Solar Farm Analysis

This module provides functionality to load and process weather data from the Bomen Solar Farm,
replicating the exact workflow from the original PVsyst comparison analysis.

Features:
- Loads Bomen weather data with proper datetime indexing
- Filters to average measurements only (columns containing "(Avg )")
- Provides data validation and physical reasonableness checks
- Supports flexible date range filtering for analysis
- Exports processed data to CSV files in the Results directory
- Compatible with solar PV analysis workflows

Usage:
    from weather_data_loader import (load_bomen_weather_data, validate_weather_data, 
                                    export_weather_data_csv, load_and_export_weather_data)
    
    # Load weather data
    weather_df = load_bomen_weather_data()
    
    # Validate physical reasonableness
    validation_results = validate_weather_data(weather_df)
    
    # Load with date filtering
    weather_df_filtered = load_bomen_weather_data(
        start_date='2021-06-01', 
        end_date='2021-08-31'
    )
    
    # Export to CSV
    export_path = export_weather_data_csv(weather_df, filename='weather_analysis.csv')
    
    # Load and export in one step
    weather_df, export_path = load_and_export_weather_data(
        start_date='2021-12-01', 
        end_date='2021-12-31',
        filename='december_weather.csv'
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeatherDataLoader:
    """
    Weather data loading and processing class for Bomen Solar Farm data.
    
    Replicates the exact workflow from the original PVsyst comparison analysis
    for loading and processing measured weather data.
    """
    
    def __init__(self, data_directory=None, results_directory=None):
        """
        Initialize weather data loader.
        
        Args:
            data_directory (str, optional): Path to weather data directory.
                                          Defaults to '../Data/weather data' relative to script location.
            results_directory (str, optional): Path to results output directory.
                                             Defaults to '../Results' relative to script location.
        """
        # Setup data directory
        if data_directory is None:
            # Default path relative to Code directory
            script_dir = Path(__file__).parent
            self.data_directory = script_dir.parent / "Data" / "weather data"
        else:
            self.data_directory = Path(data_directory)
        
        # Setup results directory
        if results_directory is None:
            # Default path relative to Code directory  
            script_dir = Path(__file__).parent
            self.results_directory = script_dir.parent / "Results"
        else:
            self.results_directory = Path(results_directory)
        
        self.weather_file_path = self.data_directory / "Bomen_weather_2021.csv"
        
        logger.info(f"Weather data directory: {self.data_directory}")
        logger.info(f"Results directory: {self.results_directory}")
        logger.info(f"Weather file path: {self.weather_file_path}")
        
        # Validate file exists
        if not self.weather_file_path.exists():
            raise FileNotFoundError(f"Weather data file not found: {self.weather_file_path}")
        
        # Create results directory if it doesn't exist
        self._ensure_results_directory()
    
    def _ensure_results_directory(self):
        """
        Ensure the results directory exists, create if necessary.
        """
        try:
            if not self.results_directory.exists():
                logger.info(f"Creating results directory: {self.results_directory}")
                self.results_directory.mkdir(parents=True, exist_ok=True)
            
            # Validate write permissions
            if not os.access(self.results_directory, os.W_OK):
                raise PermissionError(f"No write permission for results directory: {self.results_directory}")
            
            logger.info("Results directory validation completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up results directory: {e}")
            raise
    
    def load_weather_data(self, start_date=None, end_date=None):
        """
        Load Bomen weather data with the exact processing workflow.
        
        This method replicates the exact steps from lines 278-296 of the original script:
        1. Read CSV without using first column as index
        2. Convert date_time to datetime and set as index
        3. Filter to columns containing "(Avg )" only
        4. Optionally filter by date range
        
        Args:
            start_date (str, optional): Start date for filtering (format: 'YYYY-MM-DD')
            end_date (str, optional): End date for filtering (format: 'YYYY-MM-DD')
        
        Returns:
            pd.DataFrame: Processed weather data with datetime index and average measurements only
        """
        logger.info("Loading Bomen weather data...")
        
        try:
            # Step 1: Read the measured weather data, do not use the first column as index
            # (Replicating line 282 from original script)
            measured_weather = pd.read_csv(self.weather_file_path, index_col=False)
            logger.info(f"Loaded weather data with {len(measured_weather)} rows and {len(measured_weather.columns)} columns")
            
            # Step 2: Ensure the date_time is in datetime format and set as index
            # (Replicating lines 285-288 from original script)
            measured_weather.index = pd.to_datetime(measured_weather['date_time'])
            measured_weather.index.name = 'Timestamp'
            
            # Step 3: Only include the columns that contain "(Avg )"
            # (Replicating line 291 from original script)
            measured_weather = measured_weather.filter(like='(Avg )')
            logger.info(f"Filtered to {len(measured_weather.columns)} average measurement columns")
            
            # Step 4: Optional date range filtering
            if start_date or end_date:
                logger.info(f"Filtering data from {start_date} to {end_date}")
                measured_weather = measured_weather.loc[start_date:end_date]
                logger.info(f"Filtered to {len(measured_weather)} rows")
            
            # Log basic information about the loaded data
            logger.info("Weather data columns:")
            for col in measured_weather.columns:
                logger.info(f"  - {col}")
            
            logger.info(f"Data period: {measured_weather.index.min()} to {measured_weather.index.max()}")
            logger.info(f"Data resolution: {measured_weather.index.to_series().diff().mode().iloc[0]}")
            
            return measured_weather
            
        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            raise
    
    def get_data_summary(self, weather_df):
        """
        Generate summary statistics for weather data.
        
        Args:
            weather_df (pd.DataFrame): Weather data DataFrame
            
        Returns:
            dict: Summary statistics and metadata
        """
        summary = {
            'data_period': {
                'start': weather_df.index.min(),
                'end': weather_df.index.max(),
                'duration_days': (weather_df.index.max() - weather_df.index.min()).days
            },
            'data_resolution': weather_df.index.to_series().diff().mode().iloc[0],
            'total_records': len(weather_df),
            'columns': list(weather_df.columns),
            'missing_data': weather_df.isnull().sum().to_dict(),
            'basic_stats': weather_df.describe().to_dict()
        }
        
        return summary
    
    def export_to_csv(self, weather_df, filename=None, include_index=True):
        """
        Export weather data DataFrame to CSV file in the Results directory.
        
        Args:
            weather_df (pd.DataFrame): Weather data DataFrame to export
            filename (str, optional): Custom filename. If None, auto-generates based on date range.
            include_index (bool, optional): Whether to include the datetime index. Defaults to True.
        
        Returns:
            str: Full path to the exported CSV file
        """
        logger.info("Exporting weather data to CSV...")
        
        try:
            # Generate filename if not provided
            if filename is None:
                start_date = weather_df.index.min().strftime('%Y-%m-%d')
                end_date = weather_df.index.max().strftime('%Y-%m-%d')
                
                if start_date == end_date:
                    filename = f"bomen_weather_data_{start_date}.csv"
                else:
                    filename = f"bomen_weather_data_{start_date}_to_{end_date}.csv"
            
            # Ensure filename has .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Full file path
            export_path = self.results_directory / filename
            
            logger.info(f"Exporting to: {export_path}")
            logger.info(f"Data shape: {weather_df.shape}")
            logger.info(f"Include datetime index: {include_index}")
            
            # Export to CSV
            weather_df.to_csv(export_path, index=include_index)
            
            logger.info(f"Successfully exported weather data to: {export_path}")
            logger.info(f"File contains {len(weather_df)} records with {len(weather_df.columns)} columns")
            
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting weather data to CSV: {e}")
            raise


def load_bomen_weather_data(data_directory=None, start_date=None, end_date=None):
    """
    Convenience function to load Bomen weather data.
    
    Args:
        data_directory (str, optional): Path to weather data directory
        start_date (str, optional): Start date for filtering (format: 'YYYY-MM-DD')
        end_date (str, optional): End date for filtering (format: 'YYYY-MM-DD')
    
    Returns:
        pd.DataFrame: Processed weather data with datetime index and average measurements only
    """
    loader = WeatherDataLoader(data_directory)
    return loader.load_weather_data(start_date, end_date)


def export_weather_data_csv(weather_df, results_directory=None, filename=None, include_index=True):
    """
    Convenience function to export weather data to CSV file.
    
    Args:
        weather_df (pd.DataFrame): Weather data DataFrame to export
        results_directory (str, optional): Path to results directory. Defaults to '../Results'
        filename (str, optional): Custom filename. If None, auto-generates based on date range
        include_index (bool, optional): Whether to include datetime index. Defaults to True
    
    Returns:
        str: Full path to the exported CSV file
    """
    # Create a temporary loader instance for export functionality
    loader = WeatherDataLoader(results_directory=results_directory)
    return loader.export_to_csv(weather_df, filename, include_index)


def load_and_export_weather_data(data_directory=None, results_directory=None, 
                                 start_date=None, end_date=None, filename=None, 
                                 include_index=True):
    """
    Convenience function to load weather data and immediately export to CSV.
    
    Args:
        data_directory (str, optional): Path to weather data directory
        results_directory (str, optional): Path to results directory
        start_date (str, optional): Start date for filtering (format: 'YYYY-MM-DD')
        end_date (str, optional): End date for filtering (format: 'YYYY-MM-DD')
        filename (str, optional): Custom filename. If None, auto-generates based on date range
        include_index (bool, optional): Whether to include datetime index. Defaults to True
    
    Returns:
        tuple: (weather_df, export_path) - DataFrame and path to exported CSV file
    """
    # Load weather data
    loader = WeatherDataLoader(data_directory, results_directory)
    weather_df = loader.load_weather_data(start_date, end_date)
    
    # Export to CSV
    export_path = loader.export_to_csv(weather_df, filename, include_index)
    
    return weather_df, export_path


def validate_weather_data(weather_df):
    """
    Validate weather data for physical reasonableness (PV Engineer validation).
    
    Performs physical validation checks on solar irradiance and temperature data
    to ensure measurements are within expected ranges for Australian solar conditions.
    
    Args:
        weather_df (pd.DataFrame): Weather data DataFrame with datetime index
    
    Returns:
        dict: Validation results with checks and warnings
    """
    logger.info("Performing physical validation of weather data...")
    
    validation_results = {
        'status': 'PASS',
        'warnings': [],
        'errors': [],
        'checks': {}
    }
    
    try:
        # Temperature validation (Australian climate ranges)
        if 'Air Temperature mean (Avg )' in weather_df.columns:
            temp_col = 'Air Temperature mean (Avg )'
            temp_min = weather_df[temp_col].min()
            temp_max = weather_df[temp_col].max()
            temp_mean = weather_df[temp_col].mean()
            
            validation_results['checks']['temperature'] = {
                'min': temp_min,
                'max': temp_max,
                'mean': temp_mean,
                'range_check': 'PASS' if -10 <= temp_min <= 50 and -10 <= temp_max <= 50 else 'FAIL'
            }
            
            # Temperature range warnings
            if temp_min < -5 or temp_max > 45:
                validation_results['warnings'].append(f"Temperature range unusual for Australian conditions: {temp_min:.1f}°C to {temp_max:.1f}°C")
        
        # GHI (Global Horizontal Irradiance) validation
        if 'GHI Irradiance mean (Avg )' in weather_df.columns:
            ghi_col = 'GHI Irradiance mean (Avg )'
            ghi_values = weather_df[ghi_col]
            # Filter out negative values that indicate nighttime/no measurement
            ghi_positive = ghi_values[ghi_values > 0]
            
            if len(ghi_positive) > 0:
                ghi_min = ghi_positive.min()
                ghi_max = ghi_positive.max()
                ghi_mean = ghi_positive.mean()
                
                validation_results['checks']['ghi'] = {
                    'min_positive': ghi_min,
                    'max': ghi_max,
                    'mean_positive': ghi_mean,
                    'range_check': 'PASS' if ghi_max <= 1500 else 'FAIL',
                    'negative_count': len(ghi_values[ghi_values < 0])
                }
                
                # GHI range warnings
                if ghi_max > 1400:
                    validation_results['warnings'].append(f"Very high GHI detected: {ghi_max:.1f} W/m² (>1400 W/m² unusual)")
            else:
                validation_results['warnings'].append("No positive GHI values found in dataset")
        
        # POA (Plane of Array) irradiance validation
        if 'POA Irradiance mean (Avg )' in weather_df.columns:
            poa_col = 'POA Irradiance mean (Avg )'
            poa_values = weather_df[poa_col]
            poa_positive = poa_values[poa_values > 0]
            
            if len(poa_positive) > 0:
                poa_min = poa_positive.min()
                poa_max = poa_positive.max()
                poa_mean = poa_positive.mean()
                
                validation_results['checks']['poa'] = {
                    'min_positive': poa_min,
                    'max': poa_max,
                    'mean_positive': poa_mean,
                    'range_check': 'PASS' if poa_max <= 1600 else 'FAIL',  # POA can be higher than GHI
                    'negative_count': len(poa_values[poa_values < 0])
                }
                
                # POA vs GHI relationship check
                if 'GHI Irradiance mean (Avg )' in weather_df.columns:
                    ghi_col = 'GHI Irradiance mean (Avg )'
                    # During daylight hours, POA should generally be >= GHI for tracking systems
                    daylight_mask = (weather_df[ghi_col] > 50) & (weather_df[poa_col] > 50)
                    if daylight_mask.sum() > 0:
                        poa_ghi_ratio = (weather_df.loc[daylight_mask, poa_col] / 
                                       weather_df.loc[daylight_mask, ghi_col]).mean()
                        validation_results['checks']['poa_ghi_ratio'] = {
                            'mean_ratio': poa_ghi_ratio,
                            'ratio_check': 'PASS' if 0.8 <= poa_ghi_ratio <= 2.0 else 'FAIL'
                        }
                        
                        if poa_ghi_ratio < 0.8:
                            validation_results['warnings'].append(f"Low POA/GHI ratio: {poa_ghi_ratio:.2f} (expected ≥0.8 for tracking)")
            
        # Data completeness check
        total_records = len(weather_df)
        missing_records = weather_df.isnull().sum().sum()
        completeness = (total_records - missing_records) / total_records * 100
        
        validation_results['checks']['data_completeness'] = {
            'total_records': total_records,
            'missing_records': int(missing_records),
            'completeness_percent': completeness,
            'completeness_check': 'PASS' if completeness >= 95 else 'WARN'
        }
        
        if completeness < 95:
            validation_results['warnings'].append(f"Data completeness is {completeness:.1f}% (< 95%)")
        
        # Temporal continuity check
        time_diffs = weather_df.index.to_series().diff().dropna()
        expected_interval = pd.Timedelta(minutes=5)  # 5-minute resolution expected
        irregular_intervals = (time_diffs != expected_interval).sum()
        
        validation_results['checks']['temporal_continuity'] = {
            'expected_interval': str(expected_interval),
            'irregular_count': int(irregular_intervals),
            'continuity_check': 'PASS' if irregular_intervals == 0 else 'WARN'
        }
        
        if irregular_intervals > 0:
            validation_results['warnings'].append(f"Found {irregular_intervals} irregular time intervals")
        
        # Set overall status
        if validation_results['errors']:
            validation_results['status'] = 'FAIL'
        elif validation_results['warnings']:
            validation_results['status'] = 'WARN'
        
        logger.info(f"Validation completed with status: {validation_results['status']}")
        for warning in validation_results['warnings']:
            logger.warning(warning)
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        validation_results['status'] = 'ERROR'
        validation_results['errors'].append(str(e))
        return validation_results


def print_weather_summary(weather_df):
    """
    Print a formatted summary of weather data (replicating original script output).
    
    Args:
        weather_df (pd.DataFrame): Weather data DataFrame
    """
    print("\n" + "="*60)
    print("BOMEN WEATHER DATA SUMMARY")
    print("="*60)
    
    # Print head and tail (replicating lines 294-296 from original)
    print("\nFirst 5 rows:")
    print(weather_df.head())
    
    print("\nLast 5 rows:")
    print(weather_df.tail())
    
    print("\nColumns:")
    for i, col in enumerate(weather_df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nData period: {weather_df.index.min()} to {weather_df.index.max()}")
    print(f"Total records: {len(weather_df)}")
    print(f"Data resolution: {weather_df.index.to_series().diff().mode().iloc[0]}")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(weather_df.describe())


if __name__ == "__main__":
    """
    Example usage when run as standalone script with CSV export capability.
    """
    try:
        print("Loading Bomen weather data...")
        
        # Load weather data
        weather_data = load_bomen_weather_data()
        
        # Print summary (replicating original script behavior)
        print_weather_summary(weather_data)
        
        # Validate data
        validation = validate_weather_data(weather_data)
        
        print(f"\nValidation Status: {validation['status']}")
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        # Export to CSV
        print("\nExporting weather data to CSV...")
        try:
            export_path = export_weather_data_csv(weather_data)
            print(f"CSV export successful!")
            print(f"Exported to: {export_path}")
            
            # Display file info
            from pathlib import Path
            export_file = Path(export_path)
            file_size = export_file.stat().st_size / 1024  # Size in KB
            print(f"File size: {file_size:.1f} KB")
            
        except Exception as export_error:
            print(f"CSV export failed: {export_error}")
            print("Weather data loading completed successfully, but CSV export encountered an error.")
        
        print("\nWeather data processing completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()