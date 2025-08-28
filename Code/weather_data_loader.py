#!/usr/bin/env python3
"""
Weather Data Processing Module for Bomen Solar Farm Analysis

This module provides streamlined functionality to process weather data from the Bomen Solar Farm,
focusing on median aggregation across multiple weather stations and year-based filtering.

Features:
- Loads weather data from weather_data.csv with proper datetime handling
- Calculates median values across weather stations (CP01, CP02, CP03) for each parameter
- Filters data by year (default: 2022)
- Exports processed data to CSV files in the Results directory
- Simplified, focused approach optimized for weather station aggregation

Usage:
    from weather_data_loader import WeatherDataProcessor, process_weather_data
    
    # Using the class directly
    processor = WeatherDataProcessor()
    df, export_path = processor.process_weather_data(year=2022)
    
    # Using convenience function
    df, export_path = process_weather_data(year=2022)
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, List
import pvlib.atmosphere

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeatherDataProcessor:
    """
    Simplified weather data processor for Bomen Solar Farm analysis.
    
    Focuses on processing weather_data.csv with median aggregation across
    weather stations (CP01, CP02, CP03) and year-based filtering.
    """
    
    def __init__(self, data_directory: Optional[str] = None, results_directory: Optional[str] = None, 
                 wind_height_source: float = 3.0):
        """
        Initialize weather data processor.
        
        Args:
            data_directory (str, optional): Path to weather data directory.
                                          Defaults to '../Data/weather data' relative to script location.
            results_directory (str, optional): Path to results output directory.
                                             Defaults to '../Results' relative to script location.
            wind_height_source (float, optional): Height (in meters) of source wind speed measurements.
                                                 Used for wind speed height conversion to 10m. Defaults to 3.0.
        """
        # Setup data directory
        if data_directory is None:
            script_dir = Path(__file__).parent
            self.data_directory = script_dir.parent / "Data" / "weather data"
        else:
            self.data_directory = Path(data_directory)
        
        # Setup results directory
        if results_directory is None:
            script_dir = Path(__file__).parent
            self.results_directory = script_dir.parent / "Results"
        else:
            self.results_directory = Path(results_directory)
        
        # Define the target weather data file
        self.weather_data_path = self.data_directory / "weather_data.csv"
        
        # Store wind height conversion parameters
        self.wind_height_source = wind_height_source
        
        logger.info(f"Data directory: {self.data_directory}")
        logger.info(f"Results directory: {self.results_directory}")
        logger.info(f"Weather data file: {self.weather_data_path}")
        logger.info(f"Wind height source: {self.wind_height_source}m (will convert to 10m using power law)")
        
        # Validate weather data file exists
        if not self.weather_data_path.exists():
            raise FileNotFoundError(f"Weather data file not found: {self.weather_data_path}")
        
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
            logger.info("Results directory validation completed successfully")
        except Exception as e:
            logger.error(f"Error setting up results directory: {e}")
            raise
    
    def load_weather_data_csv(self) -> pd.DataFrame:
        """
        Load weather data from weather_data.csv with proper datetime indexing.
        
        Returns:
            pd.DataFrame: Weather data with datetime index
        """
        logger.info("Loading weather data from weather_data.csv...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(self.weather_data_path, index_col=0)
            logger.info(f"Loaded weather data with {len(df)} rows and {len(df.columns)} columns")
            
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.info("Converting index to datetime...")
                df.index = pd.to_datetime(df.index)
            df.index.name = 'Timestamp'
            
            logger.info(f"Data period: {df.index.min()} to {df.index.max()}")
            logger.info(f"Data resolution: {df.index.to_series().diff().mode().iloc[0]}")
            
            # Log sample of column names to understand structure
            logger.info("Sample columns:")
            for i, col in enumerate(df.columns[:10]):
                logger.info(f"  {i+1}. {col}")
            if len(df.columns) > 10:
                logger.info(f"  ... and {len(df.columns) - 10} more columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading weather data: {e}")
            raise
    
    def identify_parameter_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify parameter groups across weather stations (CP01, CP02, CP03).
        
        Uses comprehensive pattern matching to find ALL columns containing CPxx identifiers,
        regardless of position or additional suffixes.
        
        Args:
            df (pd.DataFrame): Input DataFrame with weather station columns
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping parameter names to their station columns
        """
        logger.info("Identifying parameter groups across weather stations...")
        
        parameter_groups = {}
        
        # Find all columns containing CP01, CP02, or CP03 anywhere in the name
        cpxx_columns = []
        for col in df.columns:
            if re.search(r'CP0[123]', col):
                cpxx_columns.append(col)
        
        logger.info(f"Found {len(cpxx_columns)} columns containing CPxx identifiers")
        
        # Group CPxx columns by parameter
        for col in cpxx_columns:
            # Extract parameter name by removing CPxx pattern and cleaning up
            parameter_name = self._extract_parameter_name(col)
            
            if parameter_name:
                if parameter_name not in parameter_groups:
                    parameter_groups[parameter_name] = []
                parameter_groups[parameter_name].append(col)
        
        # Filter to keep only parameters that have data from multiple stations
        filtered_groups = {}
        for param, cols in parameter_groups.items():
            if len(cols) >= 2:  # At least 2 stations for meaningful median
                # Sort columns to ensure consistent ordering (CP01, CP02, CP03)
                sorted_cols = sorted(cols, key=lambda x: self._extract_station_number(x))
                filtered_groups[param] = sorted_cols
                logger.info(f"  Parameter '{param}': {len(sorted_cols)} stations -> {sorted_cols}")
            else:
                logger.info(f"  Skipping '{param}': only {len(cols)} station(s)")
        
        logger.info(f"Found {len(filtered_groups)} parameter groups with multiple stations")
        return filtered_groups
    
    def _extract_parameter_name(self, column_name: str) -> Optional[str]:
        """
        Extract the base parameter name from a column containing CPxx identifier.
        
        Handles various naming patterns:
        - 'albedo_ratio_CP01' -> 'albedo_ratio'
        - 'albedo_ratio_CP01_resampled' -> 'albedo_ratio_resampled'
        - 'Air Temperature mean (Avg ) CP01' -> 'Air Temperature mean (Avg )'
        
        Args:
            column_name (str): Column name containing CPxx
            
        Returns:
            Optional[str]: Extracted parameter name, or None if extraction fails
        """
        try:
            # Pattern 1: CPxx at the end (e.g., 'albedo_ratio_CP01')
            match1 = re.match(r'^(.+?)_CP0[123]$', column_name)
            if match1:
                return match1.group(1)
            
            # Pattern 2: CPxx followed by suffix (e.g., 'albedo_ratio_CP01_resampled')
            match2 = re.match(r'^(.+?)_CP0[123]_(.+)$', column_name)
            if match2:
                base_name = match2.group(1)
                suffix = match2.group(2)
                return f"{base_name}_{suffix}"
            
            # Pattern 3: CPxx with space (e.g., 'Air Temperature mean (Avg ) CP01')
            match3 = re.match(r'^(.+)\s+CP0[123]$', column_name)
            if match3:
                return match3.group(1).strip()
            
            # Pattern 4: CPxx in middle (e.g., 'param_CP01_something_else')
            match4 = re.search(r'^(.+?)CP0[123](.*)$', column_name)
            if match4:
                prefix = match4.group(1).rstrip('_')
                suffix = match4.group(2).lstrip('_')
                if suffix:
                    return f"{prefix}_{suffix}" if prefix else suffix
                else:
                    return prefix
            
            logger.warning(f"Could not extract parameter name from: {column_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting parameter name from '{column_name}': {e}")
            return None
    
    def _extract_station_number(self, column_name: str) -> int:
        """
        Extract station number (1, 2, or 3) from CPxx pattern for sorting.
        
        Args:
            column_name (str): Column name containing CPxx
            
        Returns:
            int: Station number (1, 2, or 3), defaults to 0 if not found
        """
        match = re.search(r'CP0([123])', column_name)
        return int(match.group(1)) if match else 0
    
    def _convert_wind_speed_to_10m(self, df: pd.DataFrame, wind_speed_columns: List[str]) -> pd.DataFrame:
        """
        Convert wind speed from source height to 10m using PVLib atmospheric power law.
        
        Uses the power law: v2 = v1 * (z2/z1)^alpha where alpha = 0.143 (typical shear exponent).
        
        Args:
            df (pd.DataFrame): DataFrame containing wind speed data
            wind_speed_columns (List[str]): List of wind speed column names to convert
            
        Returns:
            pd.DataFrame: DataFrame with added "Wind Speed (10m)" columns
        """
        logger.info(f"Converting wind speed from {self.wind_height_source}m to 10m height using power law...")
        
        result_df = df.copy()
        
        for col in wind_speed_columns:
            try:
                # Get the wind speed data
                wind_speed_source = df[col]
                
                # Convert using PVLib power law with standard atmospheric shear exponent (0.143)
                # This represents neutral atmospheric stability conditions
                wind_speed_10m = pvlib.atmosphere.windspeed_powerlaw(
                    wind_speed_source, 
                    height_reference=self.wind_height_source, 
                    height_desired=10.0, 
                    exponent=0.143
                )
                
                # Create new column name for 10m wind speed
                new_col_name = col.replace('Wind Speed', 'Wind Speed (10m)')
                if new_col_name == col:  # Fallback if pattern doesn't match
                    new_col_name = f"{col}_10m"
                
                result_df[new_col_name] = wind_speed_10m
                
                # Log conversion statistics
                valid_count = wind_speed_10m.notna().sum()
                if valid_count > 0:
                    conversion_factor = 10.0 / self.wind_height_source
                    expected_factor = conversion_factor ** 0.143
                    sample_original = wind_speed_source.dropna().iloc[0] if valid_count > 0 else "N/A"
                    sample_converted = wind_speed_10m.dropna().iloc[0] if valid_count > 0 else "N/A"
                    
                    logger.info(f"  {col} -> {new_col_name}")
                    logger.info(f"    Valid conversions: {valid_count:,}")
                    logger.info(f"    Conversion factor: {expected_factor:.3f} (theoretical)")
                    logger.info(f"    Sample: {sample_original} -> {sample_converted}")
                
            except Exception as e:
                logger.error(f"Error converting wind speed for column '{col}': {e}")
                continue
        
        logger.info("Wind speed height conversion completed successfully")
        return result_df
    
    def calculate_station_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate median values across weather stations for each parameter.
        
        Args:
            df (pd.DataFrame): Input DataFrame with weather station data
            
        Returns:
            pd.DataFrame: DataFrame with median values and original individual stations removed
        """
        logger.info("Calculating median values across weather stations...")
        
        # Identify parameter groups
        parameter_groups = self.identify_parameter_groups(df)
        
        if not parameter_groups:
            logger.warning("No parameter groups found - returning original data")
            return df.copy()
        
        # Create a copy of the DataFrame to work with
        result_df = df.copy()
        
        # Calculate medians for each parameter group
        for parameter_name, station_columns in parameter_groups.items():
            try:
                # Calculate median across stations for this parameter
                median_values = df[station_columns].median(axis=1)
                
                # Create median column name
                median_col_name = f"{parameter_name}_median"
                
                # Add median column to result
                result_df[median_col_name] = median_values
                
                # Log statistics
                valid_count = median_values.notna().sum()
                logger.info(f"  {parameter_name} -> {median_col_name}: {valid_count:,} valid values")
                
                # Remove original station columns
                result_df = result_df.drop(columns=station_columns)
                logger.info(f"    Removed {len(station_columns)} individual station columns")
                
            except Exception as e:
                logger.error(f"Error calculating median for {parameter_name}: {e}")
                continue
        
        # Identify and convert wind speed columns to 10m height
        wind_speed_columns = []
        for col in result_df.columns:
            if 'wind speed' in col.lower() and col.endswith('_median'):
                wind_speed_columns.append(col)
        
        if wind_speed_columns:
            logger.info(f"Found {len(wind_speed_columns)} wind speed median columns for height conversion")
            result_df = self._convert_wind_speed_to_10m(result_df, wind_speed_columns)
        else:
            logger.info("No wind speed median columns found for height conversion")
        
        logger.info(f"Final DataFrame: {len(result_df)} rows, {len(result_df.columns)} columns")
        logger.info("Median calculation completed successfully")
        
        return result_df
    
    def filter_by_year(self, df: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
        """
        Filter DataFrame by specified year.
        
        Args:
            df (pd.DataFrame): Input DataFrame with datetime index
            year (int, optional): Year to filter by. Defaults to 2022.
            
        Returns:
            pd.DataFrame: Filtered DataFrame for the specified year
        """
        logger.info(f"Filtering data for year {year}...")
        
        try:
            # Ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have a datetime index for year filtering")
            
            # Filter by year
            year_mask = df.index.year == year
            filtered_df = df[year_mask].copy()
            
            if len(filtered_df) == 0:
                available_years = sorted(df.index.year.unique())
                raise ValueError(f"No data found for year {year}. Available years: {available_years}")
            
            logger.info(f"Filtered from {len(df)} to {len(filtered_df)} records for year {year}")
            logger.info(f"Date range: {filtered_df.index.min()} to {filtered_df.index.max()}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering by year {year}: {e}")
            raise
    
    def export_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None, 
                      include_index: bool = True) -> str:
        """
        Export DataFrame to CSV file in the Results directory.
        
        Args:
            df (pd.DataFrame): DataFrame to export
            filename (str, optional): Custom filename. If None, auto-generates based on content.
            include_index (bool, optional): Whether to include the datetime index. Defaults to True.
            
        Returns:
            str: Full path to the exported CSV file
        """
        logger.info("Exporting processed weather data to CSV...")
        
        try:
            # Generate filename if not provided
            if filename is None:
                if len(df) > 0:
                    start_year = df.index.min().year
                    end_year = df.index.max().year
                    
                    if start_year == end_year:
                        filename = f"weather_data_median_{start_year}.csv"
                    else:
                        filename = f"weather_data_median_{start_year}_to_{end_year}.csv"
                else:
                    filename = "weather_data_median_empty.csv"
            
            # Ensure filename has .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Full file path
            export_path = self.results_directory / filename
            
            logger.info(f"Exporting to: {export_path}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Include datetime index: {include_index}")
            
            # Count median columns
            median_columns = [col for col in df.columns if col.endswith('_median')]
            logger.info(f"Exporting {len(median_columns)} median parameters:")
            for col in median_columns:
                logger.info(f"  - {col}")
            
            # Export to CSV
            df.to_csv(export_path, index=include_index)
            
            logger.info(f"Successfully exported weather data to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting weather data to CSV: {e}")
            raise
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for processed weather data.
        
        Args:
            df (pd.DataFrame): Processed weather data DataFrame
            
        Returns:
            Dict: Summary statistics and metadata
        """
        summary = {
            'data_period': {
                'start': df.index.min(),
                'end': df.index.max(),
                'duration_days': (df.index.max() - df.index.min()).days
            },
            'data_resolution': df.index.to_series().diff().mode().iloc[0],
            'total_records': len(df),
            'total_columns': len(df.columns),
            'median_parameters': [col for col in df.columns if col.endswith('_median')],
            'missing_data_percent': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        return summary
    
    def process_weather_data(self, year: int = 2022) -> Tuple[pd.DataFrame, str]:
        """
        Complete processing pipeline: load -> calculate medians -> filter -> export.
        
        Args:
            year (int, optional): Year to filter by. Defaults to 2022.
            
        Returns:
            Tuple[pd.DataFrame, str]: Processed DataFrame and path to exported CSV file
        """
        logger.info(f"Starting complete weather data processing pipeline for year {year}...")
        
        # Step 1: Load data
        df = self.load_weather_data_csv()
        
        # Step 2: Calculate medians
        df_with_medians = self.calculate_station_medians(df)
        
        # Step 3: Filter by year
        df_filtered = self.filter_by_year(df_with_medians, year)

        # Add a clipping to ensure all values are above or equal to zero
        df_filtered = df_filtered.clip(lower=0)

        # Step 4: Export
        export_path = self.export_to_csv(df_filtered)
        
        # Generate summary
        summary = self.get_data_summary(df_filtered)
        logger.info("Processing completed successfully!")
        logger.info(f"Summary: {summary['total_records']:,} records, {len(summary['median_parameters'])} median parameters")
        
        return df_filtered, export_path


# Convenience functions for easy usage
def process_weather_data(year: int = 2022, data_directory: Optional[str] = None, 
                        results_directory: Optional[str] = None, 
                        wind_height_source: float = 3.0) -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to process weather data with median aggregation and year filtering.
    
    Args:
        year (int, optional): Year to filter by. Defaults to 2022.
        data_directory (str, optional): Path to weather data directory
        results_directory (str, optional): Path to results directory
        wind_height_source (float, optional): Height (in meters) of source wind speed measurements.
                                             Used for wind speed height conversion to 10m. Defaults to 3.0.
        
    Returns:
        Tuple[pd.DataFrame, str]: Processed DataFrame and path to exported CSV file
    """
    processor = WeatherDataProcessor(data_directory, results_directory, wind_height_source)
    return processor.process_weather_data(year)


def load_processed_weather_data(year: int = 2022, data_directory: Optional[str] = None, 
                                wind_height_source: float = 3.0) -> pd.DataFrame:
    """
    Convenience function to load and process weather data without exporting.
    
    Args:
        year (int, optional): Year to filter by. Defaults to 2022.
        data_directory (str, optional): Path to weather data directory
        wind_height_source (float, optional): Height (in meters) of source wind speed measurements.
                                             Used for wind speed height conversion to 10m. Defaults to 3.0.
        
    Returns:
        pd.DataFrame: Processed weather data with median values
    """
    processor = WeatherDataProcessor(data_directory, wind_height_source=wind_height_source)
    df = processor.load_weather_data_csv()
    df_with_medians = processor.calculate_station_medians(df)
    df_filtered = processor.filter_by_year(df_with_medians, year)
    return df_filtered


if __name__ == "__main__":
    """
    Example usage when run as standalone script.
    """
    try:
        print("Processing weather data with median aggregation...")
        
        # Process weather data for 2022 (default)
        processed_df, export_path = process_weather_data(year=2022)
        
        print(f"\nProcessing completed successfully!")
        print(f"Processed {len(processed_df):,} records")
        print(f"Data period: {processed_df.index.min()} to {processed_df.index.max()}")
        print(f"Exported to: {export_path}")
        
        # Display median parameter columns
        median_columns = [col for col in processed_df.columns if col.endswith('_median')]
        print(f"\nMedian parameters calculated ({len(median_columns)}):")
        for i, col in enumerate(median_columns, 1):
            param_name = col.replace('_median', '')
            sample_val = processed_df[col].dropna().iloc[0] if len(processed_df[col].dropna()) > 0 else "N/A"
            print(f"  {i}. {param_name}: {sample_val} (sample value)")
        
        # Display basic statistics
        print(f"\nBasic Statistics:")
        print(f"Total columns: {len(processed_df.columns)}")
        print(f"Data completeness: {((1 - processed_df.isnull().sum().sum() / (len(processed_df) * len(processed_df.columns))) * 100):.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()