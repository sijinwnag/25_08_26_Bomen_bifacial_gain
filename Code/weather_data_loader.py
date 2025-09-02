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
                 wind_height_source: float = 3.0, use_csv_ghi: bool = True,
                 ghi_csv_pattern: str = "Bomen_weather_{year}.csv",
                 use_csv_temperature: bool = True,
                 temperature_csv_pattern: str = "Bomen_weather_{year}.csv"):
        """
        Initialize weather data processor.
        
        Args:
            data_directory (str, optional): Path to weather data directory.
                                          Defaults to '../Data/weather data' relative to script location.
            results_directory (str, optional): Path to results output directory.
                                             Defaults to '../Results' relative to script location.
            wind_height_source (float, optional): Height (in meters) of source wind speed measurements.
                                                 Used for wind speed height conversion to 10m. Defaults to 3.0.
            use_csv_ghi (bool, optional): Whether to use CSV-based GHI data to replace Excel GHI data.
                                        Defaults to True for enhanced data quality.
            ghi_csv_pattern (str, optional): Pattern for CSV GHI files. Use {year} placeholder for year substitution.
                                           Defaults to "Bomen_weather_{year}.csv".
            use_csv_temperature (bool, optional): Whether to use CSV-based temperature data to replace Excel temperature data.
                                                Defaults to True for enhanced data quality.
            temperature_csv_pattern (str, optional): Pattern for CSV temperature files. Use {year} placeholder for year substitution.
                                                   Defaults to "Bomen_weather_{year}.csv".
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
        
        # Store GHI CSV replacement parameters
        self.use_csv_ghi = use_csv_ghi
        self.ghi_csv_pattern = ghi_csv_pattern
        
        # Store temperature CSV replacement parameters
        self.use_csv_temperature = use_csv_temperature
        self.temperature_csv_pattern = temperature_csv_pattern
        
        logger.info(f"Data directory: {self.data_directory}")
        logger.info(f"Results directory: {self.results_directory}")
        logger.info(f"Weather data file: {self.weather_data_path}")
        logger.info(f"Wind height source: {self.wind_height_source}m (will convert to 10m using power law)")
        logger.info(f"CSV GHI replacement: {'Enabled' if self.use_csv_ghi else 'Disabled'}")
        if self.use_csv_ghi:
            logger.info(f"GHI CSV pattern: {self.ghi_csv_pattern}")
        logger.info(f"CSV temperature replacement: {'Enabled' if self.use_csv_temperature else 'Disabled'}")
        if self.use_csv_temperature:
            logger.info(f"Temperature CSV pattern: {self.temperature_csv_pattern}")
        
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
    
    def load_temperature_from_csv(self, year: int = 2022) -> pd.DataFrame:
        """
        Load temperature data from year-specific CSV file.
        
        Args:
            year (int, optional): Year for temperature data loading. Defaults to 2022.
            
        Returns:
            pd.DataFrame: Temperature data with datetime index and standardized column name
            
        Raises:
            FileNotFoundError: If year-specific CSV file doesn't exist
            ValueError: If temperature column not found or data quality issues detected
        """
        logger.info(f"Loading temperature data from CSV for year {year}...")
        
        try:
            # Construct CSV file path using pattern
            csv_filename = self.temperature_csv_pattern.format(year=year)
            csv_path = self.data_directory / csv_filename
            
            # Validate CSV file exists
            if not csv_path.exists():
                raise FileNotFoundError(f"Temperature CSV file not found: {csv_path}")
            
            # Load CSV file
            temp_df = pd.read_csv(csv_path)
            logger.info(f"Loaded temperature CSV with {len(temp_df)} rows and {len(temp_df.columns)} columns")
            
            # Validate required columns exist
            required_columns = ['date_time', 'Air Temperature mean (Avg )']
            missing_columns = [col for col in required_columns if col not in temp_df.columns]
            if missing_columns:
                raise ValueError(f"Required temperature columns missing: {missing_columns}")
            
            # Convert datetime column to datetime index
            temp_df['date_time'] = pd.to_datetime(temp_df['date_time'])
            temp_df = temp_df.set_index('date_time')
            temp_df.index.name = 'Timestamp'
            
            # Extract only temperature column for merging
            temperature_column = temp_df[['Air Temperature mean (Avg )']]
            
            # Physical validation of temperature data
            temp_values = temperature_column['Air Temperature mean (Avg )']
            temp_stats = {
                'count': len(temp_values),
                'min_value': temp_values.min(),
                'max_value': temp_values.max(),
                'unrealistic_low_count': (temp_values < -40).sum(),  # °C limit for temperature
                'unrealistic_high_count': (temp_values > 50).sum(),  # °C limit for temperature
                'valid_data_percentage': (temp_values.notna().sum() / len(temp_values)) * 100
            }
            
            logger.info(f"Temperature data statistics:")
            logger.info(f"  Data period: {temperature_column.index.min()} to {temperature_column.index.max()}")
            logger.info(f"  Value range: {temp_stats['min_value']:.2f} to {temp_stats['max_value']:.2f} °C")
            logger.info(f"  Unrealistic low values: {temp_stats['unrealistic_low_count']:,} (<-40°C)")
            logger.info(f"  Unrealistic high values: {temp_stats['unrealistic_high_count']:,} (>50°C)")
            logger.info(f"  Valid data: {temp_stats['valid_data_percentage']:.1f}%")
            
            # Warning for data quality issues
            if temp_stats['unrealistic_low_count'] > 0:
                logger.warning(f"Found {temp_stats['unrealistic_low_count']} unrealistically low temperature values (<-40°C)")
            
            if temp_stats['unrealistic_high_count'] > 0:
                logger.warning(f"Found {temp_stats['unrealistic_high_count']} unrealistically high temperature values (>50°C)")
            
            logger.info("Temperature CSV loading completed successfully")
            return temperature_column
            
        except Exception as e:
            logger.error(f"Error loading temperature data from CSV: {e}")
            raise
    
    def replace_temperature_data(self, main_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace existing temperature columns with CSV-based temperature data using timestamp alignment.
        
        Args:
            main_df (pd.DataFrame): Main weather DataFrame with existing temperature columns
            temp_df (pd.DataFrame): Temperature data from CSV with datetime index
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with CSV-based temperature replacing multi-station temperature
        """
        logger.info("Replacing multi-station temperature data with CSV-based temperature...")
        
        try:
            # Create copy to avoid modifying original
            result_df = main_df.copy()
            
            # Identify and remove existing temperature columns (CP01, CP02, CP03 variants)
            temp_columns_to_remove = []
            for col in result_df.columns:
                if 'Air Temperature' in col and ('CP01' in col or 'CP02' in col or 'CP03' in col):
                    temp_columns_to_remove.append(col)
            
            if temp_columns_to_remove:
                logger.info(f"Removing existing temperature columns: {temp_columns_to_remove}")
                result_df = result_df.drop(columns=temp_columns_to_remove)
            else:
                logger.warning("No existing multi-station temperature columns found to remove")
            
            # Align timestamps and merge temperature data (left join to preserve main_df structure)
            merged_df = result_df.join(temp_df, how='left')
            
            # Check timestamp alignment quality
            temp_column_name = 'Air Temperature mean (Avg )'
            overlap_count = merged_df[temp_column_name].notna().sum()
            total_count = len(merged_df)
            overlap_ratio = overlap_count / total_count if total_count > 0 else 0
            
            logger.info(f"Timestamp alignment results:")
            logger.info(f"  Total timestamps: {total_count:,}")
            logger.info(f"  Successful alignments: {overlap_count:,}")
            logger.info(f"  Alignment ratio: {overlap_ratio:.1%}")
            
            if overlap_ratio < 0.8:  # Less than 80% overlap
                logger.warning(f"Low timestamp overlap ratio: {overlap_ratio:.1%}")
                logger.warning("Consider checking data date ranges or time resolution compatibility")
            
            # Handle missing temperature data with forward fill (limited to reasonable gaps)
            if merged_df[temp_column_name].isna().any():
                missing_before = merged_df[temp_column_name].isna().sum()
                merged_df[temp_column_name] = merged_df[temp_column_name].fillna(method='ffill', limit=12)  # Max 1 hour gap
                missing_after = merged_df[temp_column_name].isna().sum()
                filled_count = missing_before - missing_after
                
                if filled_count > 0:
                    logger.info(f"Forward-filled {filled_count} missing temperature values (limited to 1-hour gaps)")
                if missing_after > 0:
                    logger.warning(f"Remaining missing temperature values: {missing_after}")
            
            logger.info("Temperature data replacement completed successfully")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error replacing temperature data: {e}")
            raise
    
    def validate_temperature_integration(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
        """
        Validate temperature data integration with comprehensive checks.
        
        Args:
            df_before (pd.DataFrame): DataFrame before temperature replacement
            df_after (pd.DataFrame): DataFrame after temperature replacement
            
        Returns:
            Dict: Comprehensive validation report with metrics and recommendations
        """
        logger.info("Validating temperature data integration...")
        
        validation_report = {
            'data_integrity': {},
            'temperature_replacement': {},
            'physical_validation': {},
            'recommendations': []
        }
        
        try:
            # Data integrity validation
            validation_report['data_integrity'] = {
                'rows_preserved': len(df_before) == len(df_after),
                'index_alignment': df_before.index.equals(df_after.index),
                'row_count_before': len(df_before),
                'row_count_after': len(df_after)
            }
            
            # Temperature column replacement validation
            temp_cols_before = [col for col in df_before.columns if 'Air Temperature' in col]
            temp_cols_after = [col for col in df_after.columns if 'Air Temperature' in col]
            
            validation_report['temperature_replacement'] = {
                'temperature_columns_before': temp_cols_before,
                'temperature_columns_after': temp_cols_after,
                'multi_station_removed': len([col for col in temp_cols_before if 'CP0' in col]) > 0,
                'csv_temperature_added': 'Air Temperature mean (Avg )' in temp_cols_after
            }
            
            # Physical validation of new temperature data
            if 'Air Temperature mean (Avg )' in df_after.columns:
                temp_values = df_after['Air Temperature mean (Avg )']
                validation_report['physical_validation'] = {
                    'min_value': float(temp_values.min()),
                    'max_value': float(temp_values.max()),
                    'unrealistic_low_count': int((temp_values < -40).sum()),
                    'unrealistic_high_count': int((temp_values > 50).sum()),
                    'valid_data_percentage': float((temp_values.notna().sum() / len(temp_values)) * 100),
                    'mean_value': float(temp_values.mean()),
                    'std_value': float(temp_values.std())
                }
            
            # Generate recommendations
            recommendations = []
            
            if not validation_report['data_integrity']['rows_preserved']:
                recommendations.append("WARNING: Row count changed during temperature replacement")
            
            if validation_report['physical_validation'].get('unrealistic_low_count', 0) > 10:
                recommendations.append("Consider investigating unrealistically low temperature values (<-40°C)")
            
            if validation_report['physical_validation'].get('unrealistic_high_count', 0) > 10:
                recommendations.append("Consider investigating unrealistically high temperature values (>50°C)")
            
            if validation_report['physical_validation'].get('valid_data_percentage', 100) < 95:
                recommendations.append("Data completeness below 95% - verify CSV data quality")
            
            if not recommendations:
                recommendations.append("Temperature integration validation passed all checks")
            
            validation_report['recommendations'] = recommendations
            
            # Log validation summary
            logger.info("Temperature integration validation results:")
            for category, results in validation_report.items():
                if category != 'recommendations':
                    logger.info(f"  {category}: {results}")
            
            for rec in recommendations:
                if "WARNING" in rec:
                    logger.warning(f"  {rec}")
                else:
                    logger.info(f"  {rec}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error during temperature integration validation: {e}")
            validation_report['error'] = str(e)
            return validation_report
    
    def load_ghi_from_csv(self, year: int = 2022) -> pd.DataFrame:
        """
        Load GHI data from year-specific CSV file.
        
        Args:
            year (int, optional): Year for GHI data loading. Defaults to 2022.
            
        Returns:
            pd.DataFrame: GHI data with datetime index and standardized column name
            
        Raises:
            FileNotFoundError: If year-specific CSV file doesn't exist
            ValueError: If GHI column not found or data quality issues detected
        """
        logger.info(f"Loading GHI data from CSV for year {year}...")
        
        try:
            # Construct CSV file path using pattern
            csv_filename = self.ghi_csv_pattern.format(year=year)
            csv_path = self.data_directory / csv_filename
            
            # Validate CSV file exists
            if not csv_path.exists():
                raise FileNotFoundError(f"GHI CSV file not found: {csv_path}")
            
            # Load CSV file
            ghi_df = pd.read_csv(csv_path)
            logger.info(f"Loaded GHI CSV with {len(ghi_df)} rows and {len(ghi_df.columns)} columns")
            
            # Validate required columns exist
            required_columns = ['date_time', 'GHI Irradiance mean (Avg )']
            missing_columns = [col for col in required_columns if col not in ghi_df.columns]
            if missing_columns:
                raise ValueError(f"Required GHI columns missing: {missing_columns}")
            
            # Convert datetime column to datetime index
            ghi_df['date_time'] = pd.to_datetime(ghi_df['date_time'])
            ghi_df = ghi_df.set_index('date_time')
            ghi_df.index.name = 'Timestamp'
            
            # Extract only GHI column for merging
            ghi_column = ghi_df[['GHI Irradiance mean (Avg )']]
            
            # Physical validation of GHI data
            ghi_values = ghi_column['GHI Irradiance mean (Avg )']
            ghi_stats = {
                'count': len(ghi_values),
                'min_value': ghi_values.min(),
                'max_value': ghi_values.max(),
                'negative_count': (ghi_values < 0).sum(),
                'unrealistic_high_count': (ghi_values > 1500).sum(),  # W/m² limit for GHI
                'valid_data_percentage': (ghi_values.notna().sum() / len(ghi_values)) * 100
            }
            
            logger.info(f"GHI data statistics:")
            logger.info(f"  Data period: {ghi_column.index.min()} to {ghi_column.index.max()}")
            logger.info(f"  Value range: {ghi_stats['min_value']:.2f} to {ghi_stats['max_value']:.2f} W/m²")
            logger.info(f"  Negative values: {ghi_stats['negative_count']:,} ({ghi_stats['negative_count']/ghi_stats['count']*100:.1f}%)")
            logger.info(f"  Valid data: {ghi_stats['valid_data_percentage']:.1f}%")
            
            # Warning for data quality issues
            if ghi_stats['negative_count'] > 0:
                logger.warning(f"Found {ghi_stats['negative_count']} negative GHI values (nighttime sensor offset)")
                logger.info("Negative values will be clipped to 0 in processing pipeline")
            
            if ghi_stats['unrealistic_high_count'] > 0:
                logger.warning(f"Found {ghi_stats['unrealistic_high_count']} unrealistically high GHI values (>1500 W/m²)")
            
            logger.info("GHI CSV loading completed successfully")
            return ghi_column
            
        except Exception as e:
            logger.error(f"Error loading GHI data from CSV: {e}")
            raise
    
    def replace_ghi_data(self, main_df: pd.DataFrame, ghi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace existing GHI columns with CSV-based GHI data using timestamp alignment.
        
        Args:
            main_df (pd.DataFrame): Main weather DataFrame with existing GHI columns
            ghi_df (pd.DataFrame): GHI data from CSV with datetime index
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with CSV-based GHI replacing multi-station GHI
        """
        logger.info("Replacing multi-station GHI data with CSV-based GHI...")
        
        try:
            # Create copy to avoid modifying original
            result_df = main_df.copy()
            
            # Identify and remove existing GHI columns (CP01, CP02, CP03 variants)
            ghi_columns_to_remove = []
            for col in result_df.columns:
                if 'GHI' in col and ('CP01' in col or 'CP02' in col or 'CP03' in col):
                    ghi_columns_to_remove.append(col)
            
            if ghi_columns_to_remove:
                logger.info(f"Removing existing GHI columns: {ghi_columns_to_remove}")
                result_df = result_df.drop(columns=ghi_columns_to_remove)
            else:
                logger.warning("No existing multi-station GHI columns found to remove")
            
            # Align timestamps and merge GHI data (left join to preserve main_df structure)
            merged_df = result_df.join(ghi_df, how='left')
            
            # Check timestamp alignment quality
            ghi_column_name = 'GHI Irradiance mean (Avg )'
            overlap_count = merged_df[ghi_column_name].notna().sum()
            total_count = len(merged_df)
            overlap_ratio = overlap_count / total_count if total_count > 0 else 0
            
            logger.info(f"Timestamp alignment results:")
            logger.info(f"  Total timestamps: {total_count:,}")
            logger.info(f"  Successful alignments: {overlap_count:,}")
            logger.info(f"  Alignment ratio: {overlap_ratio:.1%}")
            
            if overlap_ratio < 0.8:  # Less than 80% overlap
                logger.warning(f"Low timestamp overlap ratio: {overlap_ratio:.1%}")
                logger.warning("Consider checking data date ranges or time resolution compatibility")
            
            # Handle missing GHI data with forward fill (limited to reasonable gaps)
            if merged_df[ghi_column_name].isna().any():
                missing_before = merged_df[ghi_column_name].isna().sum()
                merged_df[ghi_column_name] = merged_df[ghi_column_name].fillna(method='ffill', limit=12)  # Max 1 hour gap
                missing_after = merged_df[ghi_column_name].isna().sum()
                filled_count = missing_before - missing_after
                
                if filled_count > 0:
                    logger.info(f"Forward-filled {filled_count} missing GHI values (limited to 1-hour gaps)")
                if missing_after > 0:
                    logger.warning(f"Remaining missing GHI values: {missing_after}")
            
            logger.info("GHI data replacement completed successfully")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error replacing GHI data: {e}")
            raise
    
    def validate_ghi_integration(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
        """
        Validate GHI data integration with comprehensive checks.
        
        Args:
            df_before (pd.DataFrame): DataFrame before GHI replacement
            df_after (pd.DataFrame): DataFrame after GHI replacement
            
        Returns:
            Dict: Comprehensive validation report with metrics and recommendations
        """
        logger.info("Validating GHI data integration...")
        
        validation_report = {
            'data_integrity': {},
            'ghi_replacement': {},
            'physical_validation': {},
            'recommendations': []
        }
        
        try:
            # Data integrity validation
            validation_report['data_integrity'] = {
                'rows_preserved': len(df_before) == len(df_after),
                'index_alignment': df_before.index.equals(df_after.index),
                'row_count_before': len(df_before),
                'row_count_after': len(df_after)
            }
            
            # GHI column replacement validation
            ghi_cols_before = [col for col in df_before.columns if 'GHI' in col]
            ghi_cols_after = [col for col in df_after.columns if 'GHI' in col]
            
            validation_report['ghi_replacement'] = {
                'ghi_columns_before': ghi_cols_before,
                'ghi_columns_after': ghi_cols_after,
                'multi_station_removed': len([col for col in ghi_cols_before if 'CP0' in col]) > 0,
                'csv_ghi_added': 'GHI Irradiance mean (Avg )' in ghi_cols_after
            }
            
            # Physical validation of new GHI data
            if 'GHI Irradiance mean (Avg )' in df_after.columns:
                ghi_values = df_after['GHI Irradiance mean (Avg )']
                validation_report['physical_validation'] = {
                    'min_value': float(ghi_values.min()),
                    'max_value': float(ghi_values.max()),
                    'negative_count': int((ghi_values < 0).sum()),
                    'unrealistic_high_count': int((ghi_values > 1500).sum()),
                    'valid_data_percentage': float((ghi_values.notna().sum() / len(ghi_values)) * 100),
                    'mean_value': float(ghi_values.mean()),
                    'std_value': float(ghi_values.std())
                }
            
            # Generate recommendations
            recommendations = []
            
            if not validation_report['data_integrity']['rows_preserved']:
                recommendations.append("WARNING: Row count changed during GHI replacement")
            
            if validation_report['physical_validation'].get('negative_count', 0) > 100:
                recommendations.append("Consider investigating high number of negative GHI values")
            
            if validation_report['physical_validation'].get('valid_data_percentage', 100) < 95:
                recommendations.append("Data completeness below 95% - verify CSV data quality")
            
            if validation_report['physical_validation'].get('max_value', 0) > 1400:
                recommendations.append("Very high GHI values detected - verify sensor calibration")
            
            if not recommendations:
                recommendations.append("GHI integration validation passed all checks")
            
            validation_report['recommendations'] = recommendations
            
            # Log validation summary
            logger.info("GHI integration validation results:")
            for category, results in validation_report.items():
                if category != 'recommendations':
                    logger.info(f"  {category}: {results}")
            
            for rec in recommendations:
                if "WARNING" in rec:
                    logger.warning(f"  {rec}")
                else:
                    logger.info(f"  {rec}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error during GHI integration validation: {e}")
            validation_report['error'] = str(e)
            return validation_report
    
    def calculate_station_medians(self, df: pd.DataFrame, year: int = 2022) -> pd.DataFrame:
        """
        Calculate median values across weather stations for each parameter.
        Optionally replaces GHI data with CSV-based data before processing.
        
        Args:
            df (pd.DataFrame): Input DataFrame with weather station data
            year (int, optional): Year for potential GHI CSV replacement. Defaults to 2022.
            
        Returns:
            pd.DataFrame: DataFrame with median values and original individual stations removed,
                         optionally with CSV-based GHI data replacing multi-station GHI
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
        
        # Temperature replacement with CSV data (before clipping in processing pipeline)
        if self.use_csv_temperature:
            logger.info(f"Attempting temperature replacement from CSV for year {year}...")
            try:
                # Store DataFrame before temperature replacement for validation
                df_before_temp = result_df.copy()
                
                # Load temperature data from CSV
                temp_df = self.load_temperature_from_csv(year)
                
                # Replace temperature data with CSV-based data
                result_df = self.replace_temperature_data(result_df, temp_df)
                
                # Validate temperature integration
                validation_report = self.validate_temperature_integration(df_before_temp, result_df)
                
                # Log key validation results
                if 'error' not in validation_report:
                    physical_val = validation_report.get('physical_validation', {})
                    logger.info(f"Temperature replacement successful:")
                    logger.info(f"  CSV temperature range: {physical_val.get('min_value', 'N/A'):.1f} to {physical_val.get('max_value', 'N/A'):.1f} °C")
                    logger.info(f"  Data completeness: {physical_val.get('valid_data_percentage', 'N/A'):.1f}%")
                else:
                    logger.error(f"Temperature integration validation failed: {validation_report['error']}")
                    
            except Exception as e:
                logger.warning(f"CSV temperature loading failed, using existing multi-station temperature: {e}")
                logger.info("Processing will continue with original multi-station temperature data")
        else:
            logger.info("CSV temperature replacement disabled - using existing multi-station temperature data")
        
        # GHI replacement with CSV data (before clipping in processing pipeline)
        if self.use_csv_ghi:
            logger.info(f"Attempting GHI replacement from CSV for year {year}...")
            try:
                # Store DataFrame before GHI replacement for validation
                df_before_ghi = result_df.copy()
                
                # Load GHI data from CSV
                ghi_df = self.load_ghi_from_csv(year)
                
                # Replace GHI data with CSV-based data
                result_df = self.replace_ghi_data(result_df, ghi_df)
                
                # Validate GHI integration
                validation_report = self.validate_ghi_integration(df_before_ghi, result_df)
                
                # Log key validation results
                if 'error' not in validation_report:
                    physical_val = validation_report.get('physical_validation', {})
                    logger.info(f"GHI replacement successful:")
                    logger.info(f"  CSV GHI range: {physical_val.get('min_value', 'N/A'):.1f} to {physical_val.get('max_value', 'N/A'):.1f} W/m²")
                    logger.info(f"  Data completeness: {physical_val.get('valid_data_percentage', 'N/A'):.1f}%")
                else:
                    logger.error(f"GHI integration validation failed: {validation_report['error']}")
                    
            except Exception as e:
                logger.warning(f"CSV GHI loading failed, using existing multi-station GHI: {e}")
                logger.info("Processing will continue with original multi-station GHI data")
        else:
            logger.info("CSV GHI replacement disabled - using existing multi-station GHI data")
        
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
                        filename = f"PVsyst_weather_{start_year}.csv"
                    else:
                        filename = f"PVsyst_weather_{start_year}_to_{end_year}.csv"
                else:
                    filename = "PVsyst_weather_empty.csv"
            
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
        
        # Step 2: Calculate medians (with optional GHI CSV replacement)
        df_with_medians = self.calculate_station_medians(df, year)
        
        # Step 3: Filter by year
        df_filtered = self.filter_by_year(df_with_medians, year)

        # Add a clipping to ensure all values are above or equal to zero
        df_filtered = df_filtered.clip(lower=0)

        # Step 3.5: Fill gaps in GHI data with median values
        ghi_main_col = 'GHI Irradiance mean (Avg )'
        ghi_median_col = 'GHI Irradiance mean (Avg )_median'
        
        if ghi_main_col in df_filtered.columns and ghi_median_col in df_filtered.columns:
            # Count missing values before gap filling
            missing_count_before = df_filtered[ghi_main_col].isna().sum()
            
            if missing_count_before > 0:
                logger.info(f"Filling {missing_count_before:,} missing values in '{ghi_main_col}' with '{ghi_median_col}' data...")
                
                # Fill missing values in main GHI column with median column values
                df_filtered[ghi_main_col] = df_filtered[ghi_main_col].fillna(df_filtered[ghi_median_col])
                
                # Count remaining missing values after gap filling
                missing_count_after = df_filtered[ghi_main_col].isna().sum()
                filled_count = missing_count_before - missing_count_after
                
                logger.info(f"Successfully filled {filled_count:,} gaps in GHI data using median values")
                if missing_count_after > 0:
                    logger.warning(f"Remaining missing GHI values: {missing_count_after:,} (median data also missing)")
                
                # Validate filled data
                filled_data_stats = df_filtered[ghi_main_col].dropna()
                if len(filled_data_stats) > 0:
                    logger.info(f"GHI data after gap filling: range {filled_data_stats.min():.2f} to {filled_data_stats.max():.2f} W/m²")
            else:
                logger.info(f"No missing values found in '{ghi_main_col}' - gap filling not needed")
        elif ghi_main_col not in df_filtered.columns:
            logger.warning(f"Main GHI column '{ghi_main_col}' not found - skipping gap filling")
        elif ghi_median_col not in df_filtered.columns:
            logger.warning(f"Median GHI column '{ghi_median_col}' not found - skipping gap filling")

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
    df_with_medians = processor.calculate_station_medians(df, year)
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