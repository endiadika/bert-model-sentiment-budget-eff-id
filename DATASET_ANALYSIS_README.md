# Dataset Exploration and Analysis for Budget Efficiency Sentiment Data

## Overview

This comprehensive analysis examines the Indonesian budget efficiency sentiment dataset from February 2025, containing social media conversations discussing budget efficiency policies in Indonesia. The analysis includes data quality assessment, text length distribution analysis, filtering, and visualization.

## Dataset Structure

### Raw Data (`data/raw_data.csv`)
- **Total Records**: 10,909 rows
- **Columns**: 12 fields
- **Format**: CSV with semicolon delimiter

#### Columns Description:
- `conversation_id_str`: Unique conversation identifier
- `created_at`: Timestamp of the post (datetime format)
- `favorite_count`: Number of favorites/likes (numeric)
- `full_text`: The actual text content of the post (text)
- `id_str`: Unique post identifier
- `image_url`: URL of attached images (URL, 77.2% missing)
- `in_reply_to_screen_name`: Username being replied to (text, 66.4% missing)
- `lang`: Language code (text)
- `location`: User's location (text, 59.4% missing)
- `quote_count`: Number of quote tweets (numeric)
- `reply_count`: Number of replies (numeric)
- `retweet_count`: Number of retweets (numeric)

### Labeled Data (`data/labeled_data.csv`)
- **Total Records**: 9,371 rows
- **Columns**: 39 fields (includes processed sentiment labels and features)
- **Format**: CSV with comma delimiter

## Analysis Results

### 1. Data Quality Assessment

#### Overall Quality Metrics:
- **Total Records**: 10,909
- **Duplicate Records**: 30 (0.28%)
- **Data Completeness**: High for core fields, variable for optional fields

#### Missing Values Analysis:
- **image_url**: 8,426 missing (77.2%) - Optional attachment field
- **in_reply_to_screen_name**: 7,240 missing (66.4%) - Only for reply tweets
- **location**: 6,480 missing (59.4%) - User location often not provided
- **Core fields** (full_text, id_str, created_at): No missing values

#### Data Types Distribution:
- **Numeric fields**: 6 columns (50%) - engagement metrics and counts
- **Text fields**: 4 columns (33%) - content and identifiers  
- **Datetime fields**: 1 column (8%) - timestamps
- **URL fields**: 1 column (8%) - image attachments

### 2. Text Length Analysis

#### Text Length Statistics:
- **Mean Length**: 155.23 characters
- **Median Length**: 133 characters
- **Range**: 18 - 436 characters
- **Total Texts**: 10,909 (all non-empty)

#### Length Distribution:
- **Short (6-50 chars)**: 445 texts (4.1%)
- **Medium (51-140 chars)**: 5,426 texts (49.7%)
- **Long (141-280 chars)**: 4,212 texts (38.6%)
- **Very Long (281+ chars)**: 826 texts (7.6%)

### 3. Data Filtering Results

#### Filter Criteria: Text length 5-280 characters
- **Original Count**: 10,909 records
- **Filtered Count**: 10,083 records
- **Removed Count**: 826 records (very long texts)
- **Retention Rate**: 92.43%

The filtering successfully removed extremely long texts while preserving 92.4% of the dataset, maintaining high data quality for sentiment analysis tasks.

## Analysis Scripts

### 1. `dataset_exploration.py`
Main analysis script that performs:
- Automatic CSV delimiter detection
- Comprehensive data quality assessment
- Text length distribution analysis
- Data filtering based on text length criteria
- Statistical summaries and metrics calculation

### 2. `data_visualization.py`
Visualization script that creates:
- ASCII art charts for text-based visualization
- HTML report with formatted results
- Data quality visualizations
- Text length distribution charts
- Summary statistics tables

## Output Files

### Generated Analysis Files:
- `analysis_output/analysis_results.json` - Complete analysis results in JSON format
- `analysis_output/analysis_summary.txt` - Text summary of key findings
- `analysis_output/detailed_visualizations.txt` - ASCII art visualizations
- `analysis_output/visualization_report.html` - Comprehensive HTML report
- `analysis_output/filtered_dataset.csv` - Filtered dataset (5-280 characters)

## Key Insights

### Data Quality Insights:
1. **High Data Completeness**: Core fields have no missing values
2. **Low Duplication**: Only 0.28% duplicate records
3. **Appropriate Missing Values**: Missing values are logical for optional fields
4. **Diverse Content**: High unique value counts in text fields

### Text Analysis Insights:
1. **Optimal Length Distribution**: Most texts (88.3%) fall within ideal ranges for analysis
2. **Twitter-like Characteristics**: Length distribution typical of social media content
3. **Filtering Effectiveness**: 92.43% retention rate preserves dataset quality
4. **Ready for NLP**: Text lengths suitable for sentiment analysis models

### Technical Quality:
1. **Clean Data Structure**: Well-formatted with consistent data types
2. **Minimal Preprocessing Needed**: High-quality raw data
3. **Scalable Analysis**: Efficient processing of 10K+ records
4. **Rich Metadata**: Comprehensive engagement and temporal information

## Recommendations

### For Data Usage:
1. **Handle Missing Values**: Consider imputation or exclusion based on analysis goals
2. **Remove Duplicates**: Clean the 30 duplicate records for analysis
3. **Use Filtered Dataset**: The 5-280 character filtered dataset is optimal for sentiment analysis
4. **Leverage Metadata**: Utilize engagement metrics and temporal data for enhanced analysis

### For Further Analysis:
1. **Sentiment Analysis**: Dataset is well-prepared for BERT or other NLP models
2. **Temporal Analysis**: Rich timestamp data enables time-series analysis
3. **Engagement Analysis**: Use favorite, retweet, and reply counts for popularity analysis
4. **Geographic Analysis**: Location data (where available) enables regional insights

## Usage Instructions

### Running the Analysis:
```bash
# Run complete data exploration
python dataset_exploration.py

# Generate visualizations
python data_visualization.py
```

### Requirements:
- Python 3.6+
- Standard library only (no external dependencies required)
- CSV files in `data/` directory

### Output:
All analysis results are saved to the `analysis_output/` directory with comprehensive reports and filtered datasets ready for further processing.

## Conclusion

The Indonesian budget efficiency sentiment dataset demonstrates high quality with excellent suitability for sentiment analysis and NLP tasks. The comprehensive analysis reveals clean, well-structured data with meaningful content distribution and minimal preprocessing requirements. The 92.43% retention rate after filtering confirms the dataset's readiness for machine learning applications.