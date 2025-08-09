#!/usr/bin/env python3
"""
Dataset Exploration and Analysis for Budget Efficiency Sentiment Data

This script performs comprehensive analysis of the Indonesian budget efficiency sentiment dataset
from February 2025, including data quality assessment, text length analysis, and visualizations.

Author: Data Analysis Script
Date: 2025
"""

import csv
import os
import sys
from collections import Counter, defaultdict
import re
import json
from datetime import datetime


def load_csv_data(filepath, delimiter=','):
    """
    Load CSV data with automatic delimiter detection
    
    Args:
        filepath (str): Path to CSV file
        delimiter (str): CSV delimiter (default: ',')
    
    Returns:
        tuple: (headers, data_rows)
    """
    data_rows = []
    headers = []
    
    # Try different delimiters
    delimiters = [delimiter, ';', '\t']
    
    for delim in delimiters:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=delim)
                headers = next(reader)
                data_rows = list(reader)
                
                # Check if we have reasonable number of columns
                if len(headers) > 5:  # Expect multiple columns
                    print(f"Successfully loaded {filepath} with delimiter '{delim}'")
                    return headers, data_rows
        except Exception as e:
            continue
    
    raise ValueError(f"Could not load CSV file {filepath} with any delimiter")


def analyze_data_quality(headers, data_rows):
    """
    Perform comprehensive data quality assessment
    
    Args:
        headers (list): Column headers
        data_rows (list): Data rows
    
    Returns:
        dict: Data quality metrics
    """
    if not data_rows:
        return {"error": "No data rows found"}
    
    total_rows = len(data_rows)
    total_cols = len(headers)
    
    # Initialize metrics
    metrics = {
        'total_rows': total_rows,
        'total_columns': total_cols,
        'column_info': {},
        'missing_values': {},
        'unique_values': {},
        'duplicate_rows': 0,
        'duplicate_rate': 0.0
    }
    
    # Column-wise analysis
    for col_idx, header in enumerate(headers):
        column_data = [row[col_idx] if col_idx < len(row) else '' for row in data_rows]
        
        # Count missing values (empty strings, None, 'nan', etc.)
        missing_count = sum(1 for val in column_data if not val or val.lower() in ['', 'nan', 'null', 'none'])
        
        # Count unique values
        unique_values = set(column_data)
        unique_count = len(unique_values)
        
        metrics['column_info'][header] = {
            'missing_count': missing_count,
            'missing_percentage': (missing_count / total_rows) * 100,
            'unique_count': unique_count,
            'data_type': infer_data_type(column_data[:100])  # Sample first 100 rows
        }
        
        metrics['missing_values'][header] = missing_count
        metrics['unique_values'][header] = unique_count
    
    # Detect duplicate rows
    row_signatures = []
    for row in data_rows:
        row_signature = '|'.join(row)
        row_signatures.append(row_signature)
    
    signature_counts = Counter(row_signatures)
    duplicates = sum(count - 1 for count in signature_counts.values() if count > 1)
    
    metrics['duplicate_rows'] = duplicates
    metrics['duplicate_rate'] = (duplicates / total_rows) * 100 if total_rows > 0 else 0
    
    return metrics


def infer_data_type(column_data):
    """
    Infer the data type of a column based on sample data
    
    Args:
        column_data (list): Sample column data
    
    Returns:
        str: Inferred data type
    """
    non_empty_data = [val for val in column_data if val and val.strip()]
    
    if not non_empty_data:
        return 'empty'
    
    # Check for numeric data
    numeric_count = 0
    for val in non_empty_data[:50]:  # Sample first 50 non-empty values
        try:
            float(val)
            numeric_count += 1
        except ValueError:
            pass
    
    if numeric_count / len(non_empty_data[:50]) > 0.8:
        return 'numeric'
    
    # Check for date patterns
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\w{3}\s+\w{3}\s+\d{2}\s+\d{2}:\d{2}:\d{2}',  # Mon Feb 07 23:02:38
    ]
    
    for val in non_empty_data[:10]:
        for pattern in date_patterns:
            if re.search(pattern, val):
                return 'datetime'
    
    # Check for URLs
    url_count = sum(1 for val in non_empty_data[:20] if 'http' in val.lower())
    if url_count > len(non_empty_data[:20]) * 0.3:
        return 'url'
    
    return 'text'


def analyze_text_length(headers, data_rows, text_column='full_text'):
    """
    Analyze text length distribution for the specified text column
    
    Args:
        headers (list): Column headers
        data_rows (list): Data rows
        text_column (str): Name of the text column to analyze
    
    Returns:
        dict: Text length analysis results
    """
    if text_column not in headers:
        return {"error": f"Column '{text_column}' not found in headers"}
    
    col_idx = headers.index(text_column)
    text_data = [row[col_idx] if col_idx < len(row) else '' for row in data_rows]
    
    # Calculate text lengths
    text_lengths = [len(text) if text else 0 for text in text_data]
    
    # Filter out empty texts for statistics
    non_empty_lengths = [length for length in text_lengths if length > 0]
    
    if not non_empty_lengths:
        return {"error": "No non-empty text found"}
    
    # Calculate statistics
    stats = {
        'total_texts': len(text_data),
        'non_empty_texts': len(non_empty_lengths),
        'empty_texts': len(text_data) - len(non_empty_lengths),
        'min_length': min(non_empty_lengths),
        'max_length': max(non_empty_lengths),
        'mean_length': sum(non_empty_lengths) / len(non_empty_lengths),
        'median_length': sorted(non_empty_lengths)[len(non_empty_lengths) // 2],
        'length_distribution': Counter(text_lengths),
        'length_ranges': {}
    }
    
    # Create length ranges for analysis
    ranges = [
        (0, 0, 'empty'),
        (1, 5, 'very_short'),
        (6, 50, 'short'),
        (51, 140, 'medium'),
        (141, 280, 'long'),
        (281, float('inf'), 'very_long')
    ]
    
    for min_len, max_len, range_name in ranges:
        if max_len == float('inf'):
            count = sum(1 for length in text_lengths if length >= min_len)
        else:
            count = sum(1 for length in text_lengths if min_len <= length <= max_len)
        
        stats['length_ranges'][range_name] = {
            'count': count,
            'percentage': (count / len(text_lengths)) * 100,
            'range': f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
        }
    
    return stats


def filter_data_by_text_length(headers, data_rows, text_column='full_text', min_length=5, max_length=280):
    """
    Filter data based on text length criteria
    
    Args:
        headers (list): Column headers
        data_rows (list): Data rows
        text_column (str): Name of the text column to filter by
        min_length (int): Minimum text length
        max_length (int): Maximum text length
    
    Returns:
        tuple: (filtered_headers, filtered_data_rows, filter_stats)
    """
    if text_column not in headers:
        return headers, data_rows, {"error": f"Column '{text_column}' not found"}
    
    col_idx = headers.index(text_column)
    
    filtered_rows = []
    for row in data_rows:
        text = row[col_idx] if col_idx < len(row) else ''
        text_length = len(text) if text else 0
        
        if min_length <= text_length <= max_length:
            filtered_rows.append(row)
    
    filter_stats = {
        'original_count': len(data_rows),
        'filtered_count': len(filtered_rows),
        'removed_count': len(data_rows) - len(filtered_rows),
        'retention_rate': (len(filtered_rows) / len(data_rows)) * 100 if data_rows else 0,
        'filter_criteria': f"Text length: {min_length}-{max_length} characters"
    }
    
    return headers, filtered_rows, filter_stats


def create_simple_visualization_data(data_dict):
    """
    Create simple text-based visualizations for the data
    
    Args:
        data_dict (dict): Data to visualize
    
    Returns:
        str: Text-based visualization
    """
    vis_text = ""
    
    # Create horizontal bar chart using text
    if isinstance(data_dict, dict):
        max_key_length = max(len(str(k)) for k in data_dict.keys()) if data_dict else 0
        max_value = max(data_dict.values()) if data_dict.values() else 1
        
        for key, value in sorted(data_dict.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((value / max_value) * 50) if max_value > 0 else 0
            bar = '█' * bar_length
            vis_text += f"{str(key):<{max_key_length}} │ {bar} {value}\n"
    
    return vis_text


def save_analysis_results(results, output_dir='analysis_output'):
    """
    Save analysis results to files
    
    Args:
        results (dict): Analysis results
        output_dir (str): Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as JSON
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Save summary as text
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATASET EXPLORATION AND ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. DATA QUALITY ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        if 'data_quality' in results:
            dq = results['data_quality']
            f.write(f"Total Rows: {dq.get('total_rows', 'N/A')}\n")
            f.write(f"Total Columns: {dq.get('total_columns', 'N/A')}\n")
            f.write(f"Duplicate Rows: {dq.get('duplicate_rows', 'N/A')} ({dq.get('duplicate_rate', 0):.2f}%)\n\n")
            
            f.write("Missing Values by Column:\n")
            for col, info in dq.get('column_info', {}).items():
                f.write(f"  {col}: {info['missing_count']} ({info['missing_percentage']:.2f}%)\n")
        
        f.write("\n2. TEXT LENGTH ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if 'text_analysis' in results:
            ta = results['text_analysis']
            f.write(f"Total Texts: {ta.get('total_texts', 'N/A')}\n")
            f.write(f"Non-empty Texts: {ta.get('non_empty_texts', 'N/A')}\n")
            f.write(f"Min Length: {ta.get('min_length', 'N/A')}\n")
            f.write(f"Max Length: {ta.get('max_length', 'N/A')}\n")
            f.write(f"Mean Length: {ta.get('mean_length', 0):.2f}\n")
            f.write(f"Median Length: {ta.get('median_length', 'N/A')}\n\n")
            
            f.write("Length Distribution:\n")
            for range_name, range_info in ta.get('length_ranges', {}).items():
                f.write(f"  {range_name.replace('_', ' ').title()}: {range_info['count']} ({range_info['percentage']:.2f}%)\n")
        
        f.write("\n3. FILTERED DATA STATISTICS\n")
        f.write("-" * 40 + "\n")
        if 'filter_stats' in results:
            fs = results['filter_stats']
            f.write(f"Original Count: {fs.get('original_count', 'N/A')}\n")
            f.write(f"Filtered Count: {fs.get('filtered_count', 'N/A')}\n")
            f.write(f"Removed Count: {fs.get('removed_count', 'N/A')}\n")
            f.write(f"Retention Rate: {fs.get('retention_rate', 0):.2f}%\n")
            f.write(f"Filter Criteria: {fs.get('filter_criteria', 'N/A')}\n")


def save_filtered_dataset(headers, filtered_data, output_path):
    """
    Save filtered dataset to CSV file
    
    Args:
        headers (list): Column headers
        filtered_data (list): Filtered data rows
        output_path (str): Output file path
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(filtered_data)
    
    print(f"Filtered dataset saved to: {output_path}")


def main():
    """
    Main function to run the dataset exploration and analysis
    """
    print("=" * 80)
    print("DATASET EXPLORATION AND ANALYSIS FOR BUDGET EFFICIENCY SENTIMENT DATA")
    print("=" * 80)
    
    # File paths
    raw_data_path = 'data/raw_data.csv'
    labeled_data_path = 'data/labeled_data.csv'
    
    results = {}
    
    # Check if files exist
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} not found!")
        return
    
    if not os.path.exists(labeled_data_path):
        print(f"Error: {labeled_data_path} not found!")
        return
    
    print(f"\n1. Loading datasets...")
    print(f"   - Raw data: {raw_data_path}")
    print(f"   - Labeled data: {labeled_data_path}")
    
    try:
        # Load raw data
        raw_headers, raw_data = load_csv_data(raw_data_path, delimiter=';')
        print(f"   ✓ Raw data loaded: {len(raw_data)} rows, {len(raw_headers)} columns")
        
        # Load labeled data
        labeled_headers, labeled_data = load_csv_data(labeled_data_path)
        print(f"   ✓ Labeled data loaded: {len(labeled_data)} rows, {len(labeled_headers)} columns")
        
        # Use raw data for primary analysis (as specified in requirements)
        primary_headers = raw_headers
        primary_data = raw_data
        
        print(f"\n2. Data Quality Assessment...")
        data_quality = analyze_data_quality(primary_headers, primary_data)
        results['data_quality'] = data_quality
        
        print(f"   ✓ Total rows: {data_quality['total_rows']}")
        print(f"   ✓ Total columns: {data_quality['total_columns']}")
        print(f"   ✓ Duplicate rows: {data_quality['duplicate_rows']} ({data_quality['duplicate_rate']:.2f}%)")
        
        # Show missing values summary
        print(f"\n   Missing Values Summary:")
        for col, count in data_quality['missing_values'].items():
            percentage = (count / data_quality['total_rows']) * 100
            print(f"     {col}: {count} ({percentage:.2f}%)")
        
        print(f"\n3. Text Length Analysis...")
        text_analysis = analyze_text_length(primary_headers, primary_data, 'full_text')
        results['text_analysis'] = text_analysis
        
        if 'error' not in text_analysis:
            print(f"   ✓ Total texts: {text_analysis['total_texts']}")
            print(f"   ✓ Non-empty texts: {text_analysis['non_empty_texts']}")
            print(f"   ✓ Text length range: {text_analysis['min_length']} - {text_analysis['max_length']}")
            print(f"   ✓ Mean length: {text_analysis['mean_length']:.2f}")
            print(f"   ✓ Median length: {text_analysis['median_length']}")
            
            print(f"\n   Text Length Distribution:")
            for range_name, range_info in text_analysis['length_ranges'].items():
                print(f"     {range_name.replace('_', ' ').title()}: {range_info['count']} ({range_info['percentage']:.2f}%)")
        else:
            print(f"   ✗ Error: {text_analysis['error']}")
        
        print(f"\n4. Filtering Data (5-280 characters)...")
        filtered_headers, filtered_data, filter_stats = filter_data_by_text_length(
            primary_headers, primary_data, 'full_text', 5, 280
        )
        results['filter_stats'] = filter_stats
        
        print(f"   ✓ Original count: {filter_stats['original_count']}")
        print(f"   ✓ Filtered count: {filter_stats['filtered_count']}")
        print(f"   ✓ Removed count: {filter_stats['removed_count']}")
        print(f"   ✓ Retention rate: {filter_stats['retention_rate']:.2f}%")
        
        print(f"\n5. Saving Results...")
        save_analysis_results(results)
        print(f"   ✓ Analysis results saved to 'analysis_output/'")
        
        # Save filtered dataset
        filtered_output_path = 'analysis_output/filtered_dataset.csv'
        save_filtered_dataset(filtered_headers, filtered_data, filtered_output_path)
        print(f"   ✓ Filtered dataset saved to '{filtered_output_path}'")
        
        print(f"\n6. Analysis Complete!")
        print("   Check 'analysis_output/' directory for detailed results.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()