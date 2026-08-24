"""
Manual Verification Sampler for Ukraine Aid Discourse Analysis
Samples sentences to verify masking and party attribution accuracy
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

def load_and_sample_data(
    processed_file_path='Factiva_Data/processed/Factiva_v1_processed_masked.csv',
    raw_extraction_path='Factiva_Data/processed/Factiva_v1_extracted_checkpoint.csv',
    n_samples_per_month=50,
    output_path='manual_verification_sample.csv'
):
    """
    Load processed data and create samples for manual verification.
    
    Parameters:
    -----------
    processed_file_path : str
        Path to the processed file with masked sentences
    raw_extraction_path : str
        Path to the extracted checkpoint before masking
    n_samples_per_month : int
        Number of samples to take per month
    output_path : str
        Path for output Excel file
    """
    
    print("=" * 80)
    print("MANUAL VERIFICATION SAMPLER")
    print("=" * 80)
    
    # Load processed data (with masking)
    print(f"\nLoading processed data from {processed_file_path}...")
    if not os.path.exists(processed_file_path):
        print(f"Error: Could not find {processed_file_path}")
        return None
    
    df_masked = pd.read_csv(processed_file_path)
    df_masked['date'] = pd.to_datetime(df_masked['date'])
    
    # Load raw extraction data (before masking) if available
    original_sentences = {}
    if os.path.exists(raw_extraction_path):
        print(f"Loading original extraction from {raw_extraction_path}...")
        df_raw = pd.read_csv(raw_extraction_path)
        
        # Create mapping of article titles to original text
        for idx, row in df_raw.iterrows():
            title = row['title']
            # Store the raw Ukraine sentences for each party
            if 'ukraine_republican_only' in row:
                original_sentences[title] = {
                    'republican': row.get('ukraine_republican_only', ''),
                    'democrat': row.get('ukraine_democrat_only', '')
                }
    else:
        print(f"Warning: Could not find original extraction at {raw_extraction_path}")
        print("Will use sentence_text column as original")
    
    # Add month column
    df_masked['year_month'] = df_masked['date'].dt.to_period('M')
    
    # Get unique months
    months = sorted(df_masked['year_month'].unique())
    print(f"\nFound {len(months)} months of data from {months[0]} to {months[-1]}")
    
    # Sample data
    sampled_data = []
    
    for month in months:
        month_data = df_masked[df_masked['year_month'] == month]
        
        # Sample by party to ensure balance
        for party in ['republican', 'democrat']:
            party_data = month_data[month_data['party_mentioned'] == party]
            
            if len(party_data) == 0:
                continue
            
            # Sample up to n_samples_per_month/2 for each party
            n_party_samples = min(len(party_data), n_samples_per_month // 2)
            
            # Random sample
            sample_indices = np.random.choice(party_data.index, n_party_samples, replace=False)
            sample = party_data.loc[sample_indices]
            
            for idx, row in sample.iterrows():
                # Get original sentence
                title = row['title']
                
                # Try to find original sentence from raw extraction
                if title in original_sentences:
                    party_sentences = original_sentences[title].get(party, '')
                    # Split by delimiter and try to match
                    if party_sentences:
                        orig_sents = party_sentences.split(' | ')
                        # Try to find matching sentence (this is approximate)
                        original = orig_sents[0] if orig_sents else row.get('sentence_text', '')
                    else:
                        original = row.get('sentence_text', '')
                else:
                    # Fallback to sentence_text column
                    original = row.get('sentence_text', '')
                
                sampled_data.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'month': str(month),
                    'party_mentioned': party,
                    'title': title[:100],  # Truncate long titles
                    'original_sentence': original[:500] if original else '',  # Truncate long sentences
                    'masked_sentence': row.get('cleaned_sentence', '')[:500],
                    'ideology_score': row.get('ideology_score', np.nan),
                    'general_dimension': row.get('general_dimension', np.nan),
                    'democracy_dimension': row.get('democracy_dimension', np.nan),
                    'priority_dimension': row.get('priority_dimension', np.nan),
                    'article_weight': row.get('article_weight', np.nan),
                    'publisher': row.get('publisher', '')
                })
    
    # Create DataFrame
    sample_df = pd.DataFrame(sampled_data)
    
    print(f"\nSampled {len(sample_df)} sentences total")
    print(f"Party distribution in sample:")
    print(sample_df['party_mentioned'].value_counts())
    
    # Sort by date and party
    sample_df = sample_df.sort_values(['date', 'party_mentioned'])
    
    # Save to CSV
    print(f"\nSaving to {output_path}...")
    sample_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Saved sample to {output_path}")
    
    # Create comparison statistics
    print("\n" + "=" * 80)
    print("VERIFICATION STATISTICS")
    print("=" * 80)
    
    # Check for [ENTITY] masking
    mask_stats = sample_df['masked_sentence'].str.contains('[ENTITY]', na=False).sum()
    print(f"\nSentences with [ENTITY] masking: {mask_stats}/{len(sample_df)} ({mask_stats/len(sample_df)*100:.1f}%)")
    
    # Check ideology score distribution by party
    print("\nIdeology Score Statistics by Party:")
    print(sample_df.groupby('party_mentioned')['general_dimension'].describe())
    
    # Check if scores differ significantly
    if 'general_dimension' in sample_df.columns:
        rep_scores = sample_df[sample_df['party_mentioned'] == 'republican']['general_dimension'].dropna()
        dem_scores = sample_df[sample_df['party_mentioned'] == 'democrat']['general_dimension'].dropna()
        
        if len(rep_scores) > 0 and len(dem_scores) > 0:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(rep_scores, dem_scores)
            print(f"\nT-test for ideology score difference:")
            print(f"  Republican mean: {rep_scores.mean():.2f}")
            print(f"  Democrat mean: {dem_scores.mean():.2f}")
            print(f"  Difference: {rep_scores.mean() - dem_scores.mean():.2f}")
            print(f"  T-statistic: {t_stat:.3f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  ✓ Significant difference detected")
            else:
                print("  ✗ No significant difference")
    
    return sample_df

def check_deberta_artifacts(sample_df):
    """
    Check for potential DeBERTa artifacts in the annotation.
    """
    print("\n" + "=" * 80)
    print("CHECKING FOR POTENTIAL DEBERTA ARTIFACTS")
    print("=" * 80)
    
    # 1. Check if [ENTITY] masking affects scores systematically
    if 'masked_sentence' in sample_df.columns and 'general_dimension' in sample_df.columns:
        has_entity = sample_df['masked_sentence'].str.contains('[ENTITY]', na=False)
        
        print("\n1. Effect of [ENTITY] Masking on Scores:")
        
        with_entity = sample_df[has_entity]['general_dimension'].dropna()
        without_entity = sample_df[~has_entity]['general_dimension'].dropna()
        
        if len(with_entity) > 0 and len(without_entity) > 0:
            print(f"  With [ENTITY]: Mean={with_entity.mean():.2f}, Std={with_entity.std():.2f}")
            print(f"  Without [ENTITY]: Mean={without_entity.mean():.2f}, Std={without_entity.std():.2f}")
            
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(with_entity, without_entity)
            print(f"  Difference: {with_entity.mean() - without_entity.mean():.2f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print("  ⚠️ WARNING: [ENTITY] masking significantly affects scores!")
    
    # 2. Check for sentence length bias
    print("\n2. Sentence Length Effects:")
    sample_df['sentence_length'] = sample_df['masked_sentence'].str.len()
    
    # Correlation between length and score
    if 'general_dimension' in sample_df.columns:
        length_corr = sample_df[['sentence_length', 'general_dimension']].corr().iloc[0, 1]
        print(f"  Correlation between length and score: {length_corr:.3f}")
        
        if abs(length_corr) > 0.3:
            print("  ⚠️ WARNING: Moderate correlation with sentence length detected!")
    
    # 3. Check for publisher effects
    print("\n3. Publisher Effects on Annotation:")
    if 'publisher' in sample_df.columns and 'general_dimension' in sample_df.columns:
        publisher_stats = sample_df.groupby('publisher')['general_dimension'].agg(['mean', 'count'])
        publisher_stats = publisher_stats[publisher_stats['count'] >= 5]  # Only publishers with 5+ samples
        
        if len(publisher_stats) > 1:
            print(f"  Publisher variance in scores:")
            print(publisher_stats.sort_values('mean'))
            
            # Check if variance is too high
            publisher_std = publisher_stats['mean'].std()
            if publisher_std > 15:
                print(f"  ⚠️ WARNING: High variance across publishers (std={publisher_std:.2f})")
    
    # 4. Check dimension correlations
    print("\n4. Dimension Correlations (should be moderately positive):")
    dim_cols = ['general_dimension', 'democracy_dimension', 'priority_dimension']
    existing_dims = [col for col in dim_cols if col in sample_df.columns]
    
    if len(existing_dims) > 1:
        corr_matrix = sample_df[existing_dims].corr()
        print(corr_matrix)
        
        # Check for unexpected patterns
        for i in range(len(existing_dims)):
            for j in range(i+1, len(existing_dims)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val < 0.2:
                    print(f"  ⚠️ WARNING: Low correlation between {existing_dims[i]} and {existing_dims[j]}")
                elif corr_val > 0.9:
                    print(f"  ⚠️ WARNING: Very high correlation between {existing_dims[i]} and {existing_dims[j]}")

def main():
    """
    Main execution function.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Sample the data
    sample_df = load_and_sample_data(
        processed_file_path='Factiva_Data/processed/Factiva_v1_processed_masked.csv',
        raw_extraction_path='Factiva_Data/processed/Factiva_v1_extracted_checkpoint.csv',
        n_samples_per_month=50,
        output_path='manual_verification_sample.csv'
    )
    
    if sample_df is not None:
        # Check for DeBERTa artifacts
        check_deberta_artifacts(sample_df)
        
        print("\n" + "=" * 80)
        print("MANUAL VERIFICATION CHECKLIST")
        print("=" * 80)
        print("\n✓ Check the CSV file for:")
        print("  1. Are party mentions correctly identified?")
        print("  2. Is the masking removing the right entities?")
        print("  3. Do ideology scores make intuitive sense?")
        print("  4. Are there systematic biases by publisher?")
        print("  5. Do both parties have diverse viewpoints represented?")
        print("\n⚠️ Red flags to watch for:")
        print("  - All sentences from one party annotated similarly")
        print("  - [ENTITY] appearing in unexpected places")
        print("  - Scores not matching sentence sentiment")
        print("  - Publisher bias driving party differences")

if __name__ == "__main__":
    main()
