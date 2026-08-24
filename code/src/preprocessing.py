import pandas as pd
import numpy as np
import re
import spacy
from enum import Enum
from typing import Dict, List, Optional
from tqdm import tqdm
from .config import TITLES, TITLES_ABBREVIATED, PARTY_LABELS, PROTECTED_TERMS, UKRAINE_KEYWORDS

class PartyMentionType(Enum):
    """Enum for different types of party mentions"""
    REPUBLICAN_ONLY = "republican_only"  # Only Republican, no Democrat
    DEMOCRAT_ONLY = "democrat_only"      # Only Democrat, no Republican

class UkraineContentExtractor:
    """
    Extracts Ukraine passages with different party mention patterns.
    Supports article-level salience weighting.
    """
    
    def __init__(self, republican_terms: List[str], democrat_terms: List[str]):
        # Load spaCy with minimal components for efficiency
        try:
            self.nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer", "textcat"])
        except OSError:
            print("Downloading en_core_web_lg...")
            from spacy.cli import download
            download("en_core_web_lg")
            self.nlp = spacy.load("en_core_web_lg", disable=["ner", "lemmatizer", "textcat"])
        
        # Store party terms
        self.republican_terms = republican_terms
        self.democrat_terms = democrat_terms
        
        # Comprehensive Ukraine-related keywords
        self.ukraine_keywords = UKRAINE_KEYWORDS
        
        # Flatten all keywords for pattern matching
        all_ukraine_keywords = []
        for category in self.ukraine_keywords.values():
            all_ukraine_keywords.extend(category)
         
        # Create regex pattern for Ukraine (case-insensitive)
        self.ukraine_pattern = re.compile(
            r'\b(' + '|'.join([re.escape(kw) for kw in all_ukraine_keywords]) + r')\b',
            re.IGNORECASE
        )
        
        # Create separate patterns for Republican and Democrat terms
        self.republican_pattern = re.compile(
            r'\b(' + '|'.join([re.escape(term) for term in republican_terms]) + r')\b',
            re.IGNORECASE
        )
        
        self.democrat_pattern = re.compile(
            r'\b(' + '|'.join([re.escape(term) for term in democrat_terms]) + r')\b',
            re.IGNORECASE
        )
    
    def count_total_sentences(self, text: str) -> int:
        """
        Count total sentences in article using spaCy
        """
        if not text or pd.isna(text):
            return 0
        
        try:
            doc = self.nlp(text[:1000000])
            return len(list(doc.sents))
        except:
            # Fallback to simple period counting if spaCy fails
            return len(re.split(r'[.!?]+', text))
    
    def extract_ukraine_content_by_party(
        self, 
        text: str
    ) -> Dict[str, any]:
        """
        Extract Ukraine-related sentences and classify EACH SENTENCE by party mention type.
        Returns sentence counts and salience weights.
        """
        if not text or pd.isna(text):
            empty_result = {
                'sentences': {pt.value: "" for pt in PartyMentionType},
                'counts': {pt.value: 0 for pt in PartyMentionType},
                'total_sentences': 0,
                'salience_weights': {pt.value: 0.0 for pt in PartyMentionType}
            }
            return empty_result
        
        # Process text to get sentences
        doc = self.nlp(text[:1000000])  # Limit for spaCy processing
        sentences = [sent.text.strip() for sent in doc.sents]
        
        total_sentences = len(sentences)
        
        if not sentences:
            empty_result = {
                'sentences': {pt.value: "" for pt in PartyMentionType},
                'counts': {pt.value: 0 for pt in PartyMentionType},
                'total_sentences': 0,
                'salience_weights': {pt.value: 0.0 for pt in PartyMentionType}
            }
            return empty_result
        
        # Initialize collections for each category
        republican_only_sentences = []
        democrat_only_sentences = []
        
        # Process EACH sentence individually
        for sent in sentences:
            # Check if this sentence mentions Ukraine
            if not self.ukraine_pattern.search(sent):
                continue  # Skip non-Ukraine sentences
            
            # Check party mentions IN THIS SPECIFIC SENTENCE
            has_republican = bool(self.republican_pattern.search(sent))
            has_democrat = bool(self.democrat_pattern.search(sent))
            
            # Only process if there are party mentions
            if has_republican or has_democrat:
                # Classify into specific exclusive category
                # We ignore sentences with BOTH parties or NO parties (though logic ensures at least one)
                if has_republican and not has_democrat:
                    republican_only_sentences.append(sent)
                elif has_democrat and not has_republican:
                    democrat_only_sentences.append(sent)
        
        # Count sentences for each type
        counts = {
            PartyMentionType.REPUBLICAN_ONLY.value: len(republican_only_sentences),
            PartyMentionType.DEMOCRAT_ONLY.value: len(democrat_only_sentences)
        }
        
        # Calculate salience weights (proportion of article devoted to Ukraine+party content)
        salience_weights = {
            pt.value: counts[pt.value] / total_sentences if total_sentences > 0 else 0.0
            for pt in PartyMentionType
        }
        
        # Create result dictionary
        result = {
            'sentences': {
                PartyMentionType.REPUBLICAN_ONLY.value: ' | '.join(republican_only_sentences),
                PartyMentionType.DEMOCRAT_ONLY.value: ' | '.join(democrat_only_sentences)
            },
            'counts': counts,
            'total_sentences': total_sentences,
            'salience_weights': salience_weights
        }
        
        return result
    
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'body',
        salience_threshold: float = 0.0,
        apply_threshold: bool = False,
        weight_transform: str = 'log_volume'
    ) -> pd.DataFrame:
        """
        Process entire dataframe to extract Ukraine content by party mention type.
        Includes salience weighting.
        
        Args:
            df: Input DataFrame with text column
            text_column: Name of column containing article text
            salience_threshold: Minimum proportion of Ukraine content (0.0 = no threshold)
            apply_threshold: Whether to filter articles below threshold
            weight_transform: 'linear', 'sqrt', 'log', or 'log_volume' (default)
        """
        df = df.copy()
        
        print(f"\n{'='*60}")
        print("UKRAINE CONTENT EXTRACTION WITH SALIENCE WEIGHTING")
        print(f"{'='*60}")
        print(f"Salience threshold: {salience_threshold*100:.1f}%")
        print(f"Apply threshold filtering: {apply_threshold}")
        print(f"Weight transformation: {weight_transform}")
        print(f"{'='*60}\n")
        
        # Initialize columns for sentences
        for party_type in PartyMentionType:
            df[f'ukraine_{party_type.value}'] = ''
        
        # Initialize columns for counts and weights
        df['total_sentences'] = 0
        for party_type in PartyMentionType:
            df[f'ukraine_{party_type.value}_count'] = 0
            df[f'ukraine_{party_type.value}_salience'] = 0.0
            df[f'ukraine_{party_type.value}_weight'] = 0.0
        
        # Process each row
        for idx in tqdm(df.index, desc="Extracting Ukraine content"):
            text = df.loc[idx, text_column]
            
            # Extract content for all party types
            extraction_result = self.extract_ukraine_content_by_party(text)
            
            # Store sentences
            for party_type, content in extraction_result['sentences'].items():
                df.loc[idx, f'ukraine_{party_type}'] = content
            
            # Store counts
            for party_type, count in extraction_result['counts'].items():
                df.loc[idx, f'ukraine_{party_type}_count'] = count
            
            # Store total sentences
            df.loc[idx, 'total_sentences'] = extraction_result['total_sentences']
            
            # Store raw salience (linear proportion)
            for party_type, salience in extraction_result['salience_weights'].items():
                df.loc[idx, f'ukraine_{party_type}_salience'] = salience
            
            # Apply weight transformation
            for party_type in PartyMentionType:
                salience = extraction_result['salience_weights'][party_type.value]
                
                if weight_transform == 'linear':
                    weight = salience
                elif weight_transform == 'sqrt':
                    weight = np.sqrt(salience) if salience > 0 else 0
                elif weight_transform == 'log':
                    # log(1 + ukraine_sent) / log(1 + total_sent)
                    ukraine_count = extraction_result['counts'][party_type.value]
                    total_count = extraction_result['total_sentences']
                    if total_count > 0 and ukraine_count > 0:
                        weight = np.log1p(ukraine_count) / np.log1p(total_count)
                    else:
                        weight = 0
                elif weight_transform == 'log_volume':
                    # log(1 + ukraine_sent)
                    ukraine_count = extraction_result['counts'][party_type.value]
                    weight = np.log1p(ukraine_count)
                else:
                    raise ValueError(f"Unknown weight_transform: {weight_transform}")
                
                df.loc[idx, f'ukraine_{party_type.value}_weight'] = weight
        
        # Apply threshold filtering if requested
        if apply_threshold:
            print(f"\n{'='*60}")
            print(f"APPLYING {salience_threshold*100:.1f}% SALIENCE THRESHOLD")
            print(f"{'='*60}")
            
            original_len = len(df)
            
            # For each party type, filter articles below threshold
            # We'll create a mask for articles that pass threshold for at least one party type
            passes_threshold = pd.Series(False, index=df.index)
            
            for party_type in PartyMentionType:
                salience_col = f'ukraine_{party_type.value}_salience'
                party_passes = df[salience_col] >= salience_threshold
                passes_threshold = passes_threshold | party_passes
            
            df = df[passes_threshold].copy()
            
            excluded = original_len - len(df)
            print(f"\nExcluded {excluded} articles with <{salience_threshold*100:.1f}% Ukraine content ({excluded/original_len*100:.1f}% of total)")
            print(f"Retained {len(df)} articles")
        
        return df

def mask_party_terms(
    df: pd.DataFrame,
    republican_terms: List[str],
    democrat_terms: List[str],
    rep_col: str = "ukraine_republican_only",
    dem_col: str = "ukraine_democrat_only",
    rep_out_col: str = "ukraine_rep_masked",
    dem_out_col: str = "ukraine_dem_masked",
    replacement_token: str = "[ENTITY]"
) -> pd.DataFrame:
    """
    Masks exact Republican/Democrat terms with replacement_token (e.g. [ENTITY] or "").
    Consolidates neighboring tokens if replacement_token is not empty.
    """
    missing_cols = [col for col in (rep_col, dem_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # 1. Prepare Regex Patterns
    # Sort by length desc to match longest terms first
    sorted_rep = sorted(list(set(republican_terms)), key=len, reverse=True)
    sorted_dem = sorted(list(set(democrat_terms)), key=len, reverse=True)

    def create_pattern(terms):
        if not terms:
            return None
        return re.compile(
            r'\b(' + '|'.join([re.escape(t) for t in terms]) + r')\b',
            re.IGNORECASE
        )

    rep_pattern = create_pattern(sorted_rep)
    dem_pattern = create_pattern(sorted_dem)

    def process_text(text: str, pattern: re.Pattern) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Split into sentences (pipe-separated)
        sentences = [segment.strip() for segment in text.split("|") if segment.strip()]
        processed_sentences = []
        
        for sentence in sentences:
            if pattern:
                # Replace exact terms with replacement_token
                sentence = pattern.sub(replacement_token, sentence)
            
            # Consolidate neighboring tokens if we are using a visible token
            if replacement_token and replacement_token.strip():
                escaped_token = re.escape(replacement_token)
                # Regex: Match token followed by whitespace and another token
                pattern_consolidate = re.compile(f'{escaped_token}\\s+{escaped_token}', re.IGNORECASE)
                
                prev_sentence = None
                while sentence != prev_sentence:
                    prev_sentence = sentence
                    sentence = pattern_consolidate.sub(replacement_token, sentence)
            
            processed_sentences.append(sentence)
            
        return " | ".join(processed_sentences)

    print(f"\nMasking party terms ({len(sorted_rep)} Rep, {len(sorted_dem)} Dem) with '{replacement_token}'...")
    tqdm.pandas(desc="Masking party terms", leave=False)
    df = df.copy()
    df[rep_out_col] = df[rep_col].progress_apply(lambda x: process_text(x, rep_pattern))
    df[dem_out_col] = df[dem_col].progress_apply(lambda x: process_text(x, dem_pattern))
    
    return df

def remove_titles_and_labels(
    df: pd.DataFrame,
    rep_in_col: str = "ukraine_rep_masked",
    dem_in_col: str = "ukraine_dem_masked",
    rep_out_col: str = "ukraine_rep_cleaned",
    dem_out_col: str = "ukraine_dem_cleaned",
) -> pd.DataFrame:
    """Remove titles (President, Senator, etc.) and party labels."""
    missing_cols = [col for col in (rep_in_col, dem_in_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # Terms to remove
    terms_to_remove = TITLES + TITLES_ABBREVIATED + PARTY_LABELS
    # Sort by length
    terms_to_remove = sorted(list(set(terms_to_remove)), key=len, reverse=True)

    # Create flexible pattern
    patterns = []
    for term in terms_to_remove:
        escaped = re.escape(term)
        if term.endswith('.'):
            patterns.append(f"{escaped}(?=\\s|$)")
        else:
            patterns.append(f"{escaped}\\b")
    
    remove_pattern = re.compile(
        r'\b(' + '|'.join(patterns) + r')',
        re.IGNORECASE
    )

    def clean_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        sentences = [segment.strip() for segment in text.split("|") if segment.strip()]
        cleaned_sentences = []
        
        for sentence in sentences:
            # Remove titles/labels
            cleaned = remove_pattern.sub("", sentence)
            # Cleanup whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)
            cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                cleaned_sentences.append(cleaned)
                
        return " | ".join(cleaned_sentences)

    print("\nRemoving titles and labels...")
    tqdm.pandas(desc="Removing titles", leave=False)
    df = df.copy()
    df[rep_out_col] = df[rep_in_col].progress_apply(clean_text)
    df[dem_out_col] = df[dem_in_col].progress_apply(clean_text)

    return df

def explode_sentences_to_rows(df, republican_col='ukraine_rep_cleaned', 
                               democrat_col='ukraine_dem_cleaned',
                               columns_to_remove=None):
    """
    Explodes pipe-separated sentences into individual rows while preserving
    all other columns from the original dataframe.
    Includes salience weights if available.
    """
    
    print("=" * 70)
    print("EXPLODING SENTENCES TO INDIVIDUAL ROWS")
    print("=" * 70)
    
    # Define expected salience/weight columns
    rep_count_col = 'ukraine_republican_only_count'
    rep_salience_col = 'ukraine_republican_only_salience'
    rep_weight_col = 'ukraine_republican_only_weight'
    
    dem_count_col = 'ukraine_democrat_only_count'
    dem_salience_col = 'ukraine_democrat_only_salience'
    dem_weight_col = 'ukraine_democrat_only_weight'
    
    # Check if weight columns exist
    has_weights = all([
        col in df.columns for col in [
            'total_sentences', 
            rep_count_col, rep_salience_col, rep_weight_col,
            dem_count_col, dem_salience_col, dem_weight_col
        ]
    ])
    
    # Define columns to exclude from metadata
    exclude_cols = [
        republican_col, 
        democrat_col,
        # Remove original/intermediate extraction columns
        'ukraine_republican_only', 
        'ukraine_democrat_only',
        'ukraine_rep_noentity',
        'ukraine_dem_noentity'
    ]
    
    if columns_to_remove:
        exclude_cols.extend(columns_to_remove)
    
    exclude_cols = list(set(exclude_cols))
    
    # Create list to store sentence-level rows
    sentence_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Exploding sentences"):
        # Get metadata columns
        metadata = {col: row[col] for col in df.columns if col not in exclude_cols}
        
        # Process Republican sentences
        if pd.notna(row[republican_col]) and str(row[republican_col]).strip():
            rep_sentences = str(row[republican_col]).split('|')
            # Get original sentences for reference
            orig_rep_sentences = str(row.get('ukraine_republican_only', '')).split('|')
            
            if has_weights:
                article_salience_meta = {
                    'article_total_sentences': row['total_sentences'],
                    'article_ukraine_sentence_count': row[rep_count_col],
                    'article_salience': row[rep_salience_col],
                    'article_weight': row[rep_weight_col]
                }
            else:
                article_salience_meta = {}
            
            for i, sent in enumerate(rep_sentences):
                sent = sent.strip()
                if sent:
                    # Safely get original sentence
                    orig_sent = orig_rep_sentences[i].strip() if i < len(orig_rep_sentences) else ""
                    
                    sentence_rows.append({
                        **metadata,
                        **article_salience_meta,
                        'sentence_text': sent,
                        'original_sentence': orig_sent,
                        'party_mentioned': 'republican',
                        'sentence_source': republican_col
                    })
        
        # Process Democrat sentences  
        if pd.notna(row[democrat_col]) and str(row[democrat_col]).strip():
            dem_sentences = str(row[democrat_col]).split('|')
            # Get original sentences for reference
            orig_dem_sentences = str(row.get('ukraine_democrat_only', '')).split('|')
            
            if has_weights:
                article_salience_meta = {
                    'article_total_sentences': row['total_sentences'],
                    'article_ukraine_sentence_count': row[dem_count_col],
                    'article_salience': row[dem_salience_col],
                    'article_weight': row[dem_weight_col]
                }
            else:
                article_salience_meta = {}
            
            for i, sent in enumerate(dem_sentences):
                sent = sent.strip()
                if sent:
                    # Safely get original sentence
                    orig_sent = orig_dem_sentences[i].strip() if i < len(orig_dem_sentences) else ""
                    
                    sentence_rows.append({
                        **metadata,
                        **article_salience_meta,
                        'sentence_text': sent,
                        'original_sentence': orig_sent,
                        'party_mentioned': 'democrat',
                        'sentence_source': democrat_col
                    })
    
    sentences_df = pd.DataFrame(sentence_rows)
    return sentences_df

def clean_political_terms(sentences_df, 
                         republican_terms,
                         democrat_terms, 
                         admin_officials,
                         additional_terms=None,
                         sentence_col='sentence_text',
                         use_entity_mask=False,
                         filter_multi_entity=False): # Default to False as we want to inspect first
    """
    Advanced masking/removal pipeline.
    
    Mode A (Masking):
    1. Mask known terms -> [ENTITY]
    2. Mask Title + Unknown Name -> Title [ENTITY]
    3. Remove Titles
    4. Consolidate [ENTITY]
    5. Clean
    
    Mode B (Removal):
    1. Remove known terms
    2. Remove Title + Unknown Name
    3. Remove Titles
    4. Clean
    """
    
    mode = "MASKING" if use_entity_mask else "REMOVAL"
    print("\n" + "=" * 70)
    print(f"CLEANING POLITICAL TERMS (Mode: {mode})")
    print("=" * 70)
    
    # 1. Define Lists
    # ---------------------------------------------------------
    # Known entities
    all_known_names = []
    all_known_names.extend([t for t in republican_terms])
    all_known_names.extend([t for t in democrat_terms])
    all_known_names.extend([t for t in admin_officials])
    if additional_terms:
        all_known_names.extend(additional_terms)
    
    # Sort by length (longest first) to avoid partial matches
    all_known_names = sorted(list(set(all_known_names)), key=len, reverse=True)
    
    # Titles to detect (for unlisted names) and remove
    titles = TITLES + TITLES_ABBREVIATED
    # Sort titles by length to ensure "Gov." matches before "Gov"
    titles = sorted(list(set(titles)), key=len, reverse=True)
    
    # Party labels to remove
    party_labels = PARTY_LABELS
    
    # Protected terms (DO NOT MASK/REMOVE these even if they follow a title)
    protected_terms = PROTECTED_TERMS
    
    # 2. Compile Regex Patterns
    # ---------------------------------------------------------
    # Pattern for known names
    known_names_pattern = re.compile(
        r'\b(' + '|'.join([re.escape(n) for n in all_known_names]) + r')\b', 
        re.IGNORECASE
    )
    
    # Pattern for "Title + Unknown Name"
    # Handle titles with periods differently for regex construction
    title_patterns = []
    for t in titles:
        escaped = re.escape(t)
        if t.endswith('.'):
            title_patterns.append(f"{escaped}(?=\\s|$)")
        else:
            title_patterns.append(f"{escaped}\\b")
            
    titles_regex = '|'.join(title_patterns)
    
    # Note: We use \b at start, but the individual patterns handle their own end boundaries
    title_name_pattern = re.compile(
        r'\b(' + titles_regex + r')\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)?)',
        re.IGNORECASE
    )
    
    # Pattern for removing titles and party labels (cleanup)
    cleanup_terms = titles + party_labels
    cleanup_patterns = []
    for t in cleanup_terms:
        escaped = re.escape(t)
        if t.endswith('.'):
            cleanup_patterns.append(f"{escaped}(?=\\s|$)")
        else:
            cleanup_patterns.append(f"{escaped}\\b")
            
    cleanup_pattern = re.compile(
        r'\b(' + '|'.join(cleanup_patterns) + r')',
        re.IGNORECASE
    )

    def process_text(text):
        if pd.isna(text) or not str(text).strip():
            return text
        
        text = str(text)
        
        if use_entity_mask:
            # --- MASKING PIPELINE ---
            
            # Step 1: Mask Known Names -> [ENTITY]
            text = known_names_pattern.sub('[ENTITY]', text)
            
            # Step 2: Mask Unknown Names after Titles
            def mask_unknown_after_title(match):
                title = match.group(1)
                name = match.group(2)
                if any(p.lower() in name.lower() for p in protected_terms):
                    return match.group(0)
                if '[ENTITY]' in name:
                    return f"{title} [ENTITY]"
                return f"{title} [ENTITY]"
            
            text = title_name_pattern.sub(mask_unknown_after_title, text)
            
            # Step 3: Remove Titles and Party Labels
            text = cleanup_pattern.sub('', text)
            
            # Step 4: Consolidate [ENTITY]
            text = re.sub(r'(\[ENTITY\]\s*)+', '[ENTITY] ', text)
            
        else:
            # --- REMOVAL PIPELINE ---
            
            # Step 1: Remove Known Names
            text = known_names_pattern.sub('', text)
            
            # Step 2: Remove Unknown Names after Titles
            def remove_unknown_after_title(match):
                title = match.group(1)
                name = match.group(2)
                if any(p.lower() in name.lower() for p in protected_terms):
                    return match.group(0)
                # Remove both title and name
                return "" 
            
            text = title_name_pattern.sub(remove_unknown_after_title, text)
            
            # Step 3: Remove Titles and Party Labels
            text = cleanup_pattern.sub('', text)

        # Step 5/4: Final Cleanup
        
        # Consolidate [ENTITY] tokens that may have become adjacent after title removal
        # This handles "[ENTITY] [ENTITY]" -> "[ENTITY]"
        if '[ENTITY]' in text:
            text = re.sub(r'(?:\[ENTITY\]\s*)+', '[ENTITY] ', text)

        text = re.sub(r"[`']s\b", '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'^\s*[.,;:]\s*', '', text)
        
        return text.strip()
    
    # Apply cleaning
    print("\nProcessing sentences...")
    tqdm.pandas(desc=f"{mode} & Cleaning")
    sentences_df = sentences_df.copy()
    sentences_df['cleaned_sentence'] = sentences_df[sentence_col].progress_apply(process_text)
    
    # Filter out sentences that became empty or too short
    sentences_df = sentences_df[sentences_df['cleaned_sentence'].str.len() > 10].copy()
    
    # Remove duplicates
    sentences_df = sentences_df.drop_duplicates(subset=['cleaned_sentence', 'title', 'date']).copy()
    
    # Logging for Masking Mode
    if use_entity_mask:
        sentences_df['entity_count'] = sentences_df['cleaned_sentence'].str.count(re.escape('[ENTITY]'))
        multi_entity_df = sentences_df[sentences_df['entity_count'] > 1]
        count_multi = len(multi_entity_df)
        
        print(f"\n[DIAGNOSTICS] Sentences with >1 [ENTITY]: {count_multi} ({count_multi/len(sentences_df)*100:.1f}%)")
        if count_multi > 0:
            print("Sample multi-entity sentences:")
            for _, row in multi_entity_df.head(5).iterrows():
                print(f" - {row['cleaned_sentence']}")
        
        # We do NOT filter them out by default anymore, as requested.
        sentences_df = sentences_df.drop(columns=['entity_count'])
    
    return sentences_df

def sample_and_print_diagnostics(df):
    """
    Prints random samples of sentences before and after cleaning,
    and summarizes multi-entity occurrences.
    """
    print("\n" + "="*70)
    print("DIAGNOSTICS: SAMPLING SENTENCES")
    print("="*70)
    
    # Ensure date is datetime
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
        
    df['year_month'] = df['date'].dt.to_period('M')
    
    for period in sorted(df['year_month'].unique()):
        period_df = df[df['year_month'] == period]
        if len(period_df) > 0:
            sample_size = min(5, len(period_df))
            sample = period_df.sample(n=sample_size)
            
            print(f"\n--- Month: {period} (Sample Size: {sample_size}) ---")
            for _, row in sample.iterrows():
                print(f"PARTY   : {row.get('party_mentioned', 'N/A')}")
                print(f"ORIGINAL: {row.get('original_sentence', 'N/A')}")
                print(f"CLEANED : {row.get('cleaned_sentence', 'N/A')}")
                print("-" * 30)

    # Multiple Entities
    print("\n" + "="*70)
    print("DIAGNOSTICS: MULTIPLE ENTITIES")
    print("="*70)
    
    # Count [ENTITY] occurrences
    df['entity_count'] = df['cleaned_sentence'].str.count(re.escape('[ENTITY]'))
    multi_entity_df = df[df['entity_count'] > 1]
    
    print(f"Sentences with >1 [ENTITY]: {len(multi_entity_df)} out of {len(df)} ({len(multi_entity_df)/len(df)*100:.2f}%)")
    
    if len(multi_entity_df) > 0:
        sample_size = min(5, len(multi_entity_df))
        sample = multi_entity_df.sample(n=sample_size)
        print(f"\n--- Random Sample of Multi-Entity Sentences ({sample_size}) ---")
        for _, row in sample.iterrows():
            print(f"PARTY   : {row.get('party_mentioned', 'N/A')}")
            print(f"ORIGINAL: {row.get('original_sentence', 'N/A')}")
            print(f"CLEANED : {row.get('cleaned_sentence', 'N/A')}")
            print("-" * 30)
            
    # Cleanup temp columns
    if 'year_month' in df.columns:
        df.drop(columns=['year_month'], inplace=True)
    if 'entity_count' in df.columns:
        df.drop(columns=['entity_count'], inplace=True)

def explode_and_clean(df, republican_terms, democrat_terms, admin_officials, use_entity_mask=False):
    """
    Complete pipeline: 
    1. Mask exact party terms -> [ENTITY] (or remove if use_entity_mask=False)
    2. Remove titles and labels
    3. Explode sentences to rows
    4. Final cleaning (whitespace, etc.)
    """
    
    # Determine replacement token
    replacement = "[ENTITY]" if use_entity_mask else ""
    
    # Step 1: Mask exact party terms
    print(f"\nStep 1: Masking exact party terms (Replacement: '{replacement}')...")
    df = mask_party_terms(
        df,
        republican_terms=republican_terms,
        democrat_terms=democrat_terms,
        rep_col="ukraine_republican_only",
        dem_col="ukraine_democrat_only",
        rep_out_col="ukraine_rep_masked",
        dem_out_col="ukraine_dem_masked",
        replacement_token=replacement
    )
    
    # Step 2: Remove titles and labels
    print("\nStep 2: Removing titles and labels...")
    df = remove_titles_and_labels(
        df,
        rep_in_col="ukraine_rep_masked",
        dem_in_col="ukraine_dem_masked",
        rep_out_col="ukraine_rep_cleaned",
        dem_out_col="ukraine_dem_cleaned"
    )
    
    # Step 3: Explode to rows
    print("\nStep 3: Exploding sentences to rows...")
    sentence_df = explode_sentences_to_rows(
        df=df,
        republican_col='ukraine_rep_cleaned',
        democrat_col='ukraine_dem_cleaned',
        columns_to_remove=['ukraine_rep_masked', 'ukraine_dem_masked']
    )
    
    # Step 4: Final cleaning
    # We use clean_political_terms but with a simplified role since masking is done
    # We still pass the terms just in case, but the heavy lifting is done
    print("\nStep 4: Final cleaning...")
    
    # Note: We disable the complex masking logic in clean_political_terms by setting use_entity_mask=False
    # because we already did our custom masking. We just want the final cleanup steps.
    # However, clean_political_terms in "REMOVAL" mode (use_entity_mask=False) removes known names.
    # Since we already masked them to [ENTITY] or "", they won't be found/removed again, which is fine.
    
    cleaned_sentence_df = clean_political_terms(
        sentence_df,
        republican_terms=[], # Empty because we already masked/handled them
        democrat_terms=[],
        admin_officials=[], 
        sentence_col='sentence_text',
        use_entity_mask=False, # We don't need its masking logic
        filter_multi_entity=False
    )
    
    # Step 5: Diagnostics
    sample_and_print_diagnostics(cleaned_sentence_df)
    
    return cleaned_sentence_df
