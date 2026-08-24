"""LLM Batch Scorer for Pipeline C.

Scores sentences using OpenAI's Batch API for multi-dimensional ideology analysis.

Workflow:
    1. Submit: Converts sentences to JSONL, uploads, creates batch job
    2. Status: Polls OpenAI for batch progress
    3. Download: Retrieves results, parses responses, saves CSV

Output:
    data/scored/Factiva_v1_llmrated.csv
    
Batch API Limits:
    - Max 50,000 requests per batch
    - Max 200MB input file size
    - 24-hour completion window
    - Output file auto-deleted after 30 days

References:
    - OpenAI Batch API: https://platform.openai.com/docs/guides/batch
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import pandas as pd

from openai import OpenAI

# Add project root to path (parent of src/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_PATH


class LLMBatchScorer:
    """
    Batch API scorer for multi-dimensional ideology analysis.
    
    Manages the full lifecycle: submit → poll → download → save
    Includes safety guards against accidental re-submission.
    """
    
    # Paths
    OUTPUT_DIR = os.path.join(DATA_PATH, "scored")
    BATCH_DIR = os.path.join(OUTPUT_DIR, "batches")
    LOG_PATH = os.path.join(OUTPUT_DIR, "batch_log.json")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "Factiva_v1_llmrated.csv")
    INPUT_CSV = os.path.join(DATA_PATH, "processed", "Factiva_v1_processed_masked.csv")
    
    MODEL = "gpt-5.1-2025-11-13"
    
    SYSTEM_PROMPT = "You are a precise political text analyst. Return only valid JSON."
    
    PROMPT_PREFIX = """<TASK>
You will rate a sentence from U.S. news media. The sentence contains [ENTITY], which masks a U.S. political party or politician. 
If multiple [ENTITY] appear in a sentence, they all refer to members of the same party.
Answer FOUR questions about [ENTITY]'s position as expressed in this sentence.
Answer based on ONLY what this sentence explicitly conveys. Do NOT infer which party [ENTITY] represents.
Answer each question independently.
</TASK>

<QUESTIONS>

Q1: What is [ENTITY]'s stance on providing military and financial aid to Ukraine?
0 = Strong support for providing aid
100 = Strong opposition to providing aid

Q2: How does [ENTITY] characterize Ukraine?
0 = Democratic ally defending Western values
100 = Corrupt country / not a U.S. concern / futile cause

Q3: Which strategic threat does [ENTITY] prioritize?
0 = Russia as primary threat to contain
100 = China/Indo-Pacific as primary strategic focus

Q4: Where does [ENTITY] suggest resources should be allocated?
0 = International commitments (NATO, alliances, global leadership)
100 = Domestic priorities (border, infrastructure, economy)

</QUESTIONS>

<NA_GUIDANCE>
Rate each question independently. Return "NA" for a question when:
- Sentence is purely procedural (e.g., "The committee convened at 2pm")
- Sentence mentions [ENTITY] but expresses no stance on that question (e.g., "[ENTITY] attended the briefing")
- Sentence is about Ukraine but has no U.S. policy relevance (e.g., "Fighting continued in Kharkiv")
</NA_GUIDANCE>

<OUTPUT>
JSON only.
{"q1": <0-100|"NA">, "q2": <0-100|"NA">, "q3": <0-100|"NA">, "q4": <0-100|"NA">}
</OUTPUT>

<SENTENCE>
"""
    
    PROMPT_SUFFIX = """
</SENTENCE>"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the batch scorer."""
        requested_output = os.path.realpath(
            os.path.abspath(output_dir or self.OUTPUT_DIR)
        )

        self.OUTPUT_DIR = requested_output
        self.BATCH_DIR = os.path.join(self.OUTPUT_DIR, "batches")
        self.LOG_PATH = os.path.join(self.OUTPUT_DIR, "batch_log.json")
        self.OUTPUT_CSV = os.path.join(self.OUTPUT_DIR, "Factiva_v1_llmrated.csv")

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
        
        # Create directories
        os.makedirs(self.BATCH_DIR, exist_ok=True)
    
    # =========================================================================
    # Log Management
    # =========================================================================
    
    def _load_log(self) -> Dict:
        """Load batch log file."""
        if os.path.exists(self.LOG_PATH):
            with open(self.LOG_PATH, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_log(self, log: Dict):
        """Save batch log file."""
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        with open(self.LOG_PATH, 'w') as f:
            json.dump(log, f, indent=2)
    
    def _update_log(self, **updates):
        """Update specific fields in log."""
        log = self._load_log()
        log.update(updates)
        self._save_log(log)
    
    # =========================================================================
    # Safety Guards
    # =========================================================================
    
    def check_can_submit(self) -> Tuple[bool, str]:
        """
        Check if submission is allowed based on log state.
        
        Returns:
            Tuple of (can_submit: bool, message: str)
        """
        if not os.path.exists(self.LOG_PATH):
            return True, "No existing batch found. Ready to submit."
        
        log = self._load_log()
        status = log.get('status')
        
        if status in ['submitted', 'validating', 'in_progress', 'finalizing']:
            return False, (
                f"⚠️  WARNING: A batch is already {status}!\n"
                f"   Batch ID: {log.get('batch_id')}\n"
                f"   Submitted: {log.get('submitted_at')}\n\n"
                f"   Preserve this log as part of the versioned batch record.\n"
                f"   Do not delete or overwrite it to start another run.\n"
            )
        elif status == 'completed':
            return False, (
                f"⚠️  WARNING: A completed batch is waiting to be downloaded!\n"
                f"   Batch ID: {log.get('batch_id')}\n\n"
                f"   Preserve this log as part of the versioned batch record.\n"
                f"   Do not delete or overwrite it to start another run.\n"
            )
        elif status == 'downloaded':
            return False, (
                f"⚠️  WARNING: Batch already completed and downloaded!\n"
                f"   Output file: {log.get('output_file')}\n\n"
                f"   Preserve this completed versioned namespace.\n"
                f"   A future authorized run must use another directory.\n"
            )
        elif status in ['failed', 'expired', 'cancelled']:
            return False, (
                f"ℹ️  Previous batch {status}. Preserve this run record.\n"
                f"   Previous error: {log.get('error')}\n"
                f"   A future authorized retry must use another versioned directory.\n"
            )
        
        return True, "Ready to submit."
    
    def is_scoring_complete(self) -> bool:
        """
        Check if scoring is complete and output file exists.
        
        Returns:
            True if scoring is complete and output file exists
        """
        if not os.path.exists(self.LOG_PATH):
            return False
        
        log = self._load_log()
        if log.get('status') != 'downloaded':
            return False
        
        output_file = log.get('output_file')
        if not output_file or not os.path.exists(output_file):
            return False
        
        return True
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _build_prompt(self, sentence: str) -> str:
        """Build full prompt for a sentence."""
        return self.PROMPT_PREFIX + sentence + self.PROMPT_SUFFIX
    
    def _create_jsonl_request(self, idx: int, sentence: str) -> Dict:
        """
        Create a single JSONL request object.
        
        Format per OpenAI docs:
        {"custom_id": "sentence_0", "method": "POST", "url": "/v1/chat/completions", 
         "body": {"model": "...", "messages": [...], "max_completion_tokens": 100, "temperature": 0,
                  "response_format": {"type": "json_object"}}}
        """
        return {
            "custom_id": f"sentence_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.MODEL,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(sentence)}
                ],
                "temperature": 0,
                "max_completion_tokens": 100,
                "response_format": {"type": "json_object"}
            }
        }
    
    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse LLM response into scores.
        
        Returns: {q1, q2, q3, q4, error_type}
        - q1-q4: float (0-100) or None for NA
        - error_type: 'success', 'json_error', 'range_error', 'api_error'
        """
        try:
            parsed = json.loads(response_text)
            if 'error' in parsed:
                return {
                    'q1': None, 'q2': None, 'q3': None, 'q4': None,
                    'llm_error_type': 'api_error',
                    'raw_response': response_text
                }
            
            result = {'raw_response': response_text, 'llm_error_type': 'success'}
            for q in ['q1', 'q2', 'q3', 'q4']:
                val = parsed.get(q)
                if val == "NA" or val == "na" or val is None:
                    result[q] = None
                else:
                    try:
                        score = float(val)
                        if 0 <= score <= 100:
                            result[q] = score
                        else:
                            result[q] = None
                            result['llm_error_type'] = 'range_error'
                    except (ValueError, TypeError):
                        result[q] = None
                        result['llm_error_type'] = 'range_error'
            return result
        except json.JSONDecodeError:
            return {
                'q1': None, 'q2': None, 'q3': None, 'q4': None,
                'llm_error_type': 'json_error',
                'raw_response': response_text
            }
    
    def _parse_batch_result_line(self, line: str) -> Tuple[int, Dict]:
        """
        Parse a single line from batch output JSONL.
        
        Line format:
        {"id": "batch_req_123", "custom_id": "sentence_42", 
         "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "..."}}]}},
         "error": null}
        
        Returns: (sentence_index, parsed_scores_dict)
        
        Handle error cases where response=null and error!=null.
        """
        data = json.loads(line)
        custom_id = data.get('custom_id', '')
        
        # Extract index from custom_id (format: "sentence_42")
        try:
            idx = int(custom_id.replace('sentence_', ''))
        except ValueError:
            idx = -1
        
        # Check for error
        if data.get('error') is not None:
            error_info = data['error']
            return idx, {
                'q1': None, 'q2': None, 'q3': None, 'q4': None,
                'llm_error_type': 'api_error',
                'raw_response': json.dumps(error_info)
            }
        
        # Extract response content
        response = data.get('response', {})
        body = response.get('body', {})
        choices = body.get('choices', [])
        
        if not choices:
            return idx, {
                'q1': None, 'q2': None, 'q3': None, 'q4': None,
                'llm_error_type': 'api_error',
                'raw_response': 'no_choices_in_response'
            }
        
        content = choices[0].get('message', {}).get('content', '')
        
        # Parse the LLM's JSON response
        return idx, self._parse_response(content)
    
    def _estimate_cost(self, n_sentences: int) -> float:
        """
        Estimate batch API cost in USD.
        
        Batch API pricing (50% discount from standard):
        - gpt-4.1: Input $1.00/1M, Output $4.00/1M (batch: $0.50/$2.00)
        
        Estimate ~350 input tokens, ~50 output tokens per sentence.
        """
        input_tokens = n_sentences * 350
        output_tokens = n_sentences * 50
        
        # Batch pricing (50% of standard)
        input_cost = (input_tokens / 1_000_000) * 0.50  # $0.50 per 1M
        output_cost = (output_tokens / 1_000_000) * 2.00  # $2.00 per 1M
        
        return input_cost + output_cost
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def submit_batch(self) -> str:
        """
        Submit all sentences as a single batch job.
        
        Steps:
        1. Check submission is allowed (safety guard)
        2. Load input CSV
        3. Generate JSONL file with all requests
        4. Upload JSONL to OpenAI Files API (purpose="batch")
        5. Create batch job (endpoint="/v1/chat/completions", completion_window="24h")
        6. Save log with batch_id and status="submitted"
        
        Returns batch_id or raises exception.
        """
        print("\n" + "=" * 70)
        print("LLM BATCH SCORER - SUBMITTING BATCH")
        print("=" * 70)
        
        # Step 1: Load input data
        print(f"\n[1/5] Loading input data from {self.INPUT_CSV}...")
        if not os.path.exists(self.INPUT_CSV):
            raise FileNotFoundError(f"Input file not found: {self.INPUT_CSV}")
        
        df = pd.read_csv(self.INPUT_CSV)
        n_sentences = len(df)
        print(f"      Loaded {n_sentences:,} sentences")
        
        # Check for required column
        if 'cleaned_sentence' not in df.columns:
            raise ValueError("Input CSV must have 'cleaned_sentence' column")
        
        # Estimate cost
        estimated_cost = self._estimate_cost(n_sentences)
        print(f"      Estimated cost: ${estimated_cost:.2f} USD")
        
        # Step 2: Generate JSONL file
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        jsonl_filename = f"batch_input_{timestamp}.jsonl"
        jsonl_path = os.path.join(self.BATCH_DIR, jsonl_filename)
        
        print(f"\n[2/5] Generating JSONL file...")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                sentence = row['cleaned_sentence']
                request = self._create_jsonl_request(idx, sentence)
                f.write(json.dumps(request) + '\n')
        
        file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
        print(f"      Created: {jsonl_path}")
        print(f"      File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 200:
            raise ValueError(f"JSONL file too large ({file_size_mb:.2f} MB). Max is 200 MB.")
        
        # Step 3: Upload file to OpenAI
        print(f"\n[3/5] Uploading file to OpenAI...")
        with open(jsonl_path, 'rb') as f:
            batch_input_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        file_id = batch_input_file.id
        print(f"      File ID: {file_id}")
        
        # Step 4: Create batch job
        print(f"\n[4/5] Creating batch job...")
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Pipeline C ideology scoring",
                "project": "UKRAID-Politicization",
                "sentences": str(n_sentences)
            }
        )
        
        batch_id = batch.id
        print(f"      Batch ID: {batch_id}")
        print(f"      Status: {batch.status}")
        
        # Step 5: Save log
        print(f"\n[5/5] Saving batch log...")
        log = {
            "status": batch.status,
            "batch_id": batch_id,
            "input_file_id": file_id,
            "input_jsonl_path": jsonl_path,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "total_sentences": n_sentences,
            "estimated_cost_usd": estimated_cost,
            "last_checked": None,
            "last_status_response": None,
            "completed_at": None,
            "downloaded_at": None,
            "output_file": None,
            "error": None
        }
        self._save_log(log)
        print(f"      Log saved to: {self.LOG_PATH}")
        
        print("\n" + "=" * 70)
        print("BATCH SUBMITTED SUCCESSFULLY")
        print("=" * 70)
        
        return batch_id
    
    def check_status(self) -> Dict:
        """
        Check and report batch status.
        
        Calls client.batches.retrieve(batch_id)
        
        Returns dict with:
        - status: current state
        - completed: number of requests completed
        - failed: number of requests failed
        - total: total requests
        - progress_pct: percentage complete
        
        Updates log file with latest status and last_checked timestamp.
        """
        log = self._load_log()
        
        if not log.get('batch_id'):
            return {
                'status': 'no_batch',
                'batch_id': None,
                'completed': 0,
                'failed': 0,
                'total': 0,
                'progress_pct': 0,
                'error': 'No batch found in this authorized versioned namespace.'
            }
        
        batch_id = log['batch_id']
        
        try:
            batch = self.client.batches.retrieve(batch_id)
        except Exception as e:
            return {
                'status': 'error',
                'batch_id': batch_id,
                'completed': 0,
                'failed': 0,
                'total': log.get('total_sentences', 0),
                'progress_pct': 0,
                'error': str(e)
            }
        
        # Extract counts
        request_counts = batch.request_counts
        total = request_counts.total if request_counts else log.get('total_sentences', 0)
        completed = request_counts.completed if request_counts else 0
        failed = request_counts.failed if request_counts else 0
        
        progress_pct = (completed / total * 100) if total > 0 else 0
        
        # Update log
        self._update_log(
            status=batch.status,
            last_checked=datetime.now(timezone.utc).isoformat(),
            last_status_response={
                'status': batch.status,
                'completed': completed,
                'failed': failed,
                'total': total,
                'output_file_id': batch.output_file_id,
                'error_file_id': batch.error_file_id
            }
        )
        
        # Check for completion
        if batch.status == 'completed':
            self._update_log(completed_at=datetime.now(timezone.utc).isoformat())
        elif batch.status in ['failed', 'expired', 'cancelled']:
            error_msg = f"Batch {batch.status}"
            if batch.errors:
                error_msg += f": {batch.errors}"
            self._update_log(error=error_msg)
        
        result = {
            'status': batch.status,
            'batch_id': batch_id,
            'completed': completed,
            'failed': failed,
            'total': total,
            'progress_pct': progress_pct,
            'output_file_id': batch.output_file_id,
            'error_file_id': batch.error_file_id
        }
        
        # Add error info if applicable
        if batch.status in ['failed', 'expired', 'cancelled']:
            result['error'] = log.get('error', f"Batch {batch.status}")
        
        return result
    
    def download_results(self) -> pd.DataFrame:
        """
        Download completed batch results and save to CSV.
        
        Steps:
        1. Verify batch status is 'completed'
        2. Get output_file_id from batch object
        3. Download output file via client.files.content(output_file_id)
        4. Parse each JSONL line
        5. IMPORTANT: Output order may not match input - use custom_id for mapping
        6. Merge scores with original DataFrame
        7. Add batch_id and rated_at columns
        8. Save to OUTPUT_CSV
        9. Update log to status='downloaded'
        
        Returns the scored DataFrame.
        """
        print("\n" + "=" * 70)
        print("LLM BATCH SCORER - DOWNLOADING RESULTS")
        print("=" * 70)
        
        log = self._load_log()
        batch_id = log.get('batch_id')
        
        if not batch_id:
            raise ValueError("No batch found in this authorized versioned namespace.")
        
        # Step 1: Verify status
        print(f"\n[1/6] Checking batch status...")
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != 'completed':
            raise ValueError(
                f"Batch is not ready for download. Current status: {batch.status}\n"
                "Use the authorized versioned runner to check progress."
            )
        
        print(f"      Batch ID: {batch_id}")
        print(f"      Status: {batch.status}")
        
        # Step 2: Get output file ID
        output_file_id = batch.output_file_id
        if not output_file_id:
            raise ValueError("No output file available for this batch.")
        
        print(f"      Output file ID: {output_file_id}")
        
        # Step 3: Download output file
        print(f"\n[2/6] Downloading output file...")
        file_response = self.client.files.content(output_file_id)
        result_text = file_response.text
        
        # Save raw output
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_jsonl_path = os.path.join(self.BATCH_DIR, f"batch_output_{timestamp}.jsonl")
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        print(f"      Raw output saved to: {output_jsonl_path}")
        
        # Step 4: Parse results
        print(f"\n[3/6] Parsing batch results...")
        results_by_idx = {}
        lines = result_text.strip().split('\n')
        
        success_count = 0
        error_count = 0
        
        for line in lines:
            if not line.strip():
                continue
            idx, parsed = self._parse_batch_result_line(line)
            if idx >= 0:
                results_by_idx[idx] = parsed
                if parsed.get('llm_error_type') == 'success':
                    success_count += 1
                else:
                    error_count += 1
        
        print(f"      Parsed {len(results_by_idx):,} results")
        print(f"      Success: {success_count:,}, Errors: {error_count:,}")
        
        # Step 5: Load original data
        print(f"\n[4/6] Loading original data...")
        df = pd.read_csv(self.INPUT_CSV)
        print(f"      Loaded {len(df):,} sentences")
        
        # Step 6: Merge scores
        print(f"\n[5/6] Merging scores with original data...")
        
        # Initialize score columns
        df['q1'] = None
        df['q2'] = None
        df['q3'] = None
        df['q4'] = None
        df['llm_error_type'] = 'not_scored'
        df['batch_id'] = batch_id
        df['rated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Map results back to DataFrame
        matched = 0
        for idx, result in results_by_idx.items():
            if idx < len(df):
                df.at[idx, 'q1'] = result.get('q1')
                df.at[idx, 'q2'] = result.get('q2')
                df.at[idx, 'q3'] = result.get('q3')
                df.at[idx, 'q4'] = result.get('q4')
                df.at[idx, 'llm_error_type'] = result.get('llm_error_type', 'unknown')
                matched += 1
        
        print(f"      Matched {matched:,} results to original rows")
        
        # Check for unmatched
        unmatched = len(df) - matched
        if unmatched > 0:
            print(f"      ⚠️  {unmatched:,} sentences not matched (may be batch errors)")
        
        # Step 7: Save to CSV
        print(f"\n[6/6] Saving scored data...")
        df.to_csv(self.OUTPUT_CSV, index=False)
        print(f"      Saved to: {self.OUTPUT_CSV}")
        
        # Update log
        self._update_log(
            status='downloaded',
            downloaded_at=datetime.now(timezone.utc).isoformat(),
            output_file=self.OUTPUT_CSV,
            output_jsonl_path=output_jsonl_path
        )
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("DOWNLOAD COMPLETE - SUMMARY")
        print("=" * 70)
        print(f"\nTotal sentences: {len(df):,}")
        print(f"Successfully scored: {success_count:,}")
        print(f"Errors: {error_count:,}")
        
        # NA rates per dimension
        print("\nNA rates by dimension:")
        for q in ['q1', 'q2', 'q3', 'q4']:
            na_count = df[q].isna().sum()
            na_rate = na_count / len(df) * 100
            print(f"  {q.upper()}: {na_count:,} ({na_rate:.1f}%)")
        
        print(f"\nOutput saved to: {self.OUTPUT_CSV}")
        
        return df
    
    def submit_and_wait(self, poll_interval: int = 60):
        """
        Submit batch and poll until complete, then download.
        
        Prints progress every poll_interval seconds:
        [10:30:15] Batch batch_abc123: in_progress (5000/37312, 13.4%)
        [10:31:15] Batch batch_abc123: in_progress (8500/37312, 22.8%)
        ...
        [11:45:30] Batch batch_abc123: completed
        Downloading results...
        """
        # Submit
        batch_id = self.submit_batch()
        
        print("\n" + "=" * 70)
        print("WAITING FOR BATCH COMPLETION")
        print(f"Polling every {poll_interval} seconds. Press Ctrl+C to stop (batch continues on server).")
        print("=" * 70)
        
        # Poll until complete
        while True:
            status = self.check_status()
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            if status['status'] == 'completed':
                print(f"[{timestamp}] Batch {batch_id}: COMPLETED")
                break
            elif status['status'] in ['failed', 'expired', 'cancelled']:
                print(f"[{timestamp}] Batch {batch_id}: {status['status'].upper()}")
                print(f"           Error: {status.get('error', 'Unknown')}")
                return
            else:
                print(f"[{timestamp}] Batch {batch_id}: {status['status']} "
                      f"({status['completed']:,}/{status['total']:,}, {status['progress_pct']:.1f}%)")
            
            time.sleep(poll_interval)
        
        # Download
        print("\nDownloading results...")
        df = self.download_results()
        
        print("\n" + "=" * 70)
        print("BATCH SCORING COMPLETE")
        print("=" * 70)
        print(f"\nScored {len(df):,} sentences")
        print(f"Output: {self.OUTPUT_CSV}")
