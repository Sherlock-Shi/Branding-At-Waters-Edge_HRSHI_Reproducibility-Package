import pandas as pd
import requests
import os
from typing import Tuple, List, Set

class PoliticianDataManager:
    """
    Manages the retrieval and processing of US legislator data from the 
    'unitedstates/congress-legislators' repository.
    """
    
    def __init__(self, cache_dir: str = "UKRAID-Politicization/Politician_Data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.base_url = "https://unitedstates.github.io/congress-legislators/"
        self.files = {
            "current": "legislators-current.csv",
            "historical": "legislators-historical.csv"
        }
        
    def _download_file(self, filename: str) -> str:
        """Download file if not exists in cache."""
        url = self.base_url + filename
        path = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(path):
            print(f"Downloading {filename} from {url}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return None
        return path

    def get_party_lists(self) -> Tuple[List[str], List[str]]:
        """
        Returns tuple of (republican_terms, democrat_terms).
        Includes full names and unambiguous last names.
        """
        print("Loading legislator database...")
        
        # Load data
        curr_path = self._download_file(self.files["current"])
        hist_path = self._download_file(self.files["historical"])
        
        dfs = []
        if curr_path:
            dfs.append(pd.read_csv(curr_path))
        if hist_path:
            # For historical, we only want recent ones to avoid noise from 19th century
            # We can filter by birthday > 1940 as a proxy for "active in modern era"
            # or just take all and rely on ambiguity filter.
            # Let's filter by birthday to keep size manageable and relevance high.
            df_h = pd.read_csv(hist_path)
            if 'birthday' in df_h.columns:
                df_h = df_h[df_h['birthday'] > '1940-01-01'].copy()
            dfs.append(df_h)
            
        if not dfs:
            print("Warning: No legislator data available.")
            return [], []
            
        df = pd.concat(dfs, ignore_index=True)
        
        # Normalize party names
        # Common values: 'Republican', 'Democrat', 'Independent'
        df['party'] = df['party'].fillna('Unknown')
        
        reps = df[df['party'] == 'Republican']
        dems = df[df['party'] == 'Democrat']
        
        # 1. Generate Full Names (High Precision)
        # Generate variations: "First Last", "First Middle Last", "Nickname Last"
        def get_names(sub_df):
            names = set()
            for _, row in sub_df.iterrows():
                last = str(row['last_name']).strip()
                first = str(row['first_name']).strip()
                
                # Variation 1: First Last (e.g. "Paul Ryan")
                names.add(f"{first} {last}")
                
                # Variation 2: First Middle Last (e.g. "Paul D. Ryan")
                if 'middle_name' in row and pd.notna(row['middle_name']):
                    middle = str(row['middle_name']).strip()
                    names.add(f"{first} {middle} {last}")
                    
                    # Variation 2b: First Middle Initial Last (e.g. "Donald J. Trump")
                    if len(middle) > 0:
                        initial = middle[0]
                        names.add(f"{first} {initial}. {last}")
                        names.add(f"{first} {initial} {last}")
                
                # Variation 3: Nickname Last (e.g. "Chuck Schumer")
                if 'nickname' in row and pd.notna(row['nickname']):
                    names.add(f"{row['nickname']} {last}")
                    
            return names

        rep_full_names = get_names(reps)
        dem_full_names = get_names(dems)
        
        # 2. Generate Last Names (Check for Ambiguity)
        # User requested to DISABLE matching by just the last name.
        # We only use the full name combinations generated above.
        
        # rep_last_names = set(reps['last_name'].dropna().unique())
        # dem_last_names = set(dems['last_name'].dropna().unique())
        
        # Find ambiguous last names (exist in both parties)
        # ambiguous_last_names = rep_last_names.intersection(dem_last_names)
        
        # Also filter out very common words that might be names (e.g., "Young", "Green", "Scott")
        # This is a heuristic list of names that are also common English words
        # (List preserved for reference but unused for last-name matching now)
        common_word_names = {
            'Young', 'Green', 'White', 'Brown', 'Black', 'Scott', 'King', 'Hill', 
            'Baker', 'Carter', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson',
            'Thomas', 'Jackson', 'Lee', 'Hall', 'Allen', 'Wright', 'Walker', 'Rose',
            'Kelly', 'Smith', 'Johnson', 'Williams', 'Jones', 'Davis', 'Martin',
            # Expanded list to prevent false positives (e.g., "Strong", "Good", "Case")
            'Strong', 'Good', 'Case', 'Buck', 'Bacon', 'Mast', 'Cloud', 'Guest', 
            'Hunt', 'Self', 'Wild', 'Dean', 'Mann', 'Cole', 'Crane', 'Crow', 
            'Dunn', 'Flood', 'Fry', 'Mills', 'Neal', 'Nunn', 'Paul', 'Reed', 
            'Ross', 'Roy', 'Ryan', 'Steel', 'Banks', 'Barr', 'Bass', 'Bell', 
            'Berry', 'Bishop', 'Bond', 'Bowman', 'Boyle', 'Brady', 'Brooks', 
            'Budd', 'Burgess', 'Bush', 'Butler', 'Byrd', 'Cannon', 'Carson', 
            'Castor', 'Chabot', 'Cheney', 'Chu', 'Clark', 'Clarke', 'Clay', 
            'Cline', 'Clyde', 'Cohen', 'Collins', 'Comer', 'Cook', 'Cooper', 
            'Cox', 'Craig', 'Crawford', 'Crist', 'Curtis', 'Davidson', 'Day', 
            'Deal', 'Dent', 'Doyle', 'Duncan', 'Early', 'Edwards', 'Evans', 
            'Fallon', 'Fish', 'Ford', 'Foster', 'Fox', 'Frankel', 'Franklin', 
            'Frost', 'Gomez', 'Gonzales', 'Gonzalez', 'Gosar', 'Graves', 
            'Gray', 'Griffith', 'Gross', 'Guthrie', 'Hand', 'Harris', 'Hart', 
            'Hatch', 'Hayes', 'Head', 'Heard', 'Hern', 'Higgins', 'Himes', 
            'Holden', 'Holt', 'Horn', 'House', 'Howe', 'Hudson', 'Huffman', 
            'Hull', 'Hyde', 'James', 'Jordan', 'Joyce', 'Kane', 'Kean', 
            'Keller', 'Kerr', 'Key', 'Kidd', 'Kim', 'Kind', 'Kirk', 'Lamb', 
            'Lane', 'Law', 'Lawrence', 'Lawson', 'Long', 'Love', 'Low', 
            'Lucas', 'Lynch', 'Mack', 'Marsh', 'May', 'Mayo', 'Mead', 'Meek', 
            'Moon', 'Moss', 'Mott', 'Murphy', 'North', 'Owens', 'Page', 
            'Palmer', 'Park', 'Peck', 'Pell', 'Perry', 'Peters', 'Phillips', 
            'Pike', 'Pitt', 'Polk', 'Pool', 'Pope', 'Porter', 'Post', 'Price', 
            'Pryor', 'Ray', 'Read', 'Reid', 'Rice', 'Rich', 'Robb', 'Rogers', 
            'Root', 'Rush', 'Sage', 'Sand', 'Sewell', 'Shaw', 'Sherman', 
            'Simpson', 'Sims', 'Sloan', 'Snow', 'Sparks', 'Stark', 'Steele', 
            'Stevens', 'Stewart', 'Stone', 'Story', 'Stout', 'Swan', 'Swift', 
            'Taft', 'Tate', 'Terry', 'Thompson', 'Todd', 'Torres', 'Turner', 
            'Wade', 'Wall', 'Ward', 'Ware', 'Watson', 'Watt', 'Way', 'Webb', 
            'Weeks', 'Weir', 'Wells', 'West', 'Wise', 'Wolf', 'Wood', 'Woods', 
            'Wynn', 'York'
        }
        
        # ambiguous_last_names.update(common_word_names)
        
        # Filter unique last names
        # rep_unique_last = rep_last_names - ambiguous_last_names
        # dem_unique_last = dem_last_names - ambiguous_last_names
        
        # Combine
        # Only use full names as requested
        final_reps = list(rep_full_names)
        final_dems = list(dem_full_names)
        
        print(f"Generated {len(final_reps)} Republican terms and {len(final_dems)} Democrat terms.")
        # print(f"Excluded {len(ambiguous_last_names)} ambiguous/common last names (e.g., {list(ambiguous_last_names)[:5]}).")
        
        return sorted(final_reps, key=len, reverse=True), sorted(final_dems, key=len, reverse=True)

if __name__ == "__main__":
    # Test
    mgr = PoliticianDataManager()
    r, d = mgr.get_party_lists()
    print("\nSample Reps:", r[:10])
    print("\nSample Dems:", d[:10])
