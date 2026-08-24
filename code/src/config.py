from pathlib import Path

# Paths
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "data"
DATA_PATH = str(DATA_ROOT)
PROCESSED_DATA_PATH = str(DATA_ROOT / "processed")


def set_data_root(repository_root: Path) -> None:
    """Set the data tree used by the pipeline."""
    global DATA_ROOT, DATA_PATH, PROCESSED_DATA_PATH
    DATA_ROOT = Path(repository_root).expanduser().resolve() / "data"
    DATA_PATH = str(DATA_ROOT)
    PROCESSED_DATA_PATH = str(DATA_ROOT / "processed")

# Party Terms
REPUBLICAN_TERMS = [
    # general_terms
    'republican', 'republicans', 'gop', 'republican party', 'grand old party',
    # leaders
    'trump', 'donald trump', 'pence', 'mike pence', 'mccarthy', 'kevin mccarthy',
    'mike johnson',  'steve scalise',
    # senate
    'mcconnell', 'mitch mcconnell', 'wicker', 'roger wicker', 
    'jim risch', 'susan collins', 'john mccain',
    'mitt romney', 'john cornyn',
    'thom tillis', 'todd young',
    'bill cassidy', 'shelley moore capito',
    'lisa murkowski', 'mike rounds',
    # senate
    'rand paul', 'hawley', 'josh hawley', 
    'vance', 'j.d. vance', 'jd vance', 'tuberville', 'tommy tuberville',
    'mike lee', 'mike braun', 'roger marshall',
     'cynthia lummis', 'rick scott', 'lindsey graham', 'ron johnson', 'marco rubio',
    # house
    'michael mccaul', 'mike rogers',
    'mike turner', 'fitzpatrick', 'brian fitzpatrick',
    'don bacon', 'tony gonzales',
    'mike lawler', 'diaz-balart', 'mario diaz-balart',
    # house
    'marjorie taylor greene', 'mtg', 'gaetz', 'matt gaetz', 'brian mast', 
    'thomas massie', 'chip roy', 'andy biggs',
    'boebert', 'lauren boebert', 'paul gosar', 'ralph norman',
    'scott perry', 'jim jordan',
    # governors
    'desantis', 'ron desantis', 'greg abbott',
    'glenn youngkin', 'mike dewine',
    'spencer cox', 'kay ivey', 
    'brian kemp', 'sarah huckabee sanders', 'doug burgum'
]

DEMOCRAT_TERMS = [
    # general_terms
    'democrat', 'democrats', 'democratic party',
    # leaders
    'biden', 'joe biden', 'joseph biden', 'kamala harris',
    'schumer', 'chuck schumer', 'charles schumer', 'pelosi', 'nancy pelosi',
    'hakeem jeffries', 'steny hoyer', 'obama', 'barack obama',
    # senate
    'dick durbin', 'patty murray',
    'jeanne shaheen',  'jack reed',
    'mark warner', 'bob menendez',
    'ben cardin', 'chris coons',
    'richard blumenthal', 'tim kaine',
    'chris murphy', 'cory booker',
    # house
    'gregory meeks', 'adam smith', 'rosa delauro',
    'marcy kaptur', 'mike quigley',
    'jason crow', 'elissa slotkin',
    'jared golden', 'abigail spanberger',
    'pramila jayapal', 'ilhan omar',
    'ocasio-cortez', 'aoc', 'alexandria ocasio-cortez',
    # governors
    'newsom', 'gavin newsom', 'whitmer', 'gretchen whitmer',
    'pritzker', 'jb pritzker', 'j.b. pritzker', 'phil murphy',
    'kathy hochul', 'josh shapiro',
    'andy beshear', 'tony evers'
]

ADMIN_OFFICIALS = [
    'blinken', 'antony blinken', 'tony blinken', 
    'lloyd austin', 'secretary austin',
    'sullivan', 'jake sullivan', 'national security advisor',
    'yellen', 'janet yellen', 'secretary yellen',
    'william burns', 'bill burns', 'cia director',
    'samantha power', 'usaid administrator',
    'nuland', 'victoria nuland', 'toria nuland',
    'mark milley', 'general milley', 'chairman milley',
    'charles brown', 'general brown',
    'thomas-greenfield', 'linda thomas-greenfield', 'ambassador thomas-greenfield'
]

# Standard political titles (full words, safe for \b boundaries)
TITLES = [
    "president", "vice president", "vp", 
    "senator", "representative", 
    "governor", "congressman", "congresswoman", 
    "secretary", "ambassador", 
    "speaker", "leader", "chair",
    "lawmaker", "legislator",
    "gov", "sen", "rep"
]

# Abbreviated titles (ending in period, require special regex handling)
TITLES_ABBREVIATED = [
    "sen.", "rep.", "gov.", "sec.", "amb.", "gen.", "dr.", "mr.", "ms.", "mrs."
]

# Party labels to remove
PARTY_LABELS = [
    "republican", "democrat", "gop", "dem", "dems", "reps", "grand old party", "republican party", 
    'democratic party'
]

# Protected terms (DO NOT MASK/REMOVE these even if they follow a title)
PROTECTED_TERMS = [
    'ukraine', 'russia', 'kyiv', 'kiev', 'moscow', 
    'zelenskyy', 'zelensky', 'zelenskiy', 'putin', 
    'kremlin', 'white house', 'pentagon', 'washington'
]

# Ukraine-related keywords for extraction
UKRAINE_KEYWORDS = {
    'core': ['ukraine', 'ukrainian', 'kyiv', 'kiev'],
    'people': ['zelenskyy', 'zelensky', 'zelenskiy'],
    'aid': ['ukraine aid', 'military aid', 'military assistance', 
           'ukraine funding', 'ukraine support', 'ukraine package',
           'aid package', 'assistance package', 'security assistance'],
    'conflict': ['invasion', 'war in ukraine',
                'ukraine conflict', 'ukraine crisis', 'special military operation']
}
