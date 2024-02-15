# set the google gemini api key
GOOGLE_API_KEY = "AIzaSyBQOKOLox8xpKHZ8qTLdVXYpAm_-Sg6CFs"

# set the tags
TAGS_TO_IGNORE = ["script", "style"],
TAGS_TO_EXTRACT = ["span", "p", "h1", "h2", "h3", "h4", "h5", "h6"]

# define chunk size and overlap
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 128
K = 10

# list of arxiv codes to scrape
ARXIV_CODES = [
    "1706.03762", # Attention is All You Need
    "1810.04805", # BERT
    "2305.10435", # GPT-1
    "2210.13382", # Othello GPT Paper
    "2111.00396", # S4 Models
]


# set persist directory
CHROMA_PERSIST_DIR = ".DB"