# WikiCrawler

**Find link paths across Wikipedia pages — BFS, Best-First, or Bidirectional — with AI-powered explanations.**

WikiCrawler is a Python command-line tool that finds a link path between two Wikipedia articles using only internal article links. It supports multiple search strategies (guaranteed shortest or heuristic) and uses Google's Gemini AI to provide rich, context-aware explanations of the path found.

---

## Features

- **Search Strategies**:
  - `bfs`: Breadth-first search (guaranteed shortest path).
  - `best`: Heuristic best-first search (fast).
  - `bidi`: **Bidirectional BFS** (fastest & exact, default).
- **AI Integration**: Uses Google Gemini (via `gemini-2.5-flash`) to explain *why* links exist and score potential paths.
- **Visualizations**: Generates flowcharts (PNG) of the crawl graph.
- **Robust**: Retries on API errors, handles missing keys gracefully, and caches results.

---

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/wiki_crawler.git
    cd wiki_crawler
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Create a `requirements.txt` with `requests`, `networkx`, `matplotlib`, `beautifulsoup4`, `python-dotenv`)*

4.  **Configure API Key** (Optional but recommended):
    Create a `.env` file in the root directory:
    ```bash
    GEMINI_API_KEY="your_google_ai_studio_key"
    ```
    Get a key from [Google AI Studio](https://aistudio.google.com/).

---

## Usage

**Basic Path Finding:**
```bash
python3 wiki_crawler.py --start "Python (programming language)" --target "C (programming language)"
```

**With AI Explanations & Flowchart:**
```bash
python3 wiki_crawler.py \
  --start "Narendra Modi" \
  --target "Nuclear weapon" \
  --strategy bidi \
  --use-llm-hopping \
  --flowchart path_graph.png
```

---

## Testing

Run the unit tests:
```bash
python3 test_wiki_crawler.py
```