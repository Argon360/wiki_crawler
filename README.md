# WikiCrawler

**Find link paths across Wikipedia pages — BFS, Best-First, or Bidirectional — with flowcharts and explanations.**

WikiCrawler is a Python command-line tool that finds a link path between two Wikipedia articles using only internal article links. It supports multiple search strategies (guaranteed shortest or heuristic), records decision metadata so it can explain each hop, and can generate a compact flowchart (PNG) highlighting the final path. Built to power the “Wikipedia Game” and to experiment with search strategies on real-world article graphs.

---

## Features

- Search strategies:
  - `bfs` — uni-directional breadth-first search (guaranteed shortest path in clicks).
  - `best` — greedy best-first using a title-similarity heuristic (fast, not guaranteed).
  - `bidi` — bidirectional BFS using `prop=linkshere` (exact shortest path, much faster).
- Verbose tracing of API calls, queue activity and decisions.
- Decision trace: stores *why* an edge was enqueued (method, depth, heuristic score when applicable).
- Explanation generator: human-readable step-by-step explanation of the final path.
- Flowchart generation (PNG) with multiple modes to avoid clutter:
  - `path-only`, `path-neighbors`, `pruned`, `full`.
- In-memory caching of `prop=links` and `prop=linkshere` responses for the run.
- Polite API usage with configurable sleep between requests.
- Optional explanation saved to file and PNG flowchart output.

---

## Requirements

- Python 3.9+
- Pip packages:
  - `requests`
  - `networkx`
  - `matplotlib`

Install dependencies:

```bash
pip3 install requests networkx matplotlib
