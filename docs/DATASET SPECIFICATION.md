DATASET SPECIFICATION — NORMALIZED GRAPH FORMAT

File: data/pages.jsonl

Description:
Stores unique Wikipedia articles discovered during crawl.

Schema:

* page_id (int) — MediaWiki page ID
* title (str) — Canonical article title

Example:
{
"page_id": 11600,
"title": "Artificial intelligence"
}

Constraints:

* One record per page
* No duplicates
* No redirects stored separately

---

File: data/links.jsonl

Description:
Stores directed hyperlinks between articles.

Schema:

* source_id (int)
* target_id (int)

Example:
{
"source_id": 11600,
"target_id": 18978754
}

Constraints:

* Directed edges
* Article namespace only
* No duplicate edges

---

Graph Interpretation:

G = (V, E)

V = pages.jsonl records
E = links.jsonl records

Adjacency reconstruction handled downstream.

End of Spec
