DATASET SPECIFICATION — NORMALIZED GRAPH FORMAT

File: data/pages_raw.jsonl

Description:
Stores raw Wikipedia page snapshots discovered during crawl (including semantic links).

Schema:

* page_id (int) — MediaWiki page ID
* title (str) — Canonical article title
* extract (str | null) — lead intro plaintext (if available)
* categories ([str])
* sections ([{section_title, level}])
* links ([{target_title, anchor, section}])

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

File: data/links_table.jsonl

Description:
Stores directed hyperlinks between articles derived post-crawl from sanitized pages.

Schema:

* source_page_id (int)
* target_page_id (int)
* anchor_clean (str)
* section_clean (str)

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

V = pages_raw/pages_sanitized records
E = links_table records

Adjacency reconstruction handled downstream.

End of Spec
