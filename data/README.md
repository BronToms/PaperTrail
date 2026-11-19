# Demo data card

`papers.jsonl` contains ten original AI-assisted English research notes, each based
on the abstract of its linked arXiv paper (sources checked 2026-11-19).
They are paraphrases, not quotations, full papers, or author-endorsed summaries.
`kind=authored_research_note` is returned in every search result so users can
distinguish a note from an original article. Source offsets refer to the note.
The original papers are not bundled. Code and authored notes use the repository's
MIT license; this does not relicense referenced publications.

Topics: Transformer, BERT, DPR, RAG, ColBERT, LoRA, DPO, QLoRA, Sentence-BERT,
InstructGPT. Each record has a stable local ID, title, source URL, text and kind.
No personal data or user queries are collected by default.

`queries.jsonl` contains 20 answerable and 6 unsupported authored questions.
Labels are paper-level binary relevance judgments, not paragraph-level evidence
or human-verified answer labels. The questions were written with knowledge of
the corpus, so overlap and authoring bias are substantial. All use `split=demo`;
there is no claimed train/dev/test separation. Unsupported examples include both
out-of-domain requests and in-domain requests that the notes cannot answer.

Limitations: tiny English corpus, short texts, no equations/tables, no OCR,
incomplete relevance judgments, shared model-assisted authorship, no independent
annotation. Do not train or select thresholds on this set and call it a test set.
