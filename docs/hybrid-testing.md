# Enable real hybrid and reranker testing

This is an initial testing baseline, not an assertion that these models or score floors are
optimal for Persian. Chat generation remains on the existing configured model.

## Local settings

Keep the existing provider URL, key and signing secret private. Set these non-secret fields in
the project `.env`:

```dotenv
HR_RETRIEVAL_MODE=full_context
HR_API_EMBEDDING_MODEL=text-embedding-3-small
HR_API_EMBEDDING_REVISION=1
HR_RERANKER_MODEL=.runtime/models/bge-reranker-v2-m3
HR_RERANKER_REVISION=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
HR_RERANKER_LOCAL_FILES_ONLY=true
```

The hosted provider must actually support that embedding model on the configured embeddings
endpoint. This model produces 1536-dimensional vectors by default. Imports and queries use the
same model identity. Provider calls incur its normal charges and send evidence/questions to it.

## Generate course embeddings

Back up the SQLite database first, using SQLite's backup API when the service is running (a raw
file copy can omit WAL changes). Then run from the project folder in PowerShell:

```powershell
$previousRetrievalMode = $env:HR_RETRIEVAL_MODE
try {
    $env:HR_RETRIEVAL_MODE = 'hybrid'
    .\.venv\Scripts\python.exe -m hrlearnium reindex --tenant customer-demo --course captain-storm
} finally {
    $env:HR_RETRIEVAL_MODE = $previousRetrievalMode
}
.\.venv\Scripts\python.exe -m hrlearnium verify-source --tenant customer-demo --course captain-storm
```

The override applies to the indexing command, not the server default. Reindexing creates a new
document revision and new passage IDs; source bytes/text stay unchanged, and previous revisions
remain stored. It preserves chapter-aware chunking. A model/provider/revision change requires
reindexing again; do not compare query vectors with an old embedding identity.

## Obtain the local reranker

The optional runtime must be installed (`pip install -e ".[rerank]"` in this environment).
Download the pinned revision once; this is roughly a 2.3 GB download:

```powershell
$env:HF_HOME = "$PWD\.runtime\huggingface"
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-reranker-v2-m3', revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e', local_dir='.runtime/models/bge-reranker-v2-m3', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','README.md'], token=False, max_workers=2)"
Get-FileHash -Algorithm SHA256 .runtime/models/bge-reranker-v2-m3/model.safetensors
```

The expected weight-file SHA-256 for this revision is
`d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`.
The local `.runtime` directory is ignored by Git. Runtime loading stays local-only with remote
model code disabled. Preserve the pinned revision/checksum when reproducing the setup.

## Test

Restart the server to read `.env`, open `/chat`, connect the course and choose **Refresh status**.
All three retrieval modes should be available if setup checks pass. Keep **Cited explanation**
selected for LLM-written answers, and enable **Test details** to inspect the actual stages:

- Full context: question planning, per-part selection, explanation, per-part verification; no query embeddings or reranker.
- Hybrid: planning, batched query-variant embeddings, fused BM25/cosine search and the provisional relevance filter, then selection/explanation/verification.
- Hybrid + reranker: adds local cross-encoder scoring before LLM selection.

In policy v4, reranked mode keeps its broader positive-score pool before scoring rather than
discarding candidates at the plain-hybrid cosine floor. All modes still require complete semantic
support. See [the upgrade guide](retrieval-upgrade-v4.md) for changed costs and offline calibration.

The first ranked request imports the ML runtime and loads the CPU model; it is slower than a
warm request. Compare the same self-contained question after switching modes. Switching modes
starts a fresh conversation. Prerequisites alone do not prove inference or semantic quality.

Reranking returns raw logits, not calibrated confidence probabilities. Keep the optional raw
reranker floor unset until measured on labeled supported/unsupported course questions. The
existing cosine/BM25 floors are likewise provisional. A few successful smoke requests are not
a production evaluation. Include English/Persian language consistency, refusals, citation support,
retrieval coverage and latency in the next evaluation.

References: [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings),
[BGE model card](https://huggingface.co/BAAI/bge-reranker-v2-m3),
[CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
