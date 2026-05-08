# GCS Persistence for the FAISS Semantic Index

The intelligence silo stores its long-term knowledge in a FAISS vector index
on disk. `GCSIndexSync` mirrors that index to a Google Cloud Storage bucket so
that:

- **Durability**: the index survives a container restart or VM wipe.
- **Shared mesh knowledge**: every node (yours, Matt's, etc.) reads from and
  writes to the same bucket, converging on a shared semantic memory over time.

---

## How the sync works

### Upload — on every `SemanticMemory.save()`

After writing the four local files (`index.faiss`, `metadata.json`,
`embeddings.npy`, `id_order.json`), `SemanticMemory.save()` calls:

```python
GCSIndexSync().upload(self.persistence_path)
```

Each file is pushed to:

```
gs://<GCS_BUCKET>/<GCS_SYNC_PATH>/<filename>
```

Example with defaults:

```
gs://gigaton-memory/intelligence-silo/shared-memory/index.faiss
gs://gigaton-memory/intelligence-silo/shared-memory/metadata.json
...
```

### Download — on cold start inside `SemanticMemory._load()`

Before reading local files, `SemanticMemory._load()` calls:

```python
GCSIndexSync().sync(self.persistence_path)
```

`sync()` logic:

1. If `metadata.json` is absent locally → pull everything from GCS.
2. If GCS `metadata.json` has a newer `updated` timestamp → pull.
3. Otherwise → local is current, no download.

This means a freshly deployed container or a new developer machine will
automatically hydrate its index from the shared bucket on first start.

---

## Sharing the index with Matt's node

Matt's node needs **the same two env vars** pointing at the same bucket and
path (see below). On his first `think()` cycle the silo will:

1. Find no local index.
2. Pull the shared index from GCS.
3. Begin contributing new entries back to the same bucket after each save.

No direct P2P connection is required — GCS is the shared medium.

---

## Required environment variables

| Variable | Description | Example |
|---|---|---|
| `GCS_BUCKET` | GCS bucket name | `gigaton-memory` |
| `GCS_SYNC_PATH` | Prefix inside the bucket | `intelligence-silo/shared-memory/` (default) |

If `GCS_BUCKET` is not set the silo runs in **local-only mode** — all
`GCSIndexSync` calls are no-ops and return immediately without error.

### Docker / Cloud Run

```yaml
env:
  - name: GCS_BUCKET
    value: gigaton-memory
  - name: GCS_SYNC_PATH
    value: intelligence-silo/shared-memory/
```

### Local `.env`

```
GCS_BUCKET=gigaton-memory
GCS_SYNC_PATH=intelligence-silo/shared-memory/
```

The service account or ADC credentials must have `storage.objects.create`
and `storage.objects.get` on the bucket. The `roles/storage.objectUser`
predefined role covers both.

---

## Verifying it works

After the first ingest run, check the bucket:

```bash
gcloud storage ls gs://gigaton-memory/intelligence-silo/shared-memory/
```

Expected output:

```
gs://gigaton-memory/intelligence-silo/shared-memory/embeddings.npy
gs://gigaton-memory/intelligence-silo/shared-memory/id_order.json
gs://gigaton-memory/intelligence-silo/shared-memory/index.faiss
gs://gigaton-memory/intelligence-silo/shared-memory/metadata.json
```

To force a fresh download on the next start, delete the local persistence
directory and restart the silo — `sync()` will pull from GCS automatically.

---

## Thread safety

All upload and download operations are serialised with a `threading.Lock`
inside `GCSIndexSync`. Concurrent `save()` calls from different threads will
queue rather than race.
