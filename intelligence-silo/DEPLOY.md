# Deploying the Intelligence Silo to Cloud Run

## One-time setup

1. Create the GCS bucket for the semantic index:
   ```bash
   gcloud storage buckets create gs://gigaton-silo-index \
     --project=carmen-beach-properties \
     --location=us-central1
   ```

2. Grant the Cloud Run service account access to the bucket:
   ```bash
   PROJECT_NUMBER=$(gcloud projects describe carmen-beach-properties --format='value(projectNumber)')
   gcloud storage buckets add-iam-policy-binding gs://gigaton-silo-index \
     --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
     --role="roles/storage.objectUser"
   ```

## Deploy

```bash
cd /Users/admin/Documents/GitHub/intelligence-silo
gcloud builds submit --config cloudbuild.yaml . \
  --project=carmen-beach-properties
```

Build takes ~15 minutes (PyTorch CPU wheel + sentence-transformers).

## After deploy

1. Get the service URL:
   ```bash
   gcloud run services describe intelligence-silo \
     --region=us-central1 --project=carmen-beach-properties \
     --format='value(status.url)'
   ```

2. Set it in Firebase Functions config:
   ```bash
   firebase functions:config:set gigaton.silo_url="https://intelligence-silo-XXXXX-uc.a.run.app"
   ```
   OR set as an environment variable in the Firebase Functions Cloud Run service.

3. Test:
   ```bash
   curl -X POST https://intelligence-silo-XXXXX-uc.a.run.app/process \
     -H "Content-Type: application/json" \
     -d '{"input_data": {"text": "hello"}, "deliberate": false}'
   ```

## Local → Cloud upgrade path

When a user runs the Gigaton desktop app locally, the chat UI automatically prefers
`localhost:8080` (or the URL they configure in Settings → Intelligence Keys → Gigaton).
When no local silo is running, the chat falls back to this Cloud Run service transparently.

Users get:
- **Cloud (default)**: always available, rate-limited by tier
- **Local (optional)**: unlimited usage, offline, private — no data leaves their machine
