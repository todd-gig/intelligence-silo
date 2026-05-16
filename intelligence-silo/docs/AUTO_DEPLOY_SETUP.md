# Auto-deploy on merge — one-time setup

After this is set up once, every PR merged to `main` automatically deploys to Cloud Run via Cloud Build. No more manual `gcloud builds submit`. Same pattern `gigaton-ui-system` uses for Firebase Hosting.

## One-time steps (~5 min wall-clock)

Run as `todd@gigaton.ai`:

### 1. Create the deployer service account

```bash
gcloud iam service-accounts create silo-gha-deployer \
  --display-name="intel-silo GitHub Actions deployer" \
  --project=carmen-beach-properties
```

### 2. Grant the roles it needs

```bash
SA="silo-gha-deployer@carmen-beach-properties.iam.gserviceaccount.com"
PROJECT=carmen-beach-properties

for role in \
    roles/cloudbuild.builds.editor \
    roles/iam.serviceAccountUser \
    roles/run.admin \
    roles/storage.admin \
    roles/secretmanager.viewer \
    roles/logging.viewer; do
  gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:$SA" \
    --role="$role" --condition=None
done
```

`secretmanager.viewer` lets the conditional cloudbuild logic check whether `google-drive-oauth-client` + `oauth-state-secret` exist (it reads `gcloud secrets describe` during the deploy step). All other roles same as HME setup.

### 3. Generate the SA key

```bash
gcloud iam service-accounts keys create /tmp/silo-gha-key.json \
  --iam-account=$SA \
  --project=$PROJECT
```

### 4. Set the GitHub secret

```bash
gh secret set GCP_SA_KEY_CARMEN_BEACH_PROPERTIES \
  --repo=todd-gig/intelligence-silo \
  --body="$(cat /tmp/silo-gha-key.json)"
```

### 5. Delete the local key file

```bash
shred -u /tmp/silo-gha-key.json 2>/dev/null || rm -P /tmp/silo-gha-key.json
```

## Verification

Make a trivial commit to main — the `Auto-deploy on merge to main` workflow should fire under [Actions tab](https://github.com/todd-gig/intelligence-silo/actions). Build takes ~5 min (PyTorch wheel + sentence-transformers); the workflow then health-checks `/health` and fails if not 200.

## What this changes

| Before | After |
|--------|-------|
| Every PR merge → operator runs `gcloud builds submit` manually | Every PR merge → GHA fires `gcloud builds submit` automatically |
| Risk of forgotten redeploys | Zero risk — every merge ships |
| Multi-Claude collision if two threads both deploy | Single trigger from main — no collision |
