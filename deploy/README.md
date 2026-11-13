# Cloud Run L4 transcription demo

The container contains a pinned snapshot of `ivrit-ai/whisper-large-v3-turbo`
(`f33172a8c3c6efbc040a7200e00835257cac0447`) and two **adapter-only**
checkpoints from the Technion server. Do not commit `deploy/assets/`.

```bash
mkdir -p deploy/assets/{model,digits,cardinals}
scp technion-compute-113:/home/yoadsnapir/whisper/biased-whisper/data/runs/phase4-20261107-173859-1395/epoch-01-step-001681/adapter.{json,safetensors} deploy/assets/cardinals/
scp technion-compute-113:/home/yoadsnapir/whisper/biased-whisper/data/runs/phase4-digits-20261107-204452-834d/epoch-01-step-001269/adapter.{json,safetensors} deploy/assets/digits/
uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download("ivrit-ai/whisper-large-v3-turbo", revision="f33172a8c3c6efbc040a7200e00835257cac0447", local_dir="deploy/assets/model", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])'
uv run python -m deploy.select_presets  # 20 seeded random short WAVs from data/corpus/audio/
docker buildx build --platform linux/amd64 --load -f deploy/Dockerfile -t biased-whisper-l4:local .
```

The model is the **ivrit.ai** fine-tune, not OpenAI's generic Whisper turbo;
both adapters were trained against this model. The slider is -100 (digits),
0 (untouched base), +100 (cardinals). Intermediate positions scale only the
selected adapter. The interface offers microphone recording (HTTPS required),
drag-and-drop uploads, and 20 built-in preset clips. `deploy/select_presets.py`
shuffles the local WAV corpus with a fixed seed and filters clips to 0.5–30 s.
The selected audio and its source list live only under gitignored
`deploy/assets/presets/`; the image includes the audio and a display-only
manifest, **not** the source file names. Authorized users can play and
transcribe these clips, so choose a corpus suitable for those users before
publishing the image. The inference server permits a single transcription at
once, restricts uploads to 25 MB / 30 seconds, and returns Hebrew text without
automatic number correction. It does not accept longer clips yet.

To publish a locally built image, tag/push it to Artifact Registry and deploy
using the **personal account** for every gcloud operation (never rely on the
active default, which may be the work account):

```bash
IMAGE=europe-west1-docker.pkg.dev/project-71583fab-5cd0-4bba-a6f/hebrew-tts/biased-whisper-l4:20261113
docker tag biased-whisper-l4:local "$IMAGE"
# Log in with gcloud auth print-access-token --account=yoadsn@gmail.com;
# use an isolated DOCKER_CONFIG so the credential is not stored in the user's usual Docker config.
DOCKER_CONFIG="$(mktemp -d)"; export DOCKER_CONFIG
gcloud auth print-access-token --account=yoadsn@gmail.com | docker login -u oauth2accesstoken --password-stdin https://europe-west1-docker.pkg.dev
# An isolated config cannot resolve named contexts; point Docker at Colima's socket.
DOCKER_HOST="$(docker --config "$HOME/.docker" context inspect colima --format '{{.Endpoints.docker.Host}}')"; export DOCKER_HOST
docker push "$IMAGE"
gcloud run deploy biased-whisper-l4 --image "$IMAGE" \
  --account=yoadsn@gmail.com --project=project-71583fab-5cd0-4bba-a6f --region=europe-west1 \
  --execution-environment=gen2 --port=8080 --cpu=4 --memory=16Gi \
  --gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu-boost --no-cpu-throttling --concurrency=1 --timeout=900 \
  --min-instances=0 --max-instances=1 --no-allow-unauthenticated --iap \
  --startup-probe=httpGet.path=/ready,periodSeconds=10,timeoutSeconds=10,failureThreshold=24
```

IAP protects the normal browser URL with Google sign-in. An IAM-only Cloud
Run service returns `Forbidden` in a browser because ordinary navigation does
not send a Cloud Run ID token. For an existing service, enable direct IAP and
grant it the only Cloud Run Invoker permission it needs:

```bash
gcloud run services update biased-whisper-l4 --iap --account=yoadsn@gmail.com \
  --project=project-71583fab-5cd0-4bba-a6f --region=europe-west1
gcloud run services add-iam-policy-binding biased-whisper-l4 \
  --member=serviceAccount:service-843531170072@gcp-sa-iap.iam.gserviceaccount.com \
  --role=roles/run.invoker --account=yoadsn@gmail.com \
  --project=project-71583fab-5cd0-4bba-a6f --region=europe-west1
gcloud iap web add-iam-policy-binding --member=user:yoadsn@gmail.com \
  --role=roles/iap.httpsResourceAccessor --resource-type=cloud-run \
  --service=biased-whisper-l4 --region=europe-west1 \
  --project=project-71583fab-5cd0-4bba-a6f --account=yoadsn@gmail.com
```

The service allowlist currently contains `yoadsn@gmail.com`,
`yair@lifshitz.io` and `yanirmr@gmail.com`. Repeat the last command with any
additional reviewer email. Do **not** use `allUsers`:
public GPU inference can incur unbounded charges. If a Gmail account outside
the project's Google organization is denied *after* login, configure an
External OAuth client for IAP in the Cloud Console; see [Google's Cloud Run
IAP guide](https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run#custom-oauth-client).
Minimum instances zero allows scale-to-zero; the first request can have a
long cold start.

Deployed revision `biased-whisper-l4-00002-pmq` is ready at
<https://biased-whisper-l4-que2njye2q-ew.a.run.app> (IAP Google sign-in;
service-scoped user allowlist). The image was pushed with digest
`sha256:49899a55f3ef5fded8b475209e483d815b5582c67c66264d1fcc3660bcd25410`.
