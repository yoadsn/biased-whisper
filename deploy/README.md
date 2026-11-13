# Cloud Run L4 transcription demo

The container contains a pinned snapshot of `ivrit-ai/whisper-large-v3-turbo`
(`f33172a8c3c6efbc040a7200e00835257cac0447`) and two **adapter-only**
checkpoints from the Technion server. Do not commit `deploy/assets/`.

```bash
mkdir -p deploy/assets/{model,digits,cardinals}
scp technion-compute-113:/home/yoadsnapir/whisper/biased-whisper/data/runs/phase4-20261107-173859-1395/epoch-01-step-001681/adapter.{json,safetensors} deploy/assets/cardinals/
scp technion-compute-113:/home/yoadsnapir/whisper/biased-whisper/data/runs/phase4-digits-20261107-204452-834d/epoch-01-step-001269/adapter.{json,safetensors} deploy/assets/digits/
uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download("ivrit-ai/whisper-large-v3-turbo", revision="f33172a8c3c6efbc040a7200e00835257cac0447", local_dir="deploy/assets/model", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])'
docker buildx build --platform linux/amd64 --load -f deploy/Dockerfile -t biased-whisper-l4:local .
```

The model is the **ivrit.ai** fine-tune, not OpenAI's generic Whisper turbo;
both adapters were trained against this model. The slider is -100 (digits),
0 (untouched base), +100 (cardinals). Intermediate positions scale only the
selected adapter. The inference server permits a single transcription at once,
restricts uploads to 25 MB / 30 seconds, and returns Hebrew text without
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
  --min-instances=0 --max-instances=1 --no-allow-unauthenticated \
  --startup-probe=httpGet.path=/ready,periodSeconds=10,timeoutSeconds=10,failureThreshold=24
```

Use `gcloud run services proxy biased-whisper-l4 --account=yoadsn@gmail.com
--project=project-71583fab-5cd0-4bba-a6f --region=europe-west1 --port=8080`
and open <http://localhost:8080/>. Private-by-default is deliberate: making
GPU inference publicly callable can incur unbounded charges. Minimum instances
zero allows scale-to-zero; the first request can have a long cold start.
