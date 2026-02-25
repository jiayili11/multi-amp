# upload_to_hf.py
import os
from huggingface_hub import HfApi, upload_file
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set in environment. Export it first: export HF_TOKEN='hf_xxx'")

repo_id = "jiayi11/multi_amp"  # target repo
local_root = Path.cwd()

api = HfApi()

# 1) create repo if not exists
try:
    api.create_repo(repo_id=repo_id, private=False, token=HF_TOKEN)
    print(f"Created repo {repo_id}")
except Exception as e:
    # repo may already exist
    print(f"Repo create: {e}")

# helper to upload a file
def hf_upload(local_path: Path, path_in_repo: str):
    print(f"Uploading {local_path} ({local_path.stat().st_size / 1024 / 1024:.1f} MB) -> {repo_id}/{path_in_repo} ...")
    try:
        upload_file(
            token=HF_TOKEN,
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",  # 使用 model 类型
            create_pr=False
        )
        print(f"✅ Done: {path_in_repo}")
    except Exception as e:
        print(f"❌ Failed to upload {local_path}: {e}")

# files to upload
files = []

# checkpoints
ckpt = local_root / "checkpoints" / "best_model_overall.pth"
if ckpt.exists():
    files.append((ckpt, "checkpoints/best_model_overall.pth"))
else:
    print("Warning: checkpoint not found:", ckpt)

# data archive
data_archive = local_root / "data.tar.gz"
if not data_archive.exists():
    # if not exists, try to create it (tar.gz)
    data_dir = local_root / "data"
    if data_dir.exists():
        print("Creating data.tar.gz ... (this may take a while)")
        import tarfile
        with tarfile.open(data_archive, "w:gz") as tar:
            tar.add(data_dir, arcname="data")
        print("Created", data_archive)
    else:
        print("Warning: data directory not found:", data_dir)

if data_archive.exists():
    files.append((data_archive, "data.tar.gz"))

# perform uploads
for local_path, path_in_repo in files:
    hf_upload(local_path, path_in_repo)

print("All uploads complete.")