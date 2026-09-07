from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ganlinyang/RoboTwin-MeM",
    allow_patterns=["embodiments/**", "objects/**"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
)
