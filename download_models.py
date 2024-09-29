# download models from KoeAI/llvc on Huggingface Hub
from huggingface_hub import snapshot_download
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hubert_only', action='store_true', 
                        help='whether to only download the Hubert model')
    args = parser.parse_args()

    # download models from KoeAI/llvc on Huggingface Hub
    if args.hubert_only:
        snapshot_download(repo_id="KoeAI/llvc",
                        local_dir='llvc_models',
                        allow_patterns=["*hubert_base*"])
    else:
        snapshot_download(repo_id="KoeAI/llvc",
                        local_dir='llvc_models')
    

if __name__ == "__main__":
    main()