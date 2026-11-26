"""GPQA entry point of the sentence ablation in ``run_sentence_ablation_all.py``."""
import argparse
import os

from dotenv import load_dotenv

import run_sentence_ablation_all as ablation
from llm_client import create_client

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ablation.add_arguments(parser)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    ablation.configure(args)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    ablation.run(create_client(ablation.PROVIDER), "gpqa", args)


if __name__ == "__main__":
    main()
