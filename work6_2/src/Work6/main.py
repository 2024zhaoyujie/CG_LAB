import argparse

from .config import build_config
from .train_silhouette import run_silhouette
from .train_textured import run_textured


def parse_args():
    parser = argparse.ArgumentParser(description="Work6 differentiable mesh fitting")
    parser.add_argument("mode", choices=["silhouette", "textured", "both"])
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-views", type=int, default=18)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args.mode, args.steps, args.image_size, args.num_views, args.seed)

    if args.mode in ("silhouette", "both"):
        run_dir = run_silhouette(cfg)
        print(f"silhouette outputs: {run_dir}")
    if args.mode in ("textured", "both"):
        cfg.mode = "textured"
        cfg.train.seed += 1
        run_dir = run_textured(cfg)
        print(f"textured outputs: {run_dir}")


if __name__ == "__main__":
    main()
