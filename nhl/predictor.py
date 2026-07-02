#!/usr/bin/env python3
"""NHL match prediction CLI."""
from __future__ import annotations

import argparse

from .model import predict_match


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict an NHL matchup")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--model", choices=["blend", "power", "form"], default="blend")
    args = ap.parse_args()

    pred = predict_match(args.home, args.away, neutral=args.neutral, model=args.model)
    venue = "neutral site" if args.neutral else f"{args.home} at home"
    print(f"{args.home} vs {args.away} ({venue}, model={args.model})")
    print(f"  P({args.home} win) = {pred['p_home']:.1%}")
    print(f"  P({args.away} win) = {pred['p_away']:.1%}")
    print(f"  Projected goals: {args.home} {pred['lambda_home']:.2f}, "
          f"{args.away} {pred['lambda_away']:.2f}")
    print(f"  Total: {pred['total']:.1f}   {args.home} -1.5: {pred['p_home_minus_1_5']:.1%}")


if __name__ == "__main__":
    main()
