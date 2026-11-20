import argparse
import json

from t20predict.model import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/raw/t20s_json.zip")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    result = train(args.archive, args.output)
    print(json.dumps({part: {k: v for k, v in result[part].items() if k != "calibration_bins"}
                      for part in ("train", "validation", "test")}, indent=2))
