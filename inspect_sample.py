#!/usr/bin/env python3
"""
Display a sample question from the dataset.

Usage:
    python inspect_sample.py --data_dir ./data
"""

import argparse
import json
import random
from pathlib import Path


def show_sample(data_dir: Path, dataset_name: str = None):
    """Show a random sample question."""

    datasets = ['AirLock', 'ALTO', 'UAV_VisLoc', 'VETRA']

    # Pick random dataset if not specified
    if dataset_name is None:
        dataset_name = random.choice(datasets)

    json_dir = data_dir / dataset_name / 'vqa_json'

    if not json_dir.exists():
        print(f"Error: {json_dir} not found")
        return

    # Pick random JSON file
    json_files = list(json_dir.glob('*.json'))
    if not json_files:
        print(f"Error: No JSON files found in {json_dir}")
        return

    json_file = random.choice(json_files)

    # Load and show
    with open(json_file) as f:
        data = json.load(f)

    print("=" * 70)
    print("SAMPLE QUESTION FROM DATASET")
    print("=" * 70)
    print(f"\nDataset:   {dataset_name}")
    print(f"Pair ID:   {data['pair_id']}")
    print(f"File:      {json_file.name}")

    # Show metadata
    print("\nMetadata:")
    for key, value in data['metadata'].items():
        print(f"  {key}: {value}")

    # Show first question
    if data['questions']:
        q = data['questions'][0]
        print("\n" + "-" * 70)
        print(f"Question ID:   {q['question_id']}")
        print(f"Task:          {q['task']}")
        print(f"\nQuestion:      {q['question']}")
        print("\nOptions:")
        for opt, desc in q['options'].items():
            marker = "✓" if opt in q['answer'] else " "
            print(f"  [{marker}] {opt}: {desc}")
        print(f"\nAnswer:        {q['answer']}")
        print(f"\nImages:")
        print(f"  UAV:         {q['uav_img_path']}")
        print(f"  Map:         {q['map_img_path']}")
        print(f"  Viz:         {q['vis_img_path']}")

    print("\n" + "=" * 70)
    print(f"Total questions in this pair: {len(data['questions'])}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Inspect sample question')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, choices=['AirLock', 'ALTO', 'UAV_VisLoc', 'VETRA'],
                       help='Specific dataset to sample from')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible sampling')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return

    show_sample(data_dir, args.dataset)


if __name__ == '__main__':
    main()
