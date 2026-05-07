#!/usr/bin/env python3
"""
Verify dataset structure and integrity.

Usage:
    python verify_data.py --data_dir ./data
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def verify_dataset(data_dir: Path):
    """Verify dataset structure and report statistics."""

    datasets = ['AirLock', 'ALTO', 'UAV_VisLoc', 'VETRA']

    print("Dataset Verification")
    print("=" * 60)

    total_pairs = 0
    total_questions = 0
    task_counts = Counter()

    for dataset_name in datasets:
        dataset_path = data_dir / dataset_name
        json_dir = dataset_path / 'vqa_json'
        img_dir = dataset_path / 'img_data'

        if not dataset_path.exists():
            print(f"\n{dataset_name}: NOT FOUND")
            continue

        # Count JSON files
        json_files = list(json_dir.glob('*.json')) if json_dir.exists() else []

        # Count questions and tasks
        questions = 0
        tasks = Counter()

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                questions += len(data['questions'])
                for q in data['questions']:
                    tasks[q['task']] += 1

        total_pairs += len(json_files)
        total_questions += questions
        task_counts.update(tasks)

        print(f"\n{dataset_name}:")
        print(f"  Pairs:     {len(json_files)}")
        print(f"  Questions: {questions}")
        if img_dir.exists():
            print(f"  Images:    {len(list(img_dir.iterdir()))} directories")

    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"\nTotal pairs:     {total_pairs}")
    print(f"Total questions: {total_questions}")

    print("\nQuestions by task:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Verify dataset structure')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return

    verify_dataset(data_dir)


if __name__ == '__main__':
    main()
