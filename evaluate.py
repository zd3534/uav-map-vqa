#!/usr/bin/env python3
"""
Evaluate vision-language models on UAV-Map-VQA benchmark.

Usage:
    python evaluate.py --data_dir ./data --models "qwen/qwen-vl-max" --output_dir ./results
"""

import argparse
import base64
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def build_prompt(question: dict) -> str:
    """Build prompt from question dict."""
    options_text = "\n".join(
        f"  {k}: {v}" for k, v in question["options"].items()
    )

    prompt = f"""Answer the following question about the UAV aerial image and satellite map.

Question: {question['question']}

Options:
{options_text}

Provide your answer as the option letter(s) only (e.g., "R" or "R,Y").
For questions where no objects match, answer "None"."""

    return prompt


def parse_response(response: str) -> list:
    """Parse model response to extract answer letters."""
    response = response.strip()

    # Check for "None"
    if re.search(r'\b(none|no objects?|nothing)\b', response, re.IGNORECASE):
        return []

    # Extract letters (R, Y, B, G)
    letters = re.findall(r'\b[RYBG]\b', response.upper())

    # Remove duplicates
    seen = set()
    result = []
    for letter in letters:
        if letter not in seen:
            seen.add(letter)
            result.append(letter)

    return result


def compute_metrics(predicted: list, ground_truth: list) -> dict:
    """Compute evaluation metrics."""
    pred_set = set(predicted)
    gt_set = set(ground_truth)

    # Exact match
    exact_match = 1.0 if pred_set == gt_set else 0.0

    # Precision, Recall, F1
    if len(pred_set) == 0:
        precision = 0.0
        recall = 0.0 if len(gt_set) > 0 else 1.0
    else:
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set)
        recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'exact_match': exact_match,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_question(client: OpenAI, model: str, question: dict,
                     data_dir: Path, delay: float = 0.5) -> dict:
    """Evaluate single question."""
    try:
        # Build paths
        uav_path = data_dir / question['dataset'] / 'img_data' / question['uav_img_path']
        map_path = data_dir / question['dataset'] / 'img_data' / question['map_img_path']

        # Encode images
        uav_b64 = encode_image(str(uav_path))
        map_b64 = encode_image(str(map_path))

        # Build prompt
        prompt = build_prompt(question)

        # API call
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{uav_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_b64}"}}
                ]
            }],
            max_tokens=100,
            temperature=0.0
        )

        time.sleep(delay)

        # Parse response
        model_response = response.choices[0].message.content
        predicted = parse_response(model_response)
        ground_truth = question['answer']

        # Compute metrics
        metrics = compute_metrics(predicted, ground_truth)

        return {
            'question_id': question['question_id'],
            'task': question['task'],
            'dataset': question['dataset'],
            'predicted': predicted,
            'ground_truth': ground_truth,
            'model_response': model_response,
            **metrics
        }

    except Exception as e:
        print(f"Error evaluating question {question.get('question_id', 'unknown')}: {e}")
        return None


def load_dataset(data_dir: Path, n_per_task: int = 100, seed: int = 42) -> List[dict]:
    """Load questions from all datasets."""
    random.seed(seed)

    datasets = ['AirLock', 'ALTO', 'UAV_VisLoc', 'VETRA']
    all_questions = []

    for dataset_name in datasets:
        json_dir = data_dir / dataset_name / 'vqa_json'
        if not json_dir.exists():
            print(f"Warning: {json_dir} not found, skipping...")
            continue

        # Load all JSON files
        for json_file in json_dir.glob('*.json'):
            with open(json_file) as f:
                data = json.load(f)
                for q in data['questions']:
                    q['dataset'] = dataset_name
                    all_questions.append(q)

    # Sample n_per_task from each task type
    tasks = defaultdict(list)
    for q in all_questions:
        tasks[q['task']].append(q)

    sampled = []
    for task, questions in tasks.items():
        sample_size = min(n_per_task, len(questions))
        sampled.extend(random.sample(questions, sample_size))

    print(f"Loaded {len(sampled)} questions from {len(datasets)} datasets")
    for task, questions in tasks.items():
        count = len([q for q in sampled if q['task'] == task])
        print(f"  {task}: {count}")

    return sampled


def evaluate_model(model: str, questions: List[dict], data_dir: Path,
                   output_dir: Path, api_key: str, delay: float = 0.5):
    """Evaluate model on all questions."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model}")
    print(f"{'='*60}\n")

    # Setup client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace('/', '_')
    out_dir = output_dir / model_slug / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Evaluating {question['question_id']}...", end=' ')

        result = evaluate_question(client, model, question, data_dir, delay)
        if result:
            results.append(result)
            print(f"EM={result['exact_match']:.2f}")
        else:
            print("FAILED")

    # Save results
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Compute summary statistics
    summary = compute_summary(results)

    # Save summary
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write(f"Model: {model}\n")
        f.write(f"Evaluated: {len(results)} questions\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("="*60 + "\n")
        f.write("OVERALL RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Exact Match:  {summary['overall']['exact_match']:.2%}\n")
        f.write(f"Precision:    {summary['overall']['precision']:.2%}\n")
        f.write(f"Recall:       {summary['overall']['recall']:.2%}\n")
        f.write(f"F1 Score:     {summary['overall']['f1']:.2%}\n\n")

        f.write("="*60 + "\n")
        f.write("BY TASK\n")
        f.write("="*60 + "\n\n")
        for task, metrics in sorted(summary['by_task'].items()):
            f.write(f"{task}:\n")
            f.write(f"  Exact Match: {metrics['exact_match']:.2%}\n")
            f.write(f"  F1 Score:    {metrics['f1']:.2%}\n")
            f.write(f"  Count:       {metrics['count']}\n\n")

        f.write("="*60 + "\n")
        f.write("BY DATASET\n")
        f.write("="*60 + "\n\n")
        for dataset, metrics in sorted(summary['by_dataset'].items()):
            f.write(f"{dataset}:\n")
            f.write(f"  Exact Match: {metrics['exact_match']:.2%}\n")
            f.write(f"  F1 Score:    {metrics['f1']:.2%}\n")
            f.write(f"  Count:       {metrics['count']}\n\n")

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_dir}")
    print(f"Overall Exact Match: {summary['overall']['exact_match']:.2%}")
    print(f"{'='*60}\n")

    return summary


def compute_summary(results: List[dict]) -> dict:
    """Compute summary statistics."""
    overall = defaultdict(list)
    by_task = defaultdict(lambda: defaultdict(list))
    by_dataset = defaultdict(lambda: defaultdict(list))

    for r in results:
        for metric in ['exact_match', 'precision', 'recall', 'f1']:
            overall[metric].append(r[metric])
            by_task[r['task']][metric].append(r[metric])
            by_dataset[r['dataset']][metric].append(r[metric])

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    summary = {
        'overall': {k: avg(v) for k, v in overall.items()},
        'by_task': {},
        'by_dataset': {}
    }

    for task, metrics in by_task.items():
        summary['by_task'][task] = {k: avg(v) for k, v in metrics.items()}
        summary['by_task'][task]['count'] = len(metrics['exact_match'])

    for dataset, metrics in by_dataset.items():
        summary['by_dataset'][dataset] = {k: avg(v) for k, v in metrics.items()}
        summary['by_dataset'][dataset]['count'] = len(metrics['exact_match'])

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on UAV-Map-VQA')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model IDs (e.g., qwen/qwen-vl-max)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    parser.add_argument('--n_per_task', type=int, default=100,
                       help='Questions per task (default: 100 = 2400 total)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API calls (seconds)')
    parser.add_argument('--api_key', type=str,
                       help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("Error: No API key provided. Set OPENROUTER_API_KEY or use --api_key")
        return

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    questions = load_dataset(data_dir, n_per_task=args.n_per_task, seed=args.seed)

    # Evaluate each model
    for model in args.models:
        evaluate_model(model, questions, data_dir, output_dir, api_key, args.delay)


if __name__ == '__main__':
    main()
