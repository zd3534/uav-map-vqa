# UAV-Map-VQA Benchmark

Visual Question Answering benchmark for UAV aerial imagery with satellite map grounding.

## Quick Start

### Setup (1 minute)

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
```

Get API key at: https://openrouter.ai/keys

### Run Evaluation

```bash
# Quick test (480 questions, ~30 min, ~$3)
python evaluate.py --data_dir ./data --models "qwen/qwen-vl-max" --n_per_task 20

# Full benchmark (2400 questions, ~2 hours, ~$15)
python evaluate.py --data_dir ./data --models "qwen/qwen-vl-max"

# Multiple models
python evaluate.py --data_dir ./data --models "qwen/qwen-vl-max" "gpt-4o"
```

### Verify Setup

```bash
python test_setup.py                      # Test environment
python verify_data.py --data_dir ./data   # Verify dataset
python inspect_sample.py --data_dir ./data # View sample question
```

## Dataset Structure

```
data/
├── AirLock/          # US dataset
├── ALTO/             # US dataset
├── UAV_VisLoc/       # China dataset
└── VETRA/            # Germany dataset

Each dataset contains:
├── img_data/         # UAV images, map tiles, visualizations
└── vqa_json/         # Question-answer pairs (JSON format)
```

### JSON Format

```json
{
  "pair_id": "abc123",
  "metadata": {
    "uav_image": "/path/to/uav.jpg",
    "uav_gps": {"lat": 30.387, "lon": -97.728},
    "map_size_meters": 500,
    "dataset": "AirLock"
  },
  "questions": [
    {
      "question_id": "abc123_task1_0",
      "task": "task1_map_to_uav",
      "question": "Which colored boxes on the map show objects visible in the UAV image?",
      "options": {
        "R": "Red box on map",
        "Y": "Yellow box on map",
        "B": "Blue box on map"
      },
      "answer": ["R", "Y"],
      "uav_img_path": "pair_abc123/uav_raw.jpg",
      "map_img_path": "pair_abc123/map_tile.jpg",
      "vis_img_path": "pair_abc123/vis_task1_0.jpg"
    }
  ]
}
```

## Tasks

The benchmark includes 6 VQA task types:

1. **Map-to-UAV**: Given colored boxes on map, identify which are visible in UAV view
2. **UAV-to-Map**: Given colored boxes in UAV, locate corresponding objects on map
3. **Visibility**: Determine which map objects are visible in UAV view
4. **Spatial Referring**: Directional reasoning (e.g., "What is left of landmark X?")
5. **Object Context**: Spatial relationships (e.g., "Which parking is nearest to X?")
6. **Multi-step Reasoning**: Chain reasoning across both views

## Evaluation

### Metrics

- **Exact Match**: 1.0 if predicted set equals ground truth set, else 0.0
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall

Results are aggregated overall, by task type, and by dataset.

### Sampling

- Default: 100 questions per task (6 tasks × 100 = 600 base questions)
- Questions sampled uniformly across 4 datasets
- Random seed for reproducibility (default: 42)

### Output

Results saved to `results/<model_name>/<timestamp>/`:

- `results.json` - Per-question predictions and metrics
- `summary.txt` - Overall statistics

## Supported Models

Via OpenRouter API (https://openrouter.ai/models):

**Recommended:**
- `qwen/qwen-vl-max` - Qwen VL Max (best quality)
- `qwen/qwen-vl-plus` - Qwen VL Plus (balanced)
- `openai/gpt-4o` - GPT-4o
- `openai/gpt-4o-mini` - GPT-4o Mini
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `google/gemini-2.0-flash-exp` - Gemini 2.0 Flash

**Others:**
- `minimax/minimax-01`
- `mistralai/pixtral-large-latest`

## Command Reference

### Evaluation Options

```bash
python evaluate.py \
    --data_dir ./data \              # Dataset directory
    --models "model1" "model2" \     # Model IDs (space-separated)
    --output_dir ./results \         # Output directory
    --n_per_task 100 \               # Questions per task (default: 100)
    --seed 42 \                      # Random seed (default: 42)
    --delay 0.5 \                    # Delay between API calls in seconds
    --api_key "sk-or-..."            # API key (or use OPENROUTER_API_KEY env var)
```

### Common Use Cases

```bash
# Test a new model
python evaluate.py --data_dir ./data --models "model/name" --n_per_task 20

# Handle rate limits
python evaluate.py --data_dir ./data --models "..." --delay 1.0

# Custom random seed
python evaluate.py --data_dir ./data --models "..." --seed 123

# Evaluate subset
python evaluate.py --data_dir ./data --models "..." --n_per_task 50
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "No API key" | `export OPENROUTER_API_KEY="sk-or-v1-..."` |
| "Data not found" | Ensure `./data/` contains dataset folders |
| "Module not found" | Run `pip install -r requirements.txt` |
| "Rate limited" | Add `--delay 1.0` or higher |

## Cost Estimation

Approximate costs per full evaluation (2400 questions):

- **GPT-4o**: ~$10-15
- **Qwen VL Max**: ~$5-8
- **GPT-4o Mini**: ~$2-3

Quick test (n_per_task=20, 480 questions): ~1/5 of full cost

## Files

**Core Evaluation:**
- `evaluate.py` - Main evaluation script
- `verify_data.py` - Dataset verification utility
- `inspect_sample.py` - Sample question viewer
- `test_setup.py` - Environment test
- `requirements.txt` - Dependencies (openai, pillow, numpy, requests)

**Utilities (optional):**
- `utils/osm_extractor.py` - Extract OSM features from .osm.pbf files
- `utils/tile_downloader.py` - Download satellite/OSM map tiles

### Utility Usage

**Download map tiles:**
```bash
python utils/tile_downloader.py \
    --lat 30.387 --lon -97.728 --size 500 \
    --zoom 18 --server satellite \
    --output map_tile.jpg
```

**Extract OSM features:**
```bash
# Requires: pip install osmium
python utils/osm_extractor.py \
    --lat 30.387 --lon -97.728 --size 500 \
    --input us-latest.osm.pbf \
    --output ./osm_features/
```

These utilities are provided for reference but not required for benchmark evaluation.

## License

MIT License - see LICENSE file for details.
