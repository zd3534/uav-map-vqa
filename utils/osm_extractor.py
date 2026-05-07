#!/usr/bin/env python3
"""
Extract OSM features from .osm.pbf files for a given GPS region.

Usage:
    python osm_extractor.py --lat 30.387 --lon -97.728 --size 500 \
        --input us-latest.osm.pbf --output ./output/

Requires: osmium (pip install osmium)
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


def crop_osm_region(input_pbf: str, center_lat: float, center_lon: float,
                    size_meters: float, output_pbf: str) -> bool:
    """
    Crop OSM PBF to bounding box around center point.

    Args:
        input_pbf: Path to large .osm.pbf file
        center_lat, center_lon: Center coordinates
        size_meters: Size of square region in meters
        output_pbf: Output cropped PBF file

    Returns:
        True if successful
    """
    # Approximate conversion: 1 degree lat ~ 111km, lon varies by latitude
    lat_delta = (size_meters / 2) / 111000
    lon_delta = (size_meters / 2) / (111000 * abs(center_lat / 90))

    bbox = f"{center_lon - lon_delta},{center_lat - lat_delta}," \
           f"{center_lon + lon_delta},{center_lat + lat_delta}"

    cmd = [
        'osmium', 'extract',
        '-b', bbox,
        input_pbf,
        '-o', output_pbf,
        '--overwrite'
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cropping OSM: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: osmium tool not found. Install: pip install osmium")
        return False


def parse_osm_to_geojson(input_pbf: str, output_dir: Path) -> Dict[str, str]:
    """
    Parse OSM PBF into GeoJSON feature collections.

    Args:
        input_pbf: Path to cropped .osm.pbf
        output_dir: Directory to save GeoJSON files

    Returns:
        Dict mapping layer name to output file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export layers using osmium
    layers = {
        'points': 'n/',     # Nodes (POIs)
        'lines': 'w/',      # Open ways (roads, paths)
        'polygons': 'w/',   # Closed ways (buildings, areas)
        'relations': 'r/'   # Relations (multipolygons)
    }

    output_files = {}

    for layer_name, osm_type in layers.items():
        output_file = output_dir / f"{layer_name}.geojson"

        cmd = [
            'osmium', 'export',
            input_pbf,
            '-o', str(output_file),
            '--geometry-types', 'point' if layer_name == 'points' else 'linestring,polygon',
            '--overwrite'
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            output_files[layer_name] = str(output_file)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not export {layer_name}: {e.stderr.decode()}")

    return output_files


def extract_osm_features(lat: float, lon: float, size_meters: float,
                        input_pbf: str, output_dir: str) -> Dict[str, str]:
    """
    Main function: crop and parse OSM data for a region.

    Args:
        lat, lon: Center coordinates
        size_meters: Size of square region
        input_pbf: Path to large OSM PBF file
        output_dir: Output directory

    Returns:
        Dict of output GeoJSON files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Crop to region
    cropped_pbf = output_path / 'cropped.osm.pbf'
    print(f"Cropping OSM region ({size_meters}m around {lat}, {lon})...")

    if not crop_osm_region(input_pbf, lat, lon, size_meters, str(cropped_pbf)):
        return {}

    # Parse to GeoJSON
    print("Parsing OSM features to GeoJSON...")
    geojson_files = parse_osm_to_geojson(str(cropped_pbf), output_path)

    # Save metadata
    metadata = {
        'center': {'lat': lat, 'lon': lon},
        'size_meters': size_meters,
        'layers': geojson_files
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Extracted {len(geojson_files)} layers to {output_dir}")
    return geojson_files


def main():
    parser = argparse.ArgumentParser(description='Extract OSM features for a GPS region')
    parser.add_argument('--lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--size', type=float, default=500, help='Region size in meters')
    parser.add_argument('--input', required=True, help='Input .osm.pbf file')
    parser.add_argument('--output', default='./output', help='Output directory')

    args = parser.parse_args()

    extract_osm_features(
        args.lat, args.lon, args.size,
        args.input, args.output
    )


if __name__ == '__main__':
    main()
