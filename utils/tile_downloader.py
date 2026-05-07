#!/usr/bin/env python3
"""
Download satellite/OSM map tiles for a given GPS region.

Usage:
    python tile_downloader.py --lat 30.387 --lon -97.728 --size 500 \
        --output map_tile.jpg --zoom 18

Requires: requests, pillow, numpy
"""

import argparse
import math
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert GPS coordinates to tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_latlon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to GPS (top-left corner)."""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def meters_to_pixels(lat: float, zoom: int, meters: float) -> int:
    """Convert meters to pixels at given latitude and zoom."""
    earth_circumference = 40075017.0
    meters_per_pixel = earth_circumference / (256 * 2**zoom)
    meters_per_pixel *= math.cos(math.radians(lat))
    return int(meters / meters_per_pixel)


def download_tile(x: int, y: int, zoom: int,
                 tile_server: str = 'osm') -> Optional[Image.Image]:
    """
    Download a single tile from server.

    Args:
        x, y: Tile coordinates
        zoom: Zoom level
        tile_server: 'osm' or 'satellite'

    Returns:
        PIL Image or None
    """
    servers = {
        'osm': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        'satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    }

    url = servers.get(tile_server, servers['osm'])
    url = url.format(z=zoom, x=x, y=y)

    headers = {'User-Agent': 'UAV-Map-VQA-Research/1.0'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Failed to download tile ({x}, {y}): {e}")
        return None


def download_map_region(lat: float, lon: float, size_meters: float,
                       zoom: int = 18, tile_server: str = 'osm',
                       output_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """
    Download and stitch tiles for a region.

    Args:
        lat, lon: Center coordinates
        size_meters: Size of region in meters
        zoom: Tile zoom level (default: 18)
        tile_server: 'osm' or 'satellite'
        output_size: Resize to (width, height), or None to keep original

    Returns:
        PIL Image or None
    """
    # Calculate tile coverage
    center_x, center_y = latlon_to_tile(lat, lon, zoom)
    pixels_needed = meters_to_pixels(lat, zoom, size_meters)
    tiles_needed = math.ceil(pixels_needed / 256)

    # Download tiles
    tile_images = {}
    start_x = center_x - tiles_needed // 2
    start_y = center_y - tiles_needed // 2

    print(f"Downloading {tiles_needed}×{tiles_needed} tiles (zoom {zoom})...")

    for dy in range(tiles_needed):
        for dx in range(tiles_needed):
            tx = start_x + dx
            ty = start_y + dy

            tile_img = download_tile(tx, ty, zoom, tile_server)
            if tile_img:
                tile_images[(dx, dy)] = tile_img

            # Rate limiting
            time.sleep(0.1)

    if not tile_images:
        print("Error: No tiles downloaded")
        return None

    # Stitch tiles
    canvas_size = tiles_needed * 256
    canvas = Image.new('RGB', (canvas_size, canvas_size))

    for (dx, dy), tile_img in tile_images.items():
        canvas.paste(tile_img, (dx * 256, dy * 256))

    # Crop to exact region
    center_px = canvas_size // 2
    half_size = pixels_needed // 2
    crop_box = (
        center_px - half_size,
        center_px - half_size,
        center_px + half_size,
        center_px + half_size
    )
    cropped = canvas.crop(crop_box)

    # Resize if requested
    if output_size:
        cropped = cropped.resize(output_size, Image.LANCZOS)

    return cropped


def main():
    parser = argparse.ArgumentParser(description='Download map tiles for a GPS region')
    parser.add_argument('--lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--size', type=float, default=500, help='Region size in meters')
    parser.add_argument('--zoom', type=int, default=18, help='Tile zoom level (default: 18)')
    parser.add_argument('--server', choices=['osm', 'satellite'], default='osm',
                       help='Tile server')
    parser.add_argument('--output', default='map_tile.jpg', help='Output image path')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help='Resize output to width×height pixels')

    args = parser.parse_args()

    output_size = tuple(args.resize) if args.resize else None

    img = download_map_region(
        args.lat, args.lon, args.size,
        zoom=args.zoom,
        tile_server=args.server,
        output_size=output_size
    )

    if img:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        img.save(args.output, quality=95)
        print(f"Saved map tile to {args.output} ({img.size[0]}×{img.size[1]}px)")
    else:
        print("Failed to download map region")


if __name__ == '__main__':
    main()
