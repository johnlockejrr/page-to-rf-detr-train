#!/usr/bin/env python3
"""
PAGE-XML to COCO Segmentation Format Converter for RF-DETR

This script converts PAGE-XML files with textline annotations to COCO segmentation format
for training RF-DETR models. It handles multiple PAGE-XML namespace schemas and creates
the proper directory structure with train/valid/test splits.

Usage:
    python convert_pagexml_to_coco.py --input_dir /path/to/pagexml --output_dir /path/to/coco_dataset

The script will:
1. Parse PAGE-XML files with various namespace schemas
2. Extract textline polygon coordinates
3. Convert to COCO segmentation format
4. Split dataset optimally using scikit-learn
5. Create proper directory structure for RF-DETR training
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
import hashlib

# PAGE-XML namespace schemas
PAGE_NAMESPACES = {
    '2009-03-16': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2009-03-16',
    '2010-01-12': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-01-12', 
    '2010-03-19': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19',
    '2013-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
    '2014-08-26': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2014-08-26',
    '2016-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2016-07-15',
    '2017-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15',
    '2018-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15',
    '2019-07-15': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
}

def detect_page_namespace(xml_file: Path) -> str:
    """Detect the PAGE-XML namespace from the XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract namespace from the root tag
        if root.tag.startswith('{'):
            # Extract namespace from {namespace}tag format
            namespace = root.tag.split('}')[0][1:]
            return namespace
        
        # Check the xmlns attribute
        xmlns = root.get('{http://www.w3.org/2000/xmlns/}xmlns')
        if xmlns:
            for version, namespace in PAGE_NAMESPACES.items():
                if namespace in xmlns:
                    return namespace
                
        # Default to most recent namespace
        return PAGE_NAMESPACES['2019-07-15']
        
    except Exception as e:
        print(f"Warning: Could not detect namespace for {xml_file}: {e}")
        return PAGE_NAMESPACES['2019-07-15']

def parse_polygon_coords(polygon_str: str) -> List[Tuple[float, float]]:
    """Parse polygon coordinate string and return list of (x, y) tuples."""
    if not polygon_str:
        return []
    
    # Split by spaces and parse coordinate pairs
    coords = polygon_str.strip().split()
    points = []
    
    for coord in coords:
        try:
            # Handle both "x,y" and "x y" formats
            if ',' in coord:
                x, y = coord.split(',', 1)
                x = float(x)
                y = float(y)
                points.append((x, y))
            else:
                # This shouldn't happen with PAGE-XML, but just in case
                continue
        except ValueError:
            continue
    
    return points

def extract_textlines_from_pagexml(xml_file: Path, namespace: str) -> List[List[Tuple[float, float]]]:
    """Extract textline polygon coordinates from PAGE-XML file."""
    textlines = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Use the detected namespace to find TextLine elements
        textline_elements = root.findall(f'.//{{{namespace}}}TextLine')
        
        for textline in textline_elements:
            # Look for polygon coordinates in Coords element using the same namespace
            coords_elem = textline.find(f'{{{namespace}}}Coords')
            
            if coords_elem is not None:
                polygon = coords_elem.get('points')
                if polygon:
                    coords = parse_polygon_coords(polygon)
                    if len(coords) >= 3:  # At least 3 points for a valid polygon
                        textlines.append(coords)
    
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
    
    return textlines

def polygon_to_coco_segmentation(polygon: List[Tuple[float, float]]) -> List[float]:
    """Convert polygon coordinates to COCO segmentation format (flattened list)."""
    if not polygon:
        return []
    
    # Flatten the coordinates: [x1, y1, x2, y2, ...]
    segmentation = []
    for x, y in polygon:
        segmentation.extend([float(x), float(y)])
    
    return segmentation

def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """Calculate the area of a polygon using the shoelace formula."""
    if len(polygon) < 3:
        return 0.0
    
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0

def get_polygon_bbox(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Get bounding box for a polygon [x, y, width, height]."""
    if not polygon:
        return (0.0, 0.0, 0.0, 0.0)
    
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def create_coco_annotation(annotation_id: int, image_id: int, polygon: List[Tuple[float, float]], 
                          category_id: int) -> Dict[str, Any]:
    """Create a COCO annotation entry."""
    segmentation = polygon_to_coco_segmentation(polygon)
    bbox = get_polygon_bbox(polygon)
    area = calculate_polygon_area(polygon)
    
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [segmentation],
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    }

def create_coco_image(image_id: int, file_name: str, width: int, height: int) -> Dict[str, Any]:
    """Create a COCO image entry."""
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name
    }

def create_coco_category(category_id: int, name: str) -> Dict[str, Any]:
    """Create a COCO category entry."""
    return {
        "id": category_id,
        "name": name,
        "supercategory": "background"  # Use "background" as supercategory to match Roboflow format
    }

def resize_image_and_coords(image: np.ndarray, coords: List[List[Tuple[float, float]]], 
                          target_h: int, target_w: int) -> Tuple[np.ndarray, List[List[Tuple[float, float]]]]:
    """Resize image and scale coordinates accordingly."""
    orig_h, orig_w = image.shape[:2]
    
    # Calculate scaling factors
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    # Resize image
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Scale coordinates
    scaled_coords = []
    for polygon in coords:
        scaled_polygon = []
        for x, y in polygon:
            new_x = x * scale_x
            new_y = y * scale_y
            scaled_polygon.append((new_x, new_y))
        scaled_coords.append(scaled_polygon)
    
    return resized_image, scaled_coords

def process_pagexml_files(input_dir: Path, target_width: int = None, target_height: int = None) -> List[Dict[str, Any]]:
    """Process all PAGE-XML files and return list of processed data."""
    xml_files = list(input_dir.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return []
    
    print(f"Found {len(xml_files)} XML files to process")
    if target_width and target_height:
        print(f"Target image size: {target_width}x{target_height}")
    
    processed_data = []
    
    for xml_file in tqdm(xml_files, desc="Processing PAGE-XML files"):
        try:
            # Find corresponding image file
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                potential_image = xml_file.with_suffix(ext)
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            if not image_file:
                print(f"No corresponding image found for {xml_file}")
                continue
            
            # Detect namespace and extract textlines
            namespace = detect_page_namespace(xml_file)
            textlines = extract_textlines_from_pagexml(xml_file, namespace)
            
            if not textlines:
                print(f"No textlines found in {xml_file}")
                continue
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Could not load image {image_file}")
                continue
            
            orig_height, orig_width = image.shape[:2]
            
            # Resize image and scale coordinates if target size is specified
            if target_width and target_height:
                resized_image, scaled_textlines = resize_image_and_coords(
                    image, textlines, target_height, target_width
                )
                final_width, final_height = target_width, target_height
                final_textlines = scaled_textlines
            else:
                resized_image = image
                final_width, final_height = orig_width, orig_height
                final_textlines = textlines
            
            processed_data.append({
                'xml_file': xml_file,
                'image_file': image_file,
                'textlines': final_textlines,
                'width': final_width,
                'height': final_height,
                'base_name': xml_file.stem,
                'resized_image': resized_image
            })
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_data)} files")
    return processed_data

def split_dataset(processed_data: List[Dict[str, Any]], test_size: float = 0.2, val_size: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train, validation, and test sets."""
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        processed_data, 
        test_size=test_size, 
        random_state=42
    )
    
    # Second split: separate train and validation from remaining data
    val_ratio = val_size / (1 - test_size)  # Adjust val_size for remaining data
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=val_ratio, 
        random_state=42
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_data)} files")
    print(f"  Validation: {len(val_data)} files")
    print(f"  Test: {len(test_data)} files")
    
    return train_data, val_data, test_data

def create_coco_dataset(split_data: List[Dict[str, Any]], split_name: str, output_dir: Path, 
                       category_id: int = 1) -> None:
    """Create COCO dataset for a specific split."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO structure with RF-DETR compatible categories
    # RF-DETR expects a dummy background class with id=0
    coco_data = {
        "info": {
            "description": f"PAGE-XML to COCO conversion for RF-DETR training - {split_name} split",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "PAGE-XML to COCO Converter",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "background",
                "supercategory": "none"
            },
            create_coco_category(category_id, "text_line")
        ]
    }
    
    image_id = 1
    annotation_id = 1
    
    for data in tqdm(split_data, desc=f"Creating {split_name} split"):
        # Save resized image to split directory
        image_filename = f"{data['base_name']}.jpg"
        dest_image_path = split_dir / image_filename
        
        if 'resized_image' in data:
            # Save the resized image
            cv2.imwrite(str(dest_image_path), data['resized_image'])
        else:
            # Copy original image if no resizing was done
            shutil.copy2(data['image_file'], dest_image_path)
        
        # Add image to COCO data
        coco_image = create_coco_image(
            image_id, 
            image_filename, 
            data['width'], 
            data['height']
        )
        coco_data["images"].append(coco_image)
        
        # Add annotations for each textline
        for polygon in data['textlines']:
            annotation = create_coco_annotation(
                annotation_id, 
                image_id, 
                polygon, 
                category_id
            )
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    # Save COCO annotation file
    coco_file = split_dir / "_annotations.coco.json"
    with open(coco_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Created {split_name} split with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")

def convert_pagexml_to_coco(input_dir: str, output_dir: str, test_size: float = 0.2, val_size: float = 0.1,
                           target_width: int = None, target_height: int = None):
    """Convert PAGE-XML files to COCO segmentation format with train/val/test splits."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting PAGE-XML files from {input_dir} to COCO format")
    print(f"Output directory: {output_dir}")
    print(f"Test size: {test_size}, Validation size: {val_size}")
    print("Note: Adding dummy background class (id=0) for RF-DETR compatibility")
    
    # Validate target dimensions if provided
    if target_width and target_height:
        if target_width % 56 != 0 or target_height % 56 != 0:
            print(f"Warning: Target dimensions {target_width}x{target_height} are not divisible by 56.")
            print("RF-DETR requires dimensions divisible by 56. Consider using 896x640 or similar.")
        print(f"Target image size: {target_width}x{target_height}")
    else:
        print("No target size specified - using original image dimensions")
    
    # Process all PAGE-XML files
    processed_data = process_pagexml_files(input_path, target_width, target_height)
    
    if not processed_data:
        print("No valid data found to process")
        return
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(processed_data, test_size, val_size)
    
    # Create COCO datasets for each split
    create_coco_dataset(train_data, "train", output_path)
    create_coco_dataset(val_data, "valid", output_path)
    create_coco_dataset(test_data, "test", output_path)
    
    print(f"\nConversion completed!")
    print(f"Dataset structure created at: {output_path}")
    print(f"  - train/ (with _annotations.coco.json)")
    print(f"  - valid/ (with _annotations.coco.json)")
    print(f"  - test/ (with _annotations.coco.json)")
    if target_width and target_height:
        print(f"All images resized to: {target_width}x{target_height}")

def main():
    parser = argparse.ArgumentParser(description="Convert PAGE-XML files to COCO segmentation format for RF-DETR")
    parser.add_argument("--input_dir", required=True, help="Input directory containing PAGE-XML files and images")
    parser.add_argument("--output_dir", required=True, help="Output directory for COCO format dataset")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset for test split (default: 0.2)")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of dataset for validation split (default: 0.1)")
    parser.add_argument("--target_width", type=int, default=896, help="Target image width for resizing (default: 896, must be divisible by 56)")
    parser.add_argument("--target_height", type=int, default=640, help="Target image height for resizing (default: 640, must be divisible by 56)")
    parser.add_argument("--no_resize", action="store_true", help="Disable image resizing and use original dimensions")
    
    args = parser.parse_args()
    
    # Validate split sizes
    if args.test_size + args.val_size >= 1.0:
        print("Error: test_size + val_size must be less than 1.0")
        sys.exit(1)
    
    # Set target dimensions
    target_width = None if args.no_resize else args.target_width
    target_height = None if args.no_resize else args.target_height
    
    convert_pagexml_to_coco(args.input_dir, args.output_dir, args.test_size, args.val_size, target_width, target_height)

if __name__ == "__main__":
    main()
