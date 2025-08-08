import numpy as np
from collections import defaultdict

def auto_identify_labels(annotations):
    """
    Automatically identifies which class IDs correspond to tables, headers, and data cells.
    
    Args:
        annotations (dict): {class_id: [(x_center, y_center, width, height), ...]}
    
    Returns:
        dict: {"table": class_id, "header": class_id, "data": class_id}
    """
    # --- 1. Identify Tables (Largest boxes, max 2 occurrences) ---
    class_areas = {
        class_id: sum(w * h for (_, _, w, h) in boxes)
        for class_id, boxes in annotations.items()
    }
    
    # Filter classes with ≤ 2 boxes (likely tables)
    candidate_tables = [
        class_id for class_id, boxes in annotations.items()
        if len(boxes) <= 2
    ]
    
    if not candidate_tables:
        raise ValueError("No table detected (expected ≤ 2 boxes for table class).")
    
    # Pick the class with the largest total area as the table
    table_class = max(candidate_tables, key=lambda x: class_areas[x])
    
    # --- 2. Identify Headers (Top-aligned, repeated horizontally) ---
    remaining_classes = [c for c in annotations.keys() if c != table_class]
    if not remaining_classes:
        raise ValueError("No header/data classes found.")
    
    # Class with boxes closest to the top (min y_center)
    header_class = min(
        remaining_classes,
        key=lambda c: min(box[1] for box in annotations[c])
    )
    
    # --- 3. Data Cells (Remaining classes) ---
    data_classes = [c for c in remaining_classes if c != header_class]
    if not data_classes:
        data_class = header_class  # Fallback if no separate data class
    else:
        data_class = data_classes[0]  # Use first remaining class
    
    return {
        "table": table_class,
        "header": header_class,
        "data": data_class
    }

def annotate_remaining_cells(raw_data: str):
    """
    Generates missing row annotations while preserving label logic.
    
    Args:
        raw_data (str): Raw YOLO annotations (class_id x y w h per line)
    
    Returns:
        str: New annotations in YOLO format with correct labels
    """
    # --- 1. Parse Annotations ---
    annotations = defaultdict(list)
    for line in raw_data.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        coords = tuple(map(float, parts[1:]))
        annotations[class_id].append(coords)
    
    # --- 2. Auto-Detect Labels ---
    try:
        label_map = auto_identify_labels(annotations)
    except ValueError as e:
        return f"Error: {str(e)}"
    
    table_class = label_map["table"]
    header_class = label_map["header"]
    data_class = label_map["data"]
    
    print(f"Detected Labels: Table={table_class}, Header={header_class}, Data={data_class}")
    
    # --- 3. Extract Components ---
    table_boxes = annotations.get(table_class, [])
    header_boxes = annotations.get(header_class, [])
    data_boxes = annotations.get(data_class, [])
    
    # --- 4. Estimate Table Bounds ---
    if not table_boxes:
        all_boxes = header_boxes + data_boxes
        x_centers = [b[0] for b in all_boxes]
        y_centers = [b[1] for b in all_boxes]
        table_box = (
            (min(x_centers) + max(x_centers)) / 2,
            (min(y_centers) + max(y_centers)) / 2,
            max(x_centers) - min(x_centers) + max(b[2] for b in all_boxes),
            max(y_centers) - min(y_centers) + max(b[3] for b in all_boxes)
        )
    else:
        table_box = max(table_boxes, key=lambda b: b[2] * b[3])
    
    # --- 5. Group Data Cells into Rows ---
    def group_into_rows(boxes, y_tolerance=0.01):
        if not boxes:
            return []
        boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))  # Sort by y, then x
        rows = []
        current_row = [boxes_sorted[0]]
        for box in boxes_sorted[1:]:
            if abs(box[1] - current_row[-1][1]) < y_tolerance:
                current_row.append(box)
            else:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
        return rows
    
    data_rows = group_into_rows(data_boxes)
    
    # --- 6. Generate New Rows ---
    if len(data_rows) < 2:
        return "Error: Need ≥2 data rows to infer pattern."
    
    # Calculate vertical pitch (avg y-difference between rows)
    row_pitch = np.mean([
        data_rows[i+1][0][1] - data_rows[i][0][1] 
        for i in range(len(data_rows)-1)
    ])
    
    # Generate new rows until table bottom
    new_annotations = []
    last_row = data_rows[-1]
    table_bottom = table_box[1] + table_box[3]/2
    
    while True:
        new_row = []
        for cell in last_row:
            new_cell = (cell[0], cell[1] + row_pitch, cell[2], cell[3])
            if new_cell[1] + new_cell[3]/2 >= table_bottom:
                break
            new_row.append(new_cell)
        else:
            new_annotations.extend([
                f"{data_class} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                for (x, y, w, h) in new_row
            ])
            last_row = new_row
            continue
        break
    
    # --- 7. Return Results ---
    original_lines = raw_data.strip().split('\n')
    output = "\n".join(original_lines + new_annotations)
    return output

# --- Usage Example ---
input_file = "C:\\Users\\s66\\Desktop\\check annto\\536024944-SEPT-20_page_2.txt"
with open(input_file, 'r') as f:
    raw_data = f.read()

result = annotate_remaining_cells(raw_data)
print("\nGenerated Annotations:")
print(result)