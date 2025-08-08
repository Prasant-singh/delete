# import numpy as np

# def annotate_remaining_cells(raw_data: str):
#     """
#     Automates table cell annotation based on initial labelImg data.

#     Args:
#         raw_data (str): A string containing the raw annotation data from labelImg,
#                         where each line is 'class_id x_center y_center width height'.

#     Returns:
#         list[list[tuple]]: A list of newly generated rows, where each row is a
#                            list of cell annotation tuples (x, y, w, h).
#     """

#     # --- 1. Parse and Organize Initial Data ---
#     table_box = None
#     header_cells = []
#     data_cells = []

#     lines = raw_data.strip().split('\n')
#     for line in lines:
#         parts = line.strip().split()
#         if not parts:
#             continue
#         class_id = int(parts[0])
#         coords = tuple(float(p) for p in parts[1:])

#         if class_id == 1: # Table boundary
#             table_box = coords
#         elif class_id == 2: # Header cells
#             header_cells.append(coords)
#         elif class_id == 0: # Data cells
#             data_cells.append(coords)

#     if not table_box or not data_cells or len(data_cells) < 2:
#         raise ValueError("Insufficient data. Please provide annotations for the table boundary and at least two full data rows.")

#     # Group data cells into rows based on their y_center coordinate
#     # We use a tolerance to group cells with similar y-coordinates into the same row
#     y_centers = np.array([cell[1] for cell in data_cells])
#     unique_y = []
#     # Find unique y-centers within a small tolerance
#     for y in sorted(np.unique(y_centers)):
#         if not unique_y or abs(y - unique_y[-1]) > 0.01: # 0.01 is a tolerance factor
#             unique_y.append(y)
            
#     initial_rows = []
#     for uy in unique_y:
#         row = sorted([cell for cell in data_cells if abs(cell[1] - uy) < 0.01], key=lambda c: c[0])
#         if row:
#             initial_rows.append(row)

#     if len(initial_rows) < 2:
#         raise ValueError("Could not detect at least two distinct rows from the provided data cells.")

#     # --- 2. Calculate Row Pitch and Define Template ---
#     row1_avg_y = np.mean([cell[1] for cell in initial_rows[0]])
#     row2_avg_y = np.mean([cell[1] for cell in initial_rows[1]])
    
#     # The vertical distance between the start of one row and the next
#     row_pitch = row2_avg_y - row1_avg_y

#     # The last annotated row will be our starting point for generation
#     last_known_row = initial_rows[-1]
    
#     # --- 3. Generate New Rows until Table Bottom is Reached ---
#     generated_rows = []
#     current_row_template = last_known_row
    
#     # Calculate the bottom boundary of the table
#     table_y_center, table_height = table_box[1], table_box[3]
#     table_bottom_edge = table_y_center + (table_height / 2)

#     while True:
#         next_row = []
#         for cell in current_row_template:
#             x, y, w, h = cell
#             # Create new cell by shifting the y-coordinate down by the pitch
#             new_cell = (x, y + row_pitch, w, h)
#             next_row.append(new_cell)
        
#         # Check if the newly generated row is outside the table boundary
#         # We check the bottom of the new cells (y + h/2)
#         new_row_bottom = next_row[0][1] + (next_row[0][3] / 2)
#         if new_row_bottom >= table_bottom_edge:
#             break # Stop if we've gone past the table's end

#         generated_rows.append(next_row)
#         current_row_template = next_row # The new row becomes the template for the next one

#     return generated_rows

# # --- 4. How to Use the Function ---

# # Paste the sample data you provided
# # raw_labelimg_data = """
# # 1 0.500000 0.480901 0.904915 0.627708
# # 0 0.095890 0.204675 0.093473 0.023945
# # 0 0.187349 0.204105 0.092667 0.020525
# # 0 0.385979 0.202965 0.307816 0.020525
# # 0 0.574940 0.204105 0.073328 0.018244
# # 0 0.663578 0.202965 0.103948 0.025086
# # 0 0.777196 0.204105 0.108783 0.022805
# # 0 0.888799 0.205815 0.112812 0.021665
# # 0 0.093473 0.226340 0.091861 0.021665
# # 0 0.187349 0.226910 0.092667 0.022805
# # 0 0.387994 0.225770 0.303787 0.022805
# # 0 0.575342 0.225200 0.066076 0.023945
# # 0 0.665189 0.227480 0.103948 0.021665
# # 0 0.775181 0.226340 0.112812 0.021665
# # 0 0.890411 0.227480 0.116035 0.023945
# # 2 0.097502 0.177309 0.096696 0.022805
# # 2 0.189766 0.180730 0.091056 0.019384
# # 2 0.390008 0.180160 0.304593 0.025086
# # 2 0.576148 0.178734 0.067687 0.025656
# # 2 0.667607 0.177309 0.107172 0.020525
# # 2 0.777599 0.179019 0.112812 0.022805
# # 2 0.892828 0.180730 0.111201 0.023945
# # """
# with open('C:\\Users\\s66\\Desktop\\check annto\\BOB_statement_with_password_page_1.txt', 'r') as file:
#     raw_labelimg_data = file.read()

# # Get the list of generated cell coordinates
# newly_annotated_rows = annotate_remaining_cells(raw_labelimg_data)

# # Print the results in labelImg format (class_id = 0)
# print("--- Generated Annotations for Remaining Rows ---")
# for i, row in enumerate(newly_annotated_rows):
#     print(f"\n# Generated Row {i+1}")
#     for cell in row:
#         x, y, w, h = cell
#         print(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")



import numpy as np
from collections import defaultdict

def annotate_remaining_cells(raw_data: str):
    """
    Automates table cell annotation regardless of initial labeling order.
    
    Args:
        raw_data (str): Raw annotation data where each line is 'class_id x_center y_center width height'
    
    Returns:
        list[list[tuple]]: Newly generated rows of cell annotations (x, y, w, h)
    """
    # --- 1. Parse and Classify Annotations ---
    annotations = defaultdict(list)
    lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
    
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        coords = tuple(float(p) for p in parts[1:])
        annotations[class_id].append(coords)
    
    # --- 2. Identify Components (Flexible Classification) ---
    # Strategy: Largest bbox is table, top-most row is likely header, others are data
    all_boxes = [box for boxes in annotations.values() for box in boxes]
    
    if not all_boxes:
        raise ValueError("No valid annotations found")
    
    # Find table (largest area box)
    table_box = max(all_boxes, key=lambda b: b[2]*b[3]) if len(all_boxes) > 1 else None
    
    # Separate header and data cells
    header_cells = []
    data_cells = []
    
    for class_id, boxes in annotations.items():
        if class_id == -1:  # Skip if using placeholder class
            continue
        for box in boxes:
            if box == table_box:
                continue
            if not header_cells or box[1] < header_cells[0][1]:
                header_cells.append(box)
            else:
                data_cells.append(box)
    
    # --- 3. Handle Edge Cases ---
    if not table_box:
        # Estimate table bounds from existing cells
        all_cells = header_cells + data_cells
        x_centers = [c[0] for c in all_cells]
        y_centers = [c[1] for c in all_cells]
        widths = [c[2] for c in all_cells]
        heights = [c[3] for c in all_cells]
        
        table_x = (min(x_centers) + max(x_centers)) / 2
        table_y = (min(y_centers) + max(y_centers)) / 2
        table_w = max(x_centers) + max(widths)/2 - (min(x_centers) - min(widths)/2)
        table_h = max(y_centers) + max(heights)/2 - (min(y_centers) - min(heights)/2)
        table_box = (table_x, table_y, table_w, table_h)
    
    # --- 4. Group Cells into Rows ---
    def group_into_rows(cells, y_tolerance=0.01):
        if not cells:
            return []
        
        # Sort by y-center then x-center
        sorted_cells = sorted(cells, key=lambda c: (c[1], c[0]))
        
        rows = []
        current_row = [sorted_cells[0]]
        
        for cell in sorted_cells[1:]:
            if abs(cell[1] - current_row[-1][1]) < y_tolerance:
                current_row.append(cell)
            else:
                rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cell]
        
        if current_row:
            rows.append(sorted(current_row, key=lambda c: c[0]))
        
        return rows
    
    all_rows = group_into_rows(data_cells)
    
    if len(all_rows) < 2:
        raise ValueError("Need at least two complete rows to detect pattern")
    
    # --- 5. Calculate Row Pattern ---
    row_pitch = all_rows[1][0][1] - all_rows[0][0][1]  # y-difference between rows
    
    # --- 6. Generate New Rows ---
    generated_rows = []
    last_row = all_rows[-1]
    table_bottom = table_box[1] + table_box[3]/2
    
    while True:
        new_row = []
        for cell in last_row:
            new_cell = (cell[0], cell[1] + row_pitch, cell[2], cell[3])
            new_row.append(new_cell)
        
        # Check if new row exceeds table bounds
        if new_row[0][1] + new_row[0][3]/2 >= table_bottom:
            break
            
        generated_rows.append(new_row)
        last_row = new_row
    
    return generated_rows

# --- Usage Example ---
with open('C:\\Users\\s66\\Desktop\\check annto\\BOB_statement_with_password_page_3.txt', 'r') as f:
    raw_data = f.read()

try:
    new_rows = annotate_remaining_cells(raw_data)
    print("Generated annotations:")
    for i, row in enumerate(new_rows):
        print(f"\n# Row {i+1}")
        for cell in row:
            print(f"0 {cell[0]:.6f} {cell[1]:.6f} {cell[2]:.6f} {cell[3]:.6f}")
except Exception as e:
    print(f"Error: {str(e)}")