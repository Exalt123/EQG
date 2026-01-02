import streamlit as st
import math
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Page config
st.set_page_config(
    page_title="MDF Cutting Optimizer - Advanced",
    page_icon="🪵",
    layout="wide"
)

# --- DATA CLASSES ---
@dataclass
class Piece:
    """Represents a piece to be cut."""
    job_number: str
    width: float
    height: float
    quantity: int
    thickness: Optional[float] = None
    
@dataclass
class Sheet:
    """Represents an available sheet."""
    part_number: Optional[str] = None  # Part number for actual sheets
    description: Optional[str] = None  # Sheet description
    width: float = 0
    height: float = 0
    thickness: Optional[float] = None
    cost: float = 0
    is_actual: bool = False  # True for actual sheets, False for proposed
    is_drop: bool = False  # True if this is a drop piece (marked with X in "Drop on Drop Output")
    status: str = "Available"  # Available, Used, Repurchased
    used_pieces: List[Tuple[float, float]] = None  # Already used areas on the sheet
    
    def __post_init__(self):
        if self.used_pieces is None:
            self.used_pieces = []

# --- HELPER FUNCTIONS ---
def safe_float(value, default=0.0):
    """Safely convert value to float, return default if conversion fails."""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace('$', '').replace(',', '')
        if value == "":
            return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# --- GOOGLE SHEETS CONNECTION ---
@st.cache_data(ttl=300)
def get_google_sheets_client(credentials_json):
    """Initialize Google Sheets client from service account credentials."""
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(credentials_json, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

def fetch_sheet_data(client, spreadsheet_id, sheet_name_or_id):
    """Fetch data from a specific sheet in the spreadsheet."""
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # Handle sheet name or ID
        if isinstance(sheet_name_or_id, int) or (isinstance(sheet_name_or_id, str) and sheet_name_or_id.replace('gid=', '').isdigit()):
            sheet_id_str = str(sheet_name_or_id).replace('gid=', '')
            try:
                sheet = spreadsheet.get_worksheet_by_id(int(sheet_id_str))
            except:
                all_sheets = spreadsheet.worksheets()
                for s in all_sheets:
                    if str(s.id) == sheet_id_str:
                        sheet = s
                        break
                else:
                    raise ValueError(f"Sheet ID {sheet_name_or_id} not found")
        else:
            sheet = spreadsheet.worksheet(sheet_name_or_id)
        
        records = sheet.get_all_records()
        result = []
        for record in records:
            clean_row = {k.strip().lower(): v for k, v in record.items()}
            result.append(clean_row)
        return result
    except Exception as e:
        st.error(f"Error fetching sheet data: {e}")
        return []

# --- OPTIMIZATION ENGINE ---
def calculate_parts_per_sheet(sheet_w, sheet_h, piece_w, piece_h, saw_kerf=0.125):
    """Calculate how many pieces fit on a sheet in both orientations."""
    eff_w = piece_w + saw_kerf
    eff_h = piece_h + saw_kerf
    
    if eff_w > sheet_w and eff_w > sheet_h:
        return 0, "None"
    if eff_h > sheet_w and eff_h > sheet_h:
        return 0, "None"
    
    # Normal orientation
    yield_norm = math.floor(sheet_w / eff_w) * math.floor(sheet_h / eff_h)
    
    # Rotated orientation
    yield_rot = math.floor(sheet_w / eff_h) * math.floor(sheet_h / eff_w)
    
    if yield_norm >= yield_rot:
        return yield_norm, "Normal"
    else:
        return yield_rot, "Rotated"

def can_fit_on_sheet(sheet: Sheet, piece: Piece, saw_kerf=0.125) -> Tuple[bool, int, str]:
    """Check if a piece can fit on a sheet, considering already used areas."""
    # Check thickness match
    if piece.thickness is not None and sheet.thickness is not None:
        if abs(piece.thickness - sheet.thickness) > 0.01:
            return False, 0, "Thickness mismatch"
    
    parts, orientation = calculate_parts_per_sheet(
        sheet.width, sheet.height, piece.width, piece.height, saw_kerf
    )
    
    if parts == 0:
        return False, 0, "Doesn't fit"
    
    return True, parts, orientation

def optimize_proposed_only(pieces: List[Piece], proposed_sheets: List[Sheet], saw_kerf: float = 0.125) -> Dict:
    """Calculate cost if using ONLY proposed/full sheets (no drop pieces).
    This represents the cost if we had to buy everything new without using any actual inventory.
    """
    SAW_KERF = saw_kerf
    
    # Filter to only non-drop proposed sheets (full sheets only)
    full_sheets = [s for s in proposed_sheets if not s.is_drop]
    
    if not full_sheets:
        return {"total_cost": 0, "assignments": []}
    
    # Sort by cost per square inch
    full_sheets.sort(key=lambda x: x.cost / (x.width * x.height) if (x.width * x.height) > 0 else float('inf'))
    
    results = {
        "assignments": [],
        "total_cost": 0,
        "sheets_used": []
    }
    
    remaining = {i: piece.quantity for i, piece in enumerate(pieces)}
    
    # Group pieces by thickness
    pieces_by_thickness = {}
    for piece_idx, piece in enumerate(pieces):
        if remaining[piece_idx] > 0:
            thickness_key = piece.thickness if piece.thickness else "any"
            if thickness_key not in pieces_by_thickness:
                pieces_by_thickness[thickness_key] = []
            pieces_by_thickness[thickness_key].append((piece_idx, piece, remaining[piece_idx]))
    
    # Process each thickness group
    for thickness_key, piece_list in pieces_by_thickness.items():
        matching_sheets = [s for s in full_sheets 
                          if s.thickness is None or piece_list[0][1].thickness is None 
                          or abs(s.thickness - piece_list[0][1].thickness) < 0.01]
        
        if not matching_sheets:
            continue
        
        for piece_idx, piece, qty_needed in piece_list:
            while remaining[piece_idx] > 0:
                best_sheet = None
                best_efficiency = 0
                best_parts = 0
                best_orientation = "Normal"
                
                for sheet in matching_sheets:
                    can_fit, parts_per_sheet, orientation = can_fit_on_sheet(sheet, piece, SAW_KERF)
                    if can_fit and parts_per_sheet > 0:
                        efficiency = parts_per_sheet / sheet.cost if sheet.cost > 0 else parts_per_sheet
                        if efficiency > best_efficiency:
                            best_sheet = sheet
                            best_efficiency = efficiency
                            best_parts = parts_per_sheet
                            best_orientation = orientation
                
                if best_sheet is None:
                    break
                
                sheets_needed = math.ceil(remaining[piece_idx] / best_parts)
                pieces_assigned = min(remaining[piece_idx], sheets_needed * best_parts)
                remaining[piece_idx] -= pieces_assigned
                
                results["assignments"].append({
                    "sheet": best_sheet,
                    "pieces": [{
                        "piece": piece,
                        "quantity": pieces_assigned,
                        "orientation": best_orientation,
                        "parts_per_sheet": best_parts
                    }],
                    "cost": best_sheet.cost * sheets_needed,
                    "sheets_count": sheets_needed
                })
                results["total_cost"] += best_sheet.cost * sheets_needed
                                results["sheets_used"].append(sheet.part_number or f"Proposed-{len(results['sheets_used'])}")
    
    return results

def optimize_cutting(pieces: List[Piece], available_sheets: List[Sheet], use_actual: bool, saw_kerf: float = 0.125) -> Dict:
    """Optimize cutting assignment across all pieces and sheets.
    
    Strategy:
    1. Group by thickness (each thickness = one setup/program)
    2. Prioritize actual sheets (especially with drop pieces) when use_actual=True
    3. Try to fit multiple pieces/jobs on one sheet
    4. Minimize total cost
    5. Minimize waste
    """
    SAW_KERF = saw_kerf
    
    # Separate actual and proposed sheets
    actual_sheets = [s for s in available_sheets if s.is_actual and s.status == "Available"]
    proposed_sheets = [s for s in available_sheets if not s.is_actual]
    
    # Sort actual sheets: prioritize drop pieces first (they're free/cheap), then by cost
    actual_sheets.sort(key=lambda x: (
        0 if x.is_drop else 1,  # Drop pieces first
        x.cost,  # Then by cost
        -x.width * x.height  # Smaller sheets first
    ))
    
    # Sort proposed sheets by cost per square inch
    proposed_sheets.sort(key=lambda x: x.cost / (x.width * x.height) if (x.width * x.height) > 0 else float('inf'))
    
    results = {
        "assignments": [],  # List of {sheet, pieces_assigned, cost, waste}
        "total_cost": 0,
        "total_waste": 0,
        "sheets_used": [],
        "unfulfilled_pieces": []
    }
    
    # Track remaining quantities needed
    remaining = {i: piece.quantity for i, piece in enumerate(pieces)}
    
    # First pass: Try to fill actual sheets if use_actual=True
    if use_actual and actual_sheets:
        for sheet in actual_sheets:
            sheet_assignments = []
            sheet_used_area = sum(w * h for w, h in sheet.used_pieces)
            available_area = (sheet.width * sheet.height) - sheet_used_area
            
            if available_area <= 0:
                continue
            
            # Try to fit pieces on this sheet
            for piece_idx, piece in enumerate(pieces):
                if remaining[piece_idx] <= 0:
                    continue
                
                # Check if piece fits
                can_fit, parts_per_sheet, orientation = can_fit_on_sheet(sheet, piece, SAW_KERF)
                
                if can_fit and parts_per_sheet > 0:
                    # Calculate how many we can cut from remaining quantity
                    pieces_to_cut = min(remaining[piece_idx], parts_per_sheet)
                    
                    # Check if we have enough available area
                    piece_area = piece.width * piece.height
                    needed_area = pieces_to_cut * piece_area
                    
                    if sheet_used_area + needed_area <= (sheet.width * sheet.height):
                        # Assign to this sheet
                        sheet_assignments.append({
                            "piece": piece,
                            "quantity": pieces_to_cut,
                            "orientation": orientation,
                            "parts_per_sheet": parts_per_sheet
                        })
                        remaining[piece_idx] -= pieces_to_cut
                        sheet_used_area += needed_area
            
            # If we assigned anything to this sheet, add it to results
            if sheet_assignments:
                # Calculate waste for this sheet
                total_piece_area = sum(a["piece"].width * a["piece"].height * a["quantity"] 
                                      for a in sheet_assignments)
                sheet_area = sheet.width * sheet.height
                waste_pct = ((sheet_area - sheet_used_area) / sheet_area) * 100 if sheet_area > 0 else 0
                
                results["assignments"].append({
                    "sheet": sheet,
                    "pieces": sheet_assignments,
                    "cost": sheet.cost if sheet.cost > 0 else 0,
                    "waste_pct": waste_pct,
                    "utilization_pct": (1 - waste_pct / 100) * 100
                })
                                results["sheets_used"].append(sheet.part_number or f"Sheet-{len(results['sheets_used'])}")
                results["total_cost"] += sheet.cost if sheet.cost > 0 else 0
    
    # Second pass: Use proposed sheets for remaining pieces
    # Group pieces by thickness to optimize sheet selection
    pieces_by_thickness = {}
    for piece_idx, piece in enumerate(pieces):
        if remaining[piece_idx] > 0:
            thickness_key = piece.thickness if piece.thickness else "any"
            if thickness_key not in pieces_by_thickness:
                pieces_by_thickness[thickness_key] = []
            pieces_by_thickness[thickness_key].append((piece_idx, piece, remaining[piece_idx]))
    
    # Process each thickness group
    for thickness_key, piece_list in pieces_by_thickness.items():
        # Find matching proposed sheets
        matching_sheets = [s for s in proposed_sheets 
                          if s.thickness is None or piece_list[0][1].thickness is None 
                          or abs(s.thickness - piece_list[0][1].thickness) < 0.01]
        
        if not matching_sheets:
            # No matching sheets - mark as unfulfilled
            for piece_idx, piece, qty in piece_list:
                results["unfulfilled_pieces"].append({
                    "piece": piece,
                    "quantity_needed": qty,
                    "reason": "No matching sheets available"
                })
            continue
        
        # Try to optimize sheet usage for this thickness
        for piece_idx, piece, qty_needed in piece_list:
            while remaining[piece_idx] > 0:
                best_sheet = None
                best_efficiency = 0
                best_parts = 0
                best_orientation = "Normal"
                
                # Find best sheet for remaining quantity
                for sheet in matching_sheets:
                    can_fit, parts_per_sheet, orientation = can_fit_on_sheet(sheet, piece, SAW_KERF)
                    if can_fit and parts_per_sheet > 0:
                        # Calculate efficiency (parts per dollar)
                        efficiency = parts_per_sheet / sheet.cost if sheet.cost > 0 else parts_per_sheet
                        if efficiency > best_efficiency:
                            best_sheet = sheet
                            best_efficiency = efficiency
                            best_parts = parts_per_sheet
                            best_orientation = orientation
                
                if best_sheet is None:
                    results["unfulfilled_pieces"].append({
                        "piece": piece,
                        "quantity_needed": remaining[piece_idx],
                        "reason": "No sheets can fit this piece"
                    })
                    break
                
                # Calculate sheets needed
                pieces_to_fill = min(remaining[piece_idx], best_parts)
                sheets_needed = math.ceil(pieces_to_fill / best_parts)
                
                # Create assignment
                pieces_assigned = min(remaining[piece_idx], sheets_needed * best_parts)
                remaining[piece_idx] -= pieces_assigned
                
                # Calculate waste
                total_piece_area = piece.width * piece.height * pieces_assigned
                sheet_area = best_sheet.width * best_sheet.height
                sheets_used_count = math.ceil(pieces_assigned / best_parts)
                total_sheet_area = sheet_area * sheets_used_count
                waste_pct = ((total_sheet_area - total_piece_area) / total_sheet_area) * 100 if total_sheet_area > 0 else 0
                
                results["assignments"].append({
                    "sheet": best_sheet,
                    "pieces": [{
                        "piece": piece,
                        "quantity": pieces_assigned,
                        "orientation": best_orientation,
                        "parts_per_sheet": best_parts
                    }],
                    "cost": best_sheet.cost * sheets_used_count,
                    "waste_pct": waste_pct,
                    "utilization_pct": (1 - waste_pct / 100) * 100,
                    "sheets_count": sheets_used_count
                })
                results["total_cost"] += best_sheet.cost * sheets_used_count
    
    # Calculate total waste
    total_piece_area = sum(
        sum(a["piece"].width * a["piece"].height * a["quantity"] for a in assign["pieces"])
        for assign in results["assignments"]
    )
    total_sheet_area = sum(
        assign["sheet"].width * assign["sheet"].height * assign.get("sheets_count", 1)
        for assign in results["assignments"]
    )
    results["total_waste"] = ((total_sheet_area - total_piece_area) / total_sheet_area * 100) if total_sheet_area > 0 else 0
    
    return results

def calculate_unoptimized_cost(pieces: List[Piece], available_sheets: List[Sheet], use_actual: bool, saw_kerf: float = 0.125) -> Dict:
    """Calculate cost if cutting each piece type separately (no optimization, no ganging).
    This represents the baseline/inefficient way of cutting.
    """
    SAW_KERF = saw_kerf
    
    # Separate actual and proposed sheets
    actual_sheets = [s for s in available_sheets if s.is_actual and s.status == "Available"]
    proposed_sheets = [s for s in available_sheets if not s.is_actual and not s.is_drop]
    
    # Sort by cost
    if use_actual and actual_sheets:
        actual_sheets.sort(key=lambda x: (x.cost, -x.width * x.height))
    proposed_sheets.sort(key=lambda x: x.cost / (x.width * x.height) if (x.width * x.height) > 0 else float('inf'))
    
    total_cost = 0
    assignments = []
    
    # Process each piece type separately (no ganging)
    for piece in pieces:
        best_cost = float('inf')
        best_assignment = None
        
        # Try actual sheets first if use_actual
        sheets_to_try = actual_sheets + proposed_sheets if use_actual else proposed_sheets
        
        for sheet in sheets_to_try:
            # Check thickness match
            if piece.thickness is not None and sheet.thickness is not None:
                if abs(piece.thickness - sheet.thickness) > 0.01:
                    continue
            
            can_fit, parts_per_sheet, orientation = can_fit_on_sheet(sheet, piece, SAW_KERF)
            if can_fit and parts_per_sheet > 0:
                sheets_needed = math.ceil(piece.quantity / parts_per_sheet)
                cost = sheet.cost * sheets_needed
                
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = {
                        "piece": piece,
                        "sheet": sheet,
                        "sheets_needed": sheets_needed,
                        "parts_per_sheet": parts_per_sheet,
                        "orientation": orientation,
                        "cost": cost
                    }
        
        if best_assignment:
            total_cost += best_cost
            assignments.append(best_assignment)
    
    return {
        "total_cost": total_cost,
        "assignments": assignments,
        "sheets_used": sum(a["sheets_needed"] for a in assignments)
    }

def create_cutting_programs(optimization_result: Dict) -> List[Dict]:
    """Group optimized assignments by thickness to create cutting programs.
    Each program represents one setup (same thickness) with multiple nested cuts.
    """
    programs = {}  # Key: thickness, Value: list of assignments
    
    for assign in optimization_result['assignments']:
        thickness = assign['sheet'].thickness
        thickness_key = f"{thickness}\"" if thickness else "Any"
        
        if thickness_key not in programs:
            programs[thickness_key] = {
                "thickness": thickness,
                "assignments": [],
                "total_sheets": 0,
                "total_cost": 0
            }
        
        programs[thickness_key]["assignments"].append(assign)
        programs[thickness_key]["total_sheets"] += assign.get("sheets_count", 1)
        programs[thickness_key]["total_cost"] += assign["cost"]
    
    # Convert to list and sort by thickness
    program_list = []
    for thickness_key, program_data in sorted(programs.items()):
        # Group sheets by size and pieces to create unique programs
        sheet_groups = {}
        for assign in program_data["assignments"]:
            sheet = assign['sheet']
            sheet_key = f"{sheet.width}x{sheet.height}"
            
            if sheet_key not in sheet_groups:
                sheet_groups[sheet_key] = {
                    "sheet": sheet,
                    "pieces": [],
                    "sheets_count": 0,
                    "cost": 0
                }
            
            # Add all pieces from this assignment
            for piece_assign in assign['pieces']:
                sheet_groups[sheet_key]["pieces"].append(piece_assign)
            sheet_groups[sheet_key]["sheets_count"] += assign.get("sheets_count", 1)
            sheet_groups[sheet_key]["cost"] += assign["cost"]
        
        # Create programs for each sheet size
        for sheet_key, group_data in sheet_groups.items():
            program_list.append({
                "thickness": program_data["thickness"],
                "thickness_display": thickness_key,
                "sheet": group_data["sheet"],
                "pieces": group_data["pieces"],
                "sheets_count": group_data["sheets_count"],
                "cost": group_data["cost"],
                "program_name": f"Program-{len(program_list) + 1}"
            })
    
    return program_list

# --- MAIN APP ---
st.title("🪵 MDF Cutting Optimizer - Advanced")
st.markdown("Optimize cutting across multiple jobs with actual and proposed sheets, including drop piece optimization.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Google Sheets setup
    with st.expander("❓ How to Get JSON Credentials", expanded=False):
        st.markdown("""
        **Quick Steps:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create/select a project
        3. Enable **Google Sheets API** and **Google Drive API**
        4. Create a Service Account and download JSON key
        5. Share your Google Sheet with the service account email
        """)
    
    credentials_method = st.radio("Credentials Method", ["Upload JSON", "Paste JSON"])
    credentials_json = None
    
    if credentials_method == "Upload JSON":
        cred_file = st.file_uploader("Service Account JSON", type=["json"], key="cred_upload")
        if cred_file:
            import json
            credentials_json = json.load(cred_file)
    else:
        cred_text = st.text_area("Service Account JSON", height=200, key="cred_paste")
        if cred_text:
            import json
            try:
                credentials_json = json.loads(cred_text)
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    if credentials_json:
        st.success("✓ Credentials loaded")
        spreadsheet_id = st.text_input("Spreadsheet ID", value="13wQEWzY7oxtHQX6wAeXn2G6f9Tp4fVBiYBA7Tw2u5aA")
        
        st.subheader("Sheet Configuration")
        actual_sheets_tab = st.text_input("Actual Sheets Tab", value="Actual Sheets", 
                                         help="Tab name with actual sheet inventory")
        proposed_sheets_tab = st.text_input("Proposed Sheets Tab", value="Proposed Sheets",
                                           help="Tab name with available sheet types to purchase")

# Main interface
st.header("📋 Job and Piece Input")

# Kerf setting (default 1/8" = 0.125")
st.subheader("⚙️ Cutting Settings")
kerf_setting = st.number_input(
    "Saw Kerf (inches)", 
    min_value=0.01, 
    max_value=1.0, 
    value=0.125, 
    step=0.001,
    format="%.3f",
    help="Default is 1/8\" (0.125\") - the width of material removed by the saw blade"
)

# Mode selection
sheet_mode = st.radio(
    "Sheet Mode",
    ["Actual", "Proposed", "Both"],
    help="Actual = use existing inventory, Proposed = purchase new sheets, Both = optimize using both"
)

# Job and pieces input
st.subheader("Add Pieces to Cut")

# Use session state to store pieces
if "pieces_list" not in st.session_state:
    st.session_state.pieces_list = []

# Input form for adding pieces
with st.form("add_piece_form"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        job_number = st.text_input("Job #", key="job_input")
    with col2:
        piece_width = st.number_input("Width (in)", min_value=0.1, value=14.0, step=0.1, key="width_input")
    with col3:
        piece_height = st.number_input("Height (in)", min_value=0.1, value=16.0, step=0.1, key="height_input")
    with col4:
        piece_qty = st.number_input("Quantity", min_value=1, value=100, step=1, key="qty_input")
    with col5:
        piece_thickness = st.text_input("Thickness", value="", key="thickness_input", 
                                       help="Optional - leave empty for any thickness")
    
    col_add, col_clear = st.columns(2)
    with col_add:
        add_piece = st.form_submit_button("➕ Add Piece", use_container_width=True)
    with col_clear:
        clear_all = st.form_submit_button("🗑️ Clear All", use_container_width=True)

if add_piece:
    if job_number and piece_width and piece_height and piece_qty:
        thickness_val = safe_float(piece_thickness, None) if piece_thickness else None
        new_piece = Piece(
            job_number=job_number,
            width=piece_width,
            height=piece_height,
            quantity=int(piece_qty),
            thickness=thickness_val
        )
        st.session_state.pieces_list.append(new_piece)
        st.success(f"Added: Job {job_number} - {piece_qty}x {piece_width}\" x {piece_height}\"")
        st.rerun()
    else:
        st.error("Please fill in job number, width, height, and quantity")

if clear_all:
    st.session_state.pieces_list = []
    st.rerun()

# Display current pieces
if st.session_state.pieces_list:
    st.subheader("📝 Pieces to Cut")
    pieces_df = pd.DataFrame([
        {
            "Job #": p.job_number,
            "Width": f"{p.width}\"",
            "Height": f"{p.height}\"",
            "Quantity": p.quantity,
            "Thickness": f"{p.thickness}\"" if p.thickness else "Any"
        }
        for p in st.session_state.pieces_list
    ])
    st.dataframe(pieces_df, use_container_width=True, hide_index=True)
    
    # Optimize button
    if st.button("🔍 Optimize Cutting Plan", type="primary", use_container_width=True):
        if not credentials_json:
            st.error("Please configure Google Sheets credentials first")
        else:
            with st.spinner("Fetching sheet data and optimizing..."):
                client = get_google_sheets_client(credentials_json)
                if not client:
                    st.error("Failed to connect to Google Sheets")
                else:
                    # Fetch actual sheets
                    actual_sheets_data = []
                    if sheet_mode in ["Actual", "Both"]:
                        actual_sheets_data = fetch_sheet_data(client, spreadsheet_id, actual_sheets_tab)
                    
                    # Fetch proposed sheets
                    proposed_sheets_data = []
                    if sheet_mode in ["Proposed", "Both"]:
                        proposed_sheets_data = fetch_sheet_data(client, spreadsheet_id, proposed_sheets_tab)
                    
                    # Convert to Sheet objects
                    available_sheets = []
                    
                    # Process actual sheets
                    for row in actual_sheets_data:
                        status = str(row.get('status', 'Available')).strip()
                        if status == "Available":
                            # Check for drop piece indicator
                            drop_indicator = row.get('drop on drop output') or row.get('drop_on_drop_output') or row.get('drop')
                            is_drop = str(drop_indicator).strip().upper() in ['X', 'Y', 'YES', 'TRUE', '1']
                            
                            sheet = Sheet(
                                part_number=row.get('part_number') or row.get('part number') or row.get('sheet_id') or row.get('id'),
                                description=row.get('description') or row.get('desc'),
                                width=safe_float(row.get('sheet_width') or row.get('width')),
                                height=safe_float(row.get('sheet_length') or row.get('height') or row.get('length')),
                                thickness=safe_float(row.get('thickness'), None),
                                cost=safe_float(row.get('cost') or row.get('cost_per_sheet'), 0),
                                is_actual=True,
                                is_drop=is_drop,
                                status=status
                            )
                            if sheet.width > 0 and sheet.height > 0:
                                available_sheets.append(sheet)
                    
                    # Process proposed sheets
                    for row in proposed_sheets_data:
                        # Check if this proposed sheet is marked as drop (can't be used as proposed)
                        drop_indicator = row.get('drop on drop output') or row.get('drop_on_drop_output') or row.get('drop')
                        is_drop = str(drop_indicator).strip().upper() in ['X', 'Y', 'YES', 'TRUE', '1']
                        
                        # Only add if NOT a drop (empty "Drop on Drop Output" means it can be proposed)
                        if not is_drop:
                            sheet = Sheet(
                                part_number=row.get('part_number') or row.get('part number'),
                                description=row.get('description') or row.get('desc'),
                                width=safe_float(row.get('sheet_width') or row.get('width')),
                                height=safe_float(row.get('sheet_length') or row.get('height') or row.get('length')),
                                thickness=safe_float(row.get('thickness'), None),
                                cost=safe_float(row.get('cost_per_sheet') or row.get('cost'), 0),
                                is_actual=False,
                                is_drop=False
                            )
                            if sheet.width > 0 and sheet.height > 0:
                                available_sheets.append(sheet)
                    
                    if not available_sheets:
                        st.error("No sheets found. Check your sheet tabs and data structure.")
                    else:
                        # Optimize
                        use_actual = sheet_mode in ["Actual", "Both"]
                        optimization_result = optimize_cutting(
                            st.session_state.pieces_list,
                            available_sheets,
                            use_actual,
                            kerf_setting
                        )
                        
                        # Calculate proposed-only cost (for comparison)
                        proposed_sheets_only = [s for s in available_sheets if not s.is_actual]
                        proposed_only_result = optimize_proposed_only(st.session_state.pieces_list, proposed_sheets_only, kerf_setting)
                        
                        # Calculate unoptimized cost (one sheet per piece type, no ganging)
                        unoptimized_result = calculate_unoptimized_cost(st.session_state.pieces_list, available_sheets, use_actual, kerf_setting)
                        
                        # Calculate savings breakdown
                        actual_cost = optimization_result['total_cost']
                        proposed_only_cost = proposed_only_result['total_cost']
                        savings_from_drops = max(0, proposed_only_cost - actual_cost) if use_actual else 0
                        
                        # Count drop pieces used
                        drop_pieces_used = sum(1 for assign in optimization_result['assignments'] 
                                             if assign['sheet'].is_drop)
                        actual_sheets_used = sum(1 for assign in optimization_result['assignments'] 
                                               if assign['sheet'].is_actual and not assign['sheet'].is_drop)
                        proposed_sheets_used = sum(1 for assign in optimization_result['assignments'] 
                                                  if not assign['sheet'].is_actual)
                        
                        # Calculate ganging savings (if multiple jobs/pieces on one sheet)
                        multi_job_sheets = sum(1 for assign in optimization_result['assignments'] 
                                              if len(assign['pieces']) > 1)
                        multi_piece_sheets = sum(1 for assign in optimization_result['assignments'] 
                                                 if len(set(p['piece'].job_number for p in assign['pieces'])) > 1)
                        
                        # Display results
                        st.header("📊 Optimization Results")
                        
                        # Price Comparison Section
                        if use_actual and proposed_only_cost > 0:
                            st.subheader("💰 Price Comparison")
                            comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                            
                            with comp_col1:
                                st.metric(
                                    "Actual Cost (with drops)", 
                                    f"${actual_cost:.2f}",
                                    delta=f"-${savings_from_drops:.2f}" if savings_from_drops > 0 else None,
                                    delta_color="normal"
                                )
                            
                            with comp_col2:
                                st.metric(
                                    "Proposed Cost (full sheets only)", 
                                    f"${proposed_only_cost:.2f}",
                                    delta=f"+${savings_from_drops:.2f}" if savings_from_drops > 0 else None,
                                    delta_color="inverse"
                                )
                            
                            with comp_col3:
                                savings_pct = (savings_from_drops / proposed_only_cost * 100) if proposed_only_cost > 0 else 0
                                st.metric(
                                    "Total Savings", 
                                    f"${savings_from_drops:.2f}",
                                    delta=f"{savings_pct:.1f}%"
                                )
                            
                            with comp_col4:
                                st.metric("Drop Pieces Used", drop_pieces_used)
                            
                            # Savings breakdown
                            with st.expander("📈 Savings Breakdown", expanded=True):
                                savings_data = []
                                
                                if drop_pieces_used > 0:
                                    # Estimate drop piece savings (cost of drop pieces if bought as full sheets)
                                    drop_savings_estimate = 0
                                    for assign in optimization_result['assignments']:
                                        if assign['sheet'].is_drop:
                                            # Find equivalent full sheet cost
                                            equivalent_full = None
                                            for prop_sheet in proposed_sheets_only:
                                                if (prop_sheet.thickness == assign['sheet'].thickness or 
                                                    (prop_sheet.thickness is None and assign['sheet'].thickness is None)):
                                                    if prop_sheet.width >= assign['sheet'].width and prop_sheet.height >= assign['sheet'].height:
                                                        if equivalent_full is None or prop_sheet.cost < equivalent_full.cost:
                                                            equivalent_full = prop_sheet
                                            if equivalent_full:
                                                drop_savings_estimate += equivalent_full.cost - assign['sheet'].cost
                                    
                                    savings_data.append({
                                        "Source": "Using Drop Pieces",
                                        "Savings": f"${drop_savings_estimate:.2f}",
                                        "Details": f"{drop_pieces_used} drop piece(s) used instead of full sheets"
                                    })
                                
                                if multi_job_sheets > 0:
                                    savings_data.append({
                                        "Source": "Ganging Multiple Pieces",
                                        "Savings": "Variable",
                                        "Details": f"{multi_job_sheets} sheet(s) have multiple pieces, reducing total sheets needed"
                                    })
                                
                                if multi_piece_sheets > 0:
                                    savings_data.append({
                                        "Source": "Multi-Job Optimization",
                                        "Savings": "Variable",
                                        "Details": f"{multi_piece_sheets} sheet(s) combine pieces from multiple jobs"
                                    })
                                
                                if savings_data:
                                    st.dataframe(pd.DataFrame(savings_data), use_container_width=True, hide_index=True)
                                else:
                                    st.info("No additional savings breakdown available")
                        
                        # Summary metrics
                        st.subheader("📊 Summary Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Cost", f"${optimization_result['total_cost']:.2f}")
                        with col2:
                            st.metric("Sheets Used", len(optimization_result['sheets_used']))
                        with col3:
                            st.metric("Total Waste", f"{optimization_result['total_waste']:.1f}%")
                        with col4:
                            st.metric("Assignments", len(optimization_result['assignments']))
                        
                        # Sheet type breakdown
                        if use_actual:
                            st.info(f"📦 **Sheet Breakdown:** {actual_sheets_used} actual sheet(s), {drop_pieces_used} drop piece(s), {proposed_sheets_used} proposed sheet(s)")
                        
                        # Detailed assignments
                        st.subheader("📋 Sheet Assignments")
                        
                        # Create cutting programs grouped by thickness
                        cutting_programs = create_cutting_programs(optimization_result)
                        
                        # Production Roadmap Section
                        st.subheader("🏭 Production Roadmap")
                        st.markdown("**Print this section for production team**")
                        
                        # Unoptimized vs Optimized comparison
                        unoptimized_cost = unoptimized_result['total_cost']
                        optimized_cost = optimization_result['total_cost']
                        optimization_savings = unoptimized_cost - optimized_cost
                        optimization_savings_pct = (optimization_savings / unoptimized_cost * 100) if unoptimized_cost > 0 else 0
                        
                        comp_col1, comp_col2, comp_col3 = st.columns(3)
                        with comp_col1:
                            st.metric("Unoptimized Cost", f"${unoptimized_cost:.2f}")
                        with comp_col2:
                            st.metric("Optimized Cost", f"${optimized_cost:.2f}", 
                                     delta=f"-${optimization_savings:.2f}")
                        with comp_col3:
                            st.metric("Optimization Savings", f"${optimization_savings:.2f}", 
                                     delta=f"{optimization_savings_pct:.1f}%")
                        
                        # Cutting Programs by Thickness (Setup Groups)
                        for program_idx, program in enumerate(cutting_programs, 1):
                            with st.expander(f"🔧 SETUP {program_idx}: {program['thickness_display']} Thickness - {program['sheets_count']} Sheet(s) - ${program['cost']:.2f}", expanded=True):
                                sheet = program['sheet']
                                
                                st.markdown(f"**Program Name:** {program['program_name']}")
                                st.markdown(f"**Thickness:** {program['thickness_display']}")
                                st.markdown(f"**Sheet:** {sheet.part_number or 'N/A'} - {sheet.description or ''} ({sheet.width}\" × {sheet.height}\")")
                                st.markdown(f"**Cut this program {program['sheets_count']} time(s)**")
                                
                                # Pieces nested in this program
                                st.markdown("**Pieces in this program (nested cuts):**")
                                program_pieces = []
                                for piece_assign in program['pieces']:
                                    piece = piece_assign['piece']
                                    program_pieces.append({
                                        "Job #": piece.job_number,
                                        "Size": f"{piece.width}\" × {piece.height}\"",
                                        "Quantity": piece_assign['quantity'],
                                        "Parts/Sheet": piece_assign['parts_per_sheet'],
                                        "Orientation": piece_assign['orientation']
                                    })
                                st.dataframe(pd.DataFrame(program_pieces), use_container_width=True, hide_index=True)
                                
                                # Job summary for this program
                                jobs_in_program = set(p['piece'].job_number for p in program['pieces'])
                                st.info(f"📋 **Jobs in this program:** {', '.join(sorted(jobs_in_program))}")
                        
                        # Detailed assignments (collapsed by default)
                        with st.expander("📋 Detailed Sheet Assignments"):
                            for idx, assign in enumerate(optimization_result['assignments'], 1):
                                with st.expander(f"Sheet {idx}: {assign['sheet'].part_number or 'Proposed'} - ${assign['cost']:.2f} - {assign['waste_pct']:.1f}% waste"):
                                    sheet = assign['sheet']
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**Sheet Info**")
                                        sheet_type = "Drop Piece" if sheet.is_drop else ("Actual" if sheet.is_actual else "Proposed")
                                        st.json({
                                            "Part Number": sheet.part_number or "N/A",
                                            "Description": sheet.description or "N/A",
                                            "Size": f"{sheet.width}\" x {sheet.height}\"",
                                            "Thickness": f"{sheet.thickness}\"" if sheet.thickness else "Any",
                                            "Cost": f"${sheet.cost:.2f}",
                                            "Type": sheet_type,
                                            "Sheets Used": assign.get("sheets_count", 1)
                                        })
                                
                                with col2:
                                    st.markdown("**Efficiency**")
                                    st.json({
                                        "Waste": f"{assign['waste_pct']:.1f}%",
                                        "Utilization": f"{assign['utilization_pct']:.1f}%",
                                        "Total Cost": f"${assign['cost']:.2f}"
                                    })
                                
                                st.markdown("**Pieces on This Sheet**")
                                pieces_table = []
                                for piece_assign in assign['pieces']:
                                    piece = piece_assign['piece']
                                    pieces_table.append({
                                        "Job #": piece.job_number,
                                        "Size": f"{piece.width}\" x {piece.height}\"",
                                        "Quantity": piece_assign['quantity'],
                                        "Parts/Sheet": piece_assign['parts_per_sheet'],
                                        "Orientation": piece_assign['orientation']
                                    })
                                st.dataframe(pd.DataFrame(pieces_table), use_container_width=True, hide_index=True)
                        
                        # Unfulfilled pieces
                        if optimization_result['unfulfilled_pieces']:
                            st.warning("⚠️ Some pieces could not be fulfilled")
                            unfulfilled_df = pd.DataFrame([
                                {
                                    "Job #": p['piece'].job_number,
                                    "Size": f"{p['piece'].width}\" x {p['piece'].height}\"",
                                    "Qty Needed": p['quantity_needed'],
                                    "Reason": p['reason']
                                }
                                for p in optimization_result['unfulfilled_pieces']
                            ])
                            st.dataframe(unfulfilled_df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Add pieces above to start optimizing")

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    ### Google Sheet Structure
    
    **Actual Sheets Tab** should have columns:
    - `sheet_id` or `id`: Unique identifier
    - `sheet_width` or `width`: Sheet width in inches
    - `sheet_length` or `height`: Sheet height in inches
    - `thickness`: Material thickness (optional)
    - `cost` or `cost_per_sheet`: Cost per sheet
    - `status`: "Available", "Used", or "Repurchased"
    
    **Proposed Sheets Tab** should have columns:
    - `sheet_width` or `width`: Sheet width in inches
    - `sheet_length` or `height`: Sheet height in inches
    - `thickness`: Material thickness (optional)
    - `cost_per_sheet` or `cost`: Cost per sheet
    
    ### Usage Flow
    1. Configure Google Sheets credentials
    2. Select sheet mode (Actual/Proposed/Both)
    3. Add pieces with job numbers, dimensions, and quantities
    4. Click "Optimize Cutting Plan"
    5. Review optimized assignments with cost and waste analysis
    """)

