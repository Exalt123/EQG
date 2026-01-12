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
    quantity: int = 1  # How many of this sheet are available (critical for optimization)
    is_actual: bool = False  # True for actual sheets, False for proposed
    is_drop: bool = False  # True if this is a drop piece (marked with X in "drop" column)
    status: str = "Available"  # Available or Unavailable
    used_pieces: List[Tuple[float, float]] = None  # Already used areas on the sheet
    
    def __post_init__(self):
        if self.used_pieces is None:
            self.used_pieces = []

# --- HELPER FUNCTIONS ---
def get_thickness_fractions():
    """Generate list of common thickness fractions in 32nds (like tape measure)."""
    # Common fractions in 32nds up to 1 inch
    common_fractions = [
        (1, 32), (1, 16), (3, 32), (1, 8), (5, 32), (3, 16), (7, 32), (1, 4),
        (9, 32), (5, 16), (11, 32), (3, 8), (13, 32), (7, 16), (15, 32), (1, 2),
        (17, 32), (9, 16), (19, 32), (5, 8), (21, 32), (11, 16), (23, 32), (3, 4),
        (25, 32), (13, 16), (27, 32), (7, 8), (29, 32), (15, 16), (31, 32), (1, 1)
    ]
    
    fractions_dict = {}
    for num, denom in common_fractions:
        decimal = num / denom
        # Simplify if possible (e.g., 2/4 -> 1/2)
        g = math.gcd(num, denom)
        simple_num = num // g
        simple_denom = denom // g
        display = f"{simple_num}/{simple_denom}" if simple_num != simple_denom else "1"
        # Only keep if not duplicate (prefer simplified versions)
        if decimal not in fractions_dict or len(display) < len(fractions_dict[decimal]):
            fractions_dict[decimal] = display
    
    # Convert to list and sort by decimal value
    fractions_list = [{'display': v, 'decimal': k} for k, v in sorted(fractions_dict.items())]
    return fractions_list

def decimal_to_fraction(decimal_value, fractions_list=None):
    """Convert decimal to closest fraction from list."""
    if decimal_value is None:
        return None
    if fractions_list is None:
        fractions_list = get_thickness_fractions()
    
    # Find closest fraction
    closest = None
    min_diff = float('inf')
    for frac in fractions_list:
        diff = abs(frac['decimal'] - decimal_value)
        if diff < min_diff:
            min_diff = diff
            closest = frac
    
    # If very close match (within 0.001), return it
    if closest and min_diff < 0.001:
        return closest['display']
    # Otherwise return decimal rounded to 4 places
    return f"{decimal_value:.4f}"

def fraction_to_decimal(fraction_str):
    """Convert fraction string (e.g., '1/4', '7/32') to decimal."""
    if not fraction_str or fraction_str.strip() == '':
        return None
    fraction_str = str(fraction_str).strip()
    
    # If it's already a decimal number
    try:
        return float(fraction_str)
    except ValueError:
        pass
    
    # Try to parse as fraction
    if '/' in fraction_str:
        try:
            numerator, denominator = fraction_str.split('/')
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return None
    
    return None

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
def get_google_sheets_client(credentials_json):
    """Initialize Google Sheets client from service account credentials."""
    try:
        # Validate credentials have required fields
        required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
        missing_fields = [field for field in required_fields if field not in credentials_json]
        if missing_fields:
            st.error(f"Invalid credentials: Missing required fields: {', '.join(missing_fields)}")
            return None
        
        # Method 1: Try service_account_from_dict (gspread 3.0+, handles auth internally)
        if hasattr(gspread, 'service_account_from_dict'):
            try:
                client = gspread.service_account_from_dict(credentials_json)
                # Test connection with a simple operation
                try:
                    # Try to list spreadsheets to verify auth works
                    _ = client.list_spreadsheet_files(limit=1)
                    return client
                except Exception as test_error:
                    # If test fails, fall through to alternative method
                    pass
            except Exception as e:
                # If service_account_from_dict fails, try alternative
                pass
        
        # Method 2: Create credentials with explicit scopes and use authorize
        # Use full read/write scopes (not readonly) to avoid auth issues
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Create credentials with the new google-auth 2.x API
        creds = Credentials.from_service_account_info(
            credentials_json,
            scopes=scopes
        )
        
        # Use gspread.authorize with the credentials
        client = gspread.authorize(creds)
        
        # Test the connection
        try:
            _ = client.list_spreadsheet_files(limit=1)
        except Exception as test_error:
            # Connection test failed, but continue anyway
            pass
            
        return client
            
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        st.error(f"❌ **Error connecting to Google Sheets ({error_type})**")
        
        # Provide helpful error messages based on error type
        if "permission" in error_msg.lower() or "403" in error_msg or isinstance(e, PermissionError):
            service_email = credentials_json.get('client_email', 'your service account email')
            st.warning(f"""
**Permission Error - Action Required:**

1. Open your Google Sheet
2. Click **"Share"** button (top right)
3. Add this email: `{service_email}`
4. Give it **"Viewer"** access
5. Click **"Send"** or **"Share"**
6. Refresh this page and try again
            """)
        elif "404" in error_msg or "not found" in error_msg.lower():
            st.warning("""
**Spreadsheet Not Found:**
- Check that the Spreadsheet ID is correct
- Make sure the spreadsheet exists and is accessible
            """)
        elif "_auth_request" in error_msg or "AttributeError" in error_type:
            st.warning("""
**Authentication Error Detected:**

This might be a library version issue. Try:

1. **Clear the page cache** (refresh the page)
2. **Update your libraries**:
   ```bash
   pip install --upgrade gspread google-auth
   ```
3. **Try again** - the app will attempt alternative authentication methods
            """)
        
        # Always show technical details for debugging
        with st.expander("🔍 Technical Error Details"):
            import traceback
            st.code(traceback.format_exc(), language='python')
        
        return None

def fetch_sheet_data(client, spreadsheet_id, sheet_name_or_id, credentials_json=None):
    """Fetch data from a specific sheet in the spreadsheet."""
    try:
        # Try to open the spreadsheet - this is where permission errors often occur
        try:
            spreadsheet = client.open_by_key(spreadsheet_id)
        except (PermissionError, Exception) as open_error:
            error_msg = str(open_error)
            error_type = type(open_error).__name__
            
            # If permission error, try to list available spreadsheets to help debug
            if isinstance(open_error, PermissionError) or "403" in error_msg or "permission" in error_msg.lower():
                try:
                    # Try to list spreadsheets we DO have access to
                    available_spreadsheets = client.list_spreadsheet_files(limit=20)
                    if available_spreadsheets:
                        spreadsheet_list = "\n".join([f"  • **{s['name']}** (ID: `{s['id']}`)" for s in available_spreadsheets[:10]])
                        st.info(f"""
**📋 Spreadsheets you have access to:**
{spreadsheet_list}

**⚠️ You're trying to access:** `{spreadsheet_id}`

If your spreadsheet isn't in the list above, check:
1. Is the Spreadsheet ID correct? (from URL: `/d/SPREADSHEET_ID/edit`)
2. Is it shared with: `{credentials_json.get('client_email', 'your service account') if credentials_json else 'N/A'}`
                        """)
                    else:
                        st.warning("No spreadsheets found. Make sure your service account has access to at least one spreadsheet.")
                except Exception as list_error:
                    # If we can't even list spreadsheets, it's a broader permission issue
                    pass
            
            # Re-raise the original error to be caught by the outer exception handler
            raise open_error
        
        # Handle sheet name or ID
        if isinstance(sheet_name_or_id, int) or (isinstance(sheet_name_or_id, str) and sheet_name_or_id.replace('gid=', '').isdigit()):
            sheet_id_str = str(sheet_name_or_id).replace('gid=', '')
            try:
                sheet = spreadsheet.get_worksheet_by_id(int(sheet_id_str))
            except Exception as e1:
                # Try to find by iterating through sheets
                try:
                    all_sheets = spreadsheet.worksheets()
                    sheet = None
                    for s in all_sheets:
                        if str(s.id) == sheet_id_str:
                            sheet = s
                            break
                    if sheet is None:
                        # List available sheets for debugging
                        available_ids = [str(s.id) for s in all_sheets]
                        available_names = [s.title for s in all_sheets]
                        raise ValueError(
                            f"Sheet ID '{sheet_name_or_id}' not found. "
                            f"Available sheet IDs: {available_ids[:5]}, "
                            f"Available names: {available_names[:5]}"
                        )
                except Exception as e2:
                    raise ValueError(f"Could not find sheet ID '{sheet_name_or_id}': {str(e2)}")
            except Exception as e1:
                raise ValueError(f"Error accessing sheet: {str(e1)}")
        else:
            try:
                sheet = spreadsheet.worksheet(sheet_name_or_id)
            except Exception as e:
                # List available sheet names for debugging
                try:
                    all_sheets = spreadsheet.worksheets()
                    available_names = [s.title for s in all_sheets]
                    raise ValueError(
                        f"Sheet name '{sheet_name_or_id}' not found. "
                        f"Available sheet names: {', '.join(available_names[:10])}"
                    )
                except Exception as e2:
                    raise ValueError(f"Could not find sheet '{sheet_name_or_id}': {str(e)}")
        
        records = sheet.get_all_records()
        result = []
        for record in records:
            clean_row = {k.strip().lower(): v for k, v in record.items()}
            result.append(clean_row)
        return result
    except PermissionError as e:
        # Get service account email from credentials to show in error
        try:
            service_account_email = credentials_json.get('client_email', 'your service account email')
        except:
            service_account_email = 'your service account email'
        
        error_msg = str(e)
        st.error("❌ **Permission Error: Cannot access spreadsheet**")
        
        # Show technical details
        with st.expander("🔍 Debug Information"):
            st.code(f"Spreadsheet ID: {spreadsheet_id}\nService Account: {service_account_email}\nError: {error_msg}", language='text')
        
        st.warning(f"""
**Troubleshooting:**

1. **Verify the Spreadsheet ID:**
   - Open your Google Sheet "MDFSHEETINVENTORY"
   - Look at the URL: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`
   - Copy the ID (between `/d/` and `/edit`)
   - Current ID being used: `{spreadsheet_id}`

2. **Check sharing:**
   - The sheet should be shared with: `{service_account_email}`
   - You showed it has "Editor" access - that's good!

3. **Common issues:**
   - Spreadsheet ID has extra characters or spaces
   - Using the wrong spreadsheet (different one with same name)
   - Permissions haven't propagated yet (wait 30 seconds)

4. **Try:**
   - Copy the Spreadsheet ID from the URL again
   - Paste it fresh in the "Spreadsheet ID" field
   - Make sure there are no spaces before/after
        """)
        return []
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        # Check for common error types
        if "403" in error_msg or "permission" in error_msg.lower():
            st.error("❌ **Permission Error: Service account doesn't have access**")
            st.warning("""
**To fix:**
1. Open your Google Sheet
2. Click "Share" button
3. Add your service account email (found in JSON file under "client_email")
4. Give it "Viewer" access
5. Try again!
            """)
        elif "404" in error_msg or "not found" in error_msg.lower():
            st.error(f"❌ **Sheet Not Found: '{sheet_name_or_id}'**")
            st.info("Check that the sheet name/ID is correct and exists in your spreadsheet.")
        else:
            st.error(f"Error fetching sheet data ({error_type}): {error_msg}")
        
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
                results["sheets_used"].append(best_sheet.part_number or f"Proposed-{len(results['sheets_used'])}")
    
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
    
    # Track sheet usage count (how many times each part_number has been used)
    sheet_usage_count = {}
    for sheet in actual_sheets:
        if sheet.part_number:
            if sheet.part_number not in sheet_usage_count:
                sheet_usage_count[sheet.part_number] = {"used": 0, "quantity": sheet.quantity, "sheet_template": sheet}
            else:
                # If same part_number appears multiple times, use the max quantity
                sheet_usage_count[sheet.part_number]["quantity"] = max(
                    sheet_usage_count[sheet.part_number]["quantity"], sheet.quantity
                )
    
    # First pass: Try to fill actual sheets if use_actual=True
    # Use a while loop to continue until all sheets are exhausted or all pieces fulfilled
    if use_actual and actual_sheets:
        max_iterations = 1000  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for sheet in actual_sheets:
                # Check if we've used all available quantity of this sheet type
                if sheet.part_number:
                    usage_info = sheet_usage_count.get(sheet.part_number)
                    if usage_info and usage_info["used"] >= usage_info["quantity"]:
                        continue  # Skip if we've used all available quantity
                
                sheet_assignments = []
                sheet_used_area = 0  # Each sheet starts fresh
                available_area = sheet.width * sheet.height
                
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
                            made_progress = True
                
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
                    
                    # Track that we used this sheet
                    if sheet.part_number:
                        sheet_usage_count[sheet.part_number]["used"] += 1
            
            # If no progress was made, break
            if not made_progress:
                break
    
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
        spreadsheet_id = st.text_input("Spreadsheet ID", value="1FWbgDKLZ623s_VtmB75HucEr7pl7SO-pm6TzH2ON45s")
        
        st.subheader("Sheet Configuration")
        sheet_tab_name = st.text_input("Sheet Tab Name", value="Sheet1", 
                                       help="Name of the tab in your Google Sheet containing the material data")
        
        st.info("💡 **Tip:** The app will automatically distinguish between:\n"
                "• **Actual sheets** (drop pieces with 'X' in drop column, or status='Available' with quantity > 0)\n"
                "• **Proposed sheets** (full sheets without 'X' in drop column)")

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
        thickness_options = get_thickness_fractions()
        thickness_display_options = [""] + [f"{frac['display']}\" ({frac['decimal']:.4f}\")" for frac in thickness_options]
        thickness_selected = st.selectbox(
            "Thickness", 
            options=thickness_display_options,
            key="thickness_input",
            help="Select thickness from dropdown (fractions like tape measure) or leave empty for any"
        )
        # Extract fraction from selection (e.g., "1/4\" (0.2500\")" -> "1/4")
        piece_thickness = ""
        if thickness_selected and thickness_selected != "":
            # Extract the fraction part before the backslash
            piece_thickness = thickness_selected.split('"')[0].strip()
    
    col_add, col_clear = st.columns(2)
    with col_add:
        add_piece = st.form_submit_button("➕ Add Piece", use_container_width=True)
    with col_clear:
        clear_all = st.form_submit_button("🗑️ Clear All", use_container_width=True)

if add_piece:
    if job_number and piece_width and piece_height and piece_qty:
        # Convert fraction to decimal if provided
        thickness_val = fraction_to_decimal(piece_thickness) if piece_thickness else None
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
    thickness_fractions = get_thickness_fractions()
    pieces_df = pd.DataFrame([
        {
            "Job #": p.job_number,
            "Width": f"{p.width}\"",
            "Height": f"{p.height}\"",
            "Quantity": p.quantity,
            "Thickness": decimal_to_fraction(p.thickness, thickness_fractions) if p.thickness else "Any"
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
                    # Fetch data from the single sheet tab
                    all_sheets_data = fetch_sheet_data(client, spreadsheet_id, sheet_tab_name, credentials_json)
                    
                    # Separate into actual and proposed based on data
                    actual_sheets_data = []
                    proposed_sheets_data = []
                    
                    for row in all_sheets_data:
                        # Check if it's a drop piece (has 'X' in drop column)
                        drop_value = str(row.get('drop', '') or '').strip().upper()
                        is_drop = drop_value in ['X', 'Y', 'YES', 'TRUE', '1']
                        
                        # Check status and quantity
                        status = str(row.get('status', 'Available')).strip().upper()
                        quantity = int(safe_float(row.get('quantity') or row.get('qty'), 0))
                        is_available = status in ['AVAILABLE', 'AVAIL', 'YES', 'Y', 'TRUE', '1', 'AV'] and quantity > 0
                        
                        # Actual sheets: drop pieces OR available sheets with quantity > 0
                        if is_drop or is_available:
                            actual_sheets_data.append(row)
                        
                        # Proposed sheets: all non-drop sheets (can be purchased as full sheets)
                        # Include even if currently unavailable, as they can be proposed for purchase
                        if not is_drop:
                            proposed_sheets_data.append(row)
                    
                    # Filter based on sheet_mode selection
                    if sheet_mode == "Actual":
                        proposed_sheets_data = []  # Don't use proposed if only Actual selected
                    elif sheet_mode == "Proposed":
                        actual_sheets_data = []  # Don't use actual if only Proposed selected
                    # If "Both", use both lists as is
                    
                    # Convert to Sheet objects
                    available_sheets = []
                    
                    # First pass: Load all sheets into a dictionary for lookup
                    all_sheets_dict = {}
                    for row in actual_sheets_data:
                        part_num = row.get('part_number') or row.get('part number') or row.get('sheet_id') or row.get('id')
                        if part_num:
                            all_sheets_dict[part_num] = row
                    
                    # Process actual sheets
                    for row in actual_sheets_data:
                        part_num = row.get('part_number') or row.get('part number') or row.get('sheet_id') or row.get('id')
                        
                        status = str(row.get('status', 'Available')).strip()
                        quantity = int(safe_float(row.get('quantity') or row.get('qty'), 1))
                        
                        # Check if unavailable or quantity is 0
                        is_unavailable = status.upper() in ['UNAVAILABLE', 'UNAVAIL', 'NO', 'N', 'FALSE', '0'] or quantity <= 0
                        
                        # Track if we're using a substitute
                        original_part_num = part_num
                        substitute_part_num = None
                        
                        # If unavailable, check for substitute
                        if is_unavailable:
                            substitute_part = row.get('substitute') or row.get('sub')
                            if substitute_part:
                                substitute_part = str(substitute_part).strip()
                                # Look up the substitute sheet
                                if substitute_part in all_sheets_dict:
                                    substitute_row = all_sheets_dict[substitute_part]
                                    # Check if substitute is available
                                    sub_status = str(substitute_row.get('status', 'Available')).strip()
                                    sub_quantity = int(safe_float(substitute_row.get('quantity') or substitute_row.get('qty'), 1))
                                    if sub_status.upper() in ['AVAILABLE', 'AVAIL', 'YES', 'Y', 'TRUE', '1'] and sub_quantity > 0:
                                        # Use the substitute sheet data
                                        row = substitute_row.copy()
                                        substitute_part_num = substitute_part
                        
                        # Now process the sheet (either original or substitute)
                        status = str(row.get('status', 'Available')).strip()
                        if status.upper() in ['AVAILABLE', 'AVAIL', 'YES', 'Y', 'TRUE', '1']:
                            # Check for drop piece indicator
                            drop_indicator = row.get('drop') or row.get('drop on drop output') or row.get('drop_on_drop_output')
                            is_drop = str(drop_indicator).strip().upper() in ['X', 'Y', 'YES', 'TRUE', '1']
                            
                            # Get quantity - critical for optimization
                            quantity = int(safe_float(row.get('quantity') or row.get('qty'), 1))
                            if quantity <= 0:
                                continue  # Skip if no quantity available
                            
                            # Use original part_number for tracking (so we know what was requested)
                            # But use substitute's dimensions, cost, quantity
                            sheet = Sheet(
                                part_number=original_part_num,  # Keep original part_number
                                description=row.get('description') or row.get('desc'),
                                width=safe_float(row.get('sheet_width') or row.get('width')),
                                height=safe_float(row.get('sheet_length') or row.get('height') or row.get('length')),
                                thickness=safe_float(row.get('thickness'), None),
                                cost=safe_float(row.get('cost') or row.get('cost_per_sheet'), 0),
                                quantity=quantity,
                                is_actual=True,
                                is_drop=is_drop,
                                status="Available"
                            )
                            if sheet.width > 0 and sheet.height > 0:
                                # Add substitute info to description if using substitute
                                if substitute_part_num:
                                    sheet.description = (sheet.description or '') + f" (Substitute: {substitute_part_num})"
                                available_sheets.append(sheet)
                    
                    # Process proposed sheets
                    for row in proposed_sheets_data:
                        # Check if this proposed sheet is marked as drop (can't be used as proposed)
                        drop_indicator = row.get('drop') or row.get('drop on drop output') or row.get('drop_on_drop_output')
                        is_drop = str(drop_indicator).strip().upper() in ['X', 'Y', 'YES', 'TRUE', '1']
                        
                        # Only add if NOT a drop (empty "drop" means it can be proposed)
                        if not is_drop:
                            # Proposed sheets don't have quantity limits (unlimited availability)
                            sheet = Sheet(
                                part_number=row.get('part_number') or row.get('part number'),
                                description=row.get('description') or row.get('desc'),
                                width=safe_float(row.get('sheet_width') or row.get('width')),
                                height=safe_float(row.get('sheet_length') or row.get('height') or row.get('length')),
                                thickness=safe_float(row.get('thickness'), None),
                                cost=safe_float(row.get('cost_per_sheet') or row.get('cost'), 0),
                                quantity=999999,  # Unlimited for proposed sheets
                                is_actual=False,
                                is_drop=False
                            )
                            if sheet.width > 0 and sheet.height > 0:
                                available_sheets.append(sheet)
                    
                    if not available_sheets:
                        # Provide more helpful error message
                        # Provide more helpful error message
                        error_details = []
                        if not all_sheets_data:
                            error_details.append(f"• No data found in '{sheet_tab_name}' tab")
                        else:
                            error_details.append(f"• Found {len(all_sheets_data)} total rows in '{sheet_tab_name}' tab")
                            
                        if sheet_mode in ["Actual", "Both"]:
                            if not actual_sheets_data:
                                error_details.append(f"• No actual sheets found (check 'drop' column and status/quantity)")
                            else:
                                error_details.append(f"• Found {len(actual_sheets_data)} actual sheets but none matched criteria")
                        if sheet_mode in ["Proposed", "Both"]:
                            if not proposed_sheets_data:
                                error_details.append(f"• No proposed sheets found (full sheets without 'X' in drop column)")
                            else:
                                error_details.append(f"• Found {len(proposed_sheets_data)} proposed sheets but none matched criteria")
                        
                        st.error("No sheets found. Check your sheet tab and data structure.")
                        if error_details:
                            st.info("Details:\n" + "\n".join(error_details))
                            st.info("Make sure your sheet has:\n• Correct column headers (part_number, sheet_width, sheet_length, etc.)\n• For actual sheets: 'drop' column with 'X' OR status = 'Available'/'Av' with quantity > 0\n• For proposed sheets: rows without 'X' in drop column (full sheets)\n• Valid dimensions (sheet_width and sheet_length > 0)")
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
                            thickness_fractions_detail = get_thickness_fractions()
                            for idx, assign in enumerate(optimization_result['assignments'], 1):
                                with st.expander(f"Sheet {idx}: {assign['sheet'].part_number or 'Proposed'} - ${assign['cost']:.2f} - {assign['waste_pct']:.1f}% waste"):
                                    sheet = assign['sheet']
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**Sheet Info**")
                                        sheet_type = "Drop Piece" if sheet.is_drop else ("Actual" if sheet.is_actual else "Proposed")
                                        thickness_display_json = decimal_to_fraction(sheet.thickness, thickness_fractions_detail) if sheet.thickness else "Any"
                                        if sheet.thickness and thickness_display_json and thickness_display_json != f"{sheet.thickness:.4f}":
                                            thickness_display_json = f"{thickness_display_json}\" ({sheet.thickness:.4f}\")"
                                        elif sheet.thickness:
                                            thickness_display_json = f"{sheet.thickness:.4f}\""
                                        st.json({
                                            "Part Number": sheet.part_number or "N/A",
                                            "Description": sheet.description or "N/A",
                                            "Size": f"{sheet.width}\" x {sheet.height}\"",
                                            "Thickness": thickness_display_json,
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
    
    **Your Sheet Tab** should have these columns:
    - `part_number` or `sheet_id`: Unique identifier for the sheet
    - `description`: Description of the material (optional)
    - `sheet_width` or `width`: Sheet width in inches
    - `sheet_length` or `height`: Sheet height in inches
    - `thickness`: Material thickness (decimal, e.g., 0.1875)
    - `cost` or `cost_per_sheet`: Cost per sheet
    - `drop`: Put "X" in this column if it's a drop piece (leftover)
    - `quantity`: Number of sheets available in inventory (0 = unavailable)
    - `status`: "Available" or "Av" for available, "Unavailable" or "Ur" for unavailable
    - `substitute`: (Optional) Part number of a substitute material if this one is unavailable
    
    **How it works:**
    - **Actual sheets**: Rows with "X" in the `drop` column, OR rows with `status`="Available" and `quantity` > 0
    - **Proposed sheets**: All rows (representing full sheets that can be purchased)
    
    ### Usage Flow
    1. Configure Google Sheets credentials
    2. Select sheet mode (Actual/Proposed/Both)
    3. Add pieces with job numbers, dimensions, and quantities
    4. Click "Optimize Cutting Plan"
    5. Review optimized assignments with cost and waste analysis
    """)

