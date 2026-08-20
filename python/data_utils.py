import os
import platform
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import random

# ******************************* system path handling *******************************
os_name = platform.system()
os_version = platform.release()

Windows_systemPath='D:'
Linux_systemPath='/home/debamalya'
if os_name == 'Windows':
    system_path=Windows_systemPath
elif os_name == 'Linux':
    system_path=Linux_systemPath

def clean_path(path):
    path = str(path)
    if (Windows_systemPath in path):
        path = path.replace(Windows_systemPath, system_path)
    elif (Linux_systemPath in path):
        path = path.replace(Linux_systemPath, system_path)
    else:
        path=f'{system_path}/{path}'
    path = path.replace('/',os.sep).replace('\\',os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        relative_levels = len([directory for directory in path.split(os.sep) if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels]\
                         + [directory for directory in path.split(os.sep) if directory!=''][relative_levels:])
    return Path(path)

# -----------------------------------------------------------------------------------------------------------------

def parse_complex(x):
    if x is None:
        raise ValueError(f"parse_complex: None value is not a valid complex: {x}")

    if isinstance(x, complex):
        return x

    x_str = str(x).strip()

    if x_str.lower() == "not converged":
        return complex(np.nan, np.nan)

    try:
        return complex(x_str)
    except Exception as e:
        raise ValueError(f"parse_complex: Could not parse '{x_str}' to complex") from e

# -----------------------------------------------------------------------------------------------------------------

def parse_master_log(log_path, system):
    entries = {}
    if not log_path.exists():
        raise FileNotFoundError(f"Master log file not found at: {log_path}")

    with open(log_path, 'r') as f:
        content = f.read()

    blocks = content.split('#*#*_')

    for block in blocks:
        if not block.strip('*'):
            continue
        
        try:
            header_line, rest_of_block = block.split('\n', 1)
            number = header_line.split('_')[0]
            param_text = rest_of_block.split('#')[0]
            entries[number] = param_text.strip()
        except (IndexError, ValueError):
            continue
            
    if entries:
        first_entry_text = next(iter(entries.values()))
        existing_system = first_entry_text.split('\n')[0].strip()
        
        if existing_system != system:
            print(f"System mismatch! The master log file is at \n{log_path}", flush=True)
            raise ValueError(f"The master log file belongs to \n**{existing_system}**, \nbut your script is currently set to \n**{system}**.")

    return entries

# -----------------------------------------------------------------------------------------------------------------

def merge_and_sort_csv(file_source_temp, file_existing_data, output_path, param_cols, result_cols, system, prompt_user=True):
    PARAMS_FILENAME = "parameters.txt"
    MASTER_LOG_FILENAME = "logfile.txt"
    
    file1_path = clean_path(file_source_temp)
    file2_path = clean_path(file_existing_data)
    output_path = clean_path(output_path)

    source_params_path1 = file1_path / PARAMS_FILENAME
    # NOTE: source_params_path2 removed — we will locate the matching existing CSV via logfile in file_existing_data
    master_log_path = file2_path / MASTER_LOG_FILENAME   # <-- changed: logfile lives in file_existing_data

    print(f"-----------  Temporary source path is {file1_path} -----------", flush=True)
    print(f"-----------  Destinantion Data path is {file2_path} ----------", flush=True)
    print(f"-----------  output path is {output_path} --------------------", flush=True)

    # --- Validate source params file exists and read it
    if not source_params_path1.exists():
        raise FileNotFoundError(f"Missing file: {source_params_path1}")

    with open(source_params_path1, 'r') as f:
        new_params_content = f.read().strip()

    first_line = new_params_content.split('\n')[0].strip()

    if first_line != system:
        print(first_line)
        raise ValueError(
            f"It is not the \n**{system}** \nin the first line of \n**{source_params_path1}** \nPlease check the file content."
        )

    # --- Locate matching entry in the master logfile inside file_existing_data
    if not master_log_path.exists():
        raise FileNotFoundError(f"Missing master logfile: {master_log_path}")

    existing_entries = parse_master_log(master_log_path, system)
    matched_number = None

    # compare stripped blocks for an exact match
    for number, params_text in existing_entries.items():
        if params_text.strip() == new_params_content:
            matched_number = number
            break

    if matched_number is None:
        # no matching parameter block found in logfile
        raise ValueError(
            f"No matching entry found in '{master_log_path}'.\n"
            f"The parameters in '{source_params_path1}' do not match any 5-digit block in the logfile."
        )

    if prompt_user:
        checked = input("Have you checked files paths? Type yes to continue: ")
        if checked != "yes":
            raise ValueError("Please check files paths before running the script.")

    # found a matching 5-digit ID: set the existing-data CSV path accordingly
    f1 = file1_path / 'data.csv'
    f2 = file2_path / f"{matched_number}.csv"

    print(f"Matched logfile ID: {matched_number} -> Using existing CSV: {f2}", flush=True)



    final_output_file = output_path / 'data.csv'

    # If output_path is the same as the existing-data folder (file2_path)
    # and we have a matched 5-digit file, then write into that 5-digit CSV instead.
    if (output_path.resolve() == file2_path.resolve()) and matched_number:
        final_output_file = file2_path / f"{matched_number}.csv"

    # Ensure output folder exists (keeps previous behavior)
    final_output_file.parent.mkdir(parents=True, exist_ok=True)


    try:
        exists1 = f1.exists()
        exists2 = f2.exists()

        if not exists1 and not exists2:
            print(f"Error: Neither {f1} nor {f2} exists. Nothing to do.", flush=True)
            return

        elif exists1 and not exists2:
            print('-'*80, flush=True)
            print(f"{f2} not found.\n Copying {f1}\n to {output_path} ...\n Type agree to do that copy", flush=True)
            print('-'*80, flush=True)
            if (input("Check output crtically before proceeding") != 'agree'):
                raise ValueError("User aborted the copy operation.")
            shutil.copy2(f1, output_path)
            return

        elif not exists1 and exists2:
            print('-'*80, flush=True)
            print(f"{f1} not found.\n Copying {f2}\n to {output_path} ...\n Type agree to do that copy", flush=True)
            print('-'*80, flush=True)
            if (input("Check output crtically before proceeding") != 'agree'):
                raise ValueError("User aborted the copy operation.")  
            shutil.copy2(f2, output_path)
            return
    
        df1 = pd.read_csv(f1, sep=None, engine='python')
        df2 = pd.read_csv(f2, sep=None, engine='python')

        # MODIFICATION 2: Removed hardcoded lists. 
        # Using the arguments passed to the function instead.
        all_cols = param_cols + result_cols

        if not set(all_cols).issubset(df1.columns) or not set(all_cols).issubset(df2.columns):
            raise ValueError(f"Input files must contain columns: {all_cols}")

        df1['_source'] = 'existing'
        df1['_orig_idx'] = df1.index + 2
        df2['_source'] = 'incoming'
        df2['_orig_idx'] = df2.index + 2

        # Now it is safe to select them
        cols_to_keep = all_cols + ['_source', '_orig_idx']
        df1 = df1[cols_to_keep]
        df2 = df2[cols_to_keep]

        combined_df = pd.concat([df1, df2], ignore_index=True)

        combined_df = combined_df.drop_duplicates(subset=all_cols)

        indices_to_drop = []
        
        if combined_df.duplicated(subset=param_cols).any():
            conflicts = combined_df[combined_df.duplicated(subset=param_cols, keep=False)]
            
            conflict_existing_indices = []
            conflict_incoming_indices = []
            any_conflict_found = False

            for params, group in conflicts.groupby(param_cols):
                conflict_found = False
                for col in result_cols:
                    # 1. Convert column to complex numbers
                    vals = group[col].apply(parse_complex).values
                    
                    ref = vals[0]

                    # Check if values match the reference
                    # rtol=1e-3: Allows 0.1% deviation for Large numbers (e.g. 0.300 vs 0.301 is OK)
                    # atol=1e-3: Allows 0.001 deviation for Tiny numbers (e.g. 0.0001 vs -0.0001 is OK)
                    matches = np.isclose(vals, ref, rtol=1e-3, atol=1e-3)

                    # If any value is NOT a match (and not NaN), it is a conflict
                    if not np.all(matches | np.isnan(vals)):
                        conflict_found = True
                        break

                if conflict_found:
                    any_conflict_found = True
                    row_existing = group[group['_source'] == 'existing']
                    row_incoming = group[group['_source'] == 'incoming']

                    conflict_existing_indices.extend(row_existing.index.tolist())
                    conflict_incoming_indices.extend(row_incoming.index.tolist())

                    ex_rows_str = ", ".join(row_existing['_orig_idx'].astype(str)) if not row_existing.empty else "None"
                    in_rows_str = ", ".join(row_incoming['_orig_idx'].astype(str)) if not row_incoming.empty else "None"

                    # Handle params printing for arbitrary length
                    clean_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in zip(param_cols, params)}
                    
                    print(f"\n[!] Conflict detected for {clean_params}", flush=True)
                    print(f"    Existing (Rows {ex_rows_str}):", flush=True)
                    print(row_existing[result_cols].to_string(index=False, header=False), flush=True)
                    print(f"    Incoming (Rows {in_rows_str}):", flush=True)
                    print(row_incoming[result_cols].to_string(index=False, header=False), flush=True)

            if any_conflict_found:
                print("\n" + "="*80, flush=True)
                while True:
                    choice = input(">>> Resolve: Keep [E]xisting (ignore incoming) or Prefer [I]ncoming (overwrite existing)? (existing/incoming): ").strip().lower()
                    
                    if choice == 'existing' or choice == 'e':
                        indices_to_drop.extend(conflict_incoming_indices)
                        print("    -> Action: Keeping Existing Data for ALL conflicts.", flush=True)
                        break
                    elif choice == 'incoming' or choice == 'i':
                        indices_to_drop.extend(conflict_existing_indices)
                        print("    -> Action: Overwriting with Incoming Data for ALL conflicts.", flush=True)
                        break
                    else:
                        print("    Invalid input. Please type 'existing' or 'incoming'.", flush=True)

        if indices_to_drop:
            combined_df = combined_df.drop(indices_to_drop)

        # Remove remaining duplicates (those within tolerance or resolved), keeping the last entry
        combined_df = combined_df.drop_duplicates(subset=param_cols, keep='last')

        sorted_df = combined_df.sort_values(by=param_cols, ascending=True)
        final_df = sorted_df[all_cols] 

        temp_out = str(final_output_file.with_suffix('.tmp'))
        final_df.to_csv(temp_out, index=False)
        os.replace(temp_out, final_output_file)
        print(f"Successfully merged {len(final_df)} rows into '{output_path}'.", flush=True)

    except ValueError as e:
        print(f"Error: {e}", flush=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", flush=True)

# -----------------------------------------------------------------------------------------------------------------

def merge_and_minimize_csv(file_paths, output_path, param_cols, result_cols, target_col, prompt_user=True):
    """
    Merges multiple CSV files (each expected to have data.csv) assuming parameters
    are identical across files. For each unique parameter set selects the row
    where `target_col` (a member of result_cols) is minimized (based on real part).
    Ties are broken deterministically by:
      1) smallest absolute imaginary part of target,
      2) lexicographic source filename,
      3) original row index (smallest).
    The output layout keeps only param_cols + result_cols.
    """
    output_path = clean_path(output_path)

    print(f"----------- OUTPUT_PATH is {output_path} -----------", flush=True)

    if prompt_user:
        checked = input("Have you checked OUTPUT_PATH? Type yes to continue: ")
        if checked != "yes":
            raise ValueError("Please check OUTPUT_PATH before running the script.")
    try:
        # Ensure arguments are lists
        if isinstance(param_cols, str):
            param_cols = [param_cols]
        if isinstance(result_cols, str):
            result_cols = [result_cols]

        all_cols = list(param_cols) + list(result_cols)

        if target_col not in result_cols:
            raise ValueError(f"target_col '{target_col}' must be in result_cols.")

        normalized_paths = [clean_path(p) for p in file_paths]

        dfs = []
        # 1. Read all valid CSV files
        for path in normalized_paths:
            f = path / 'data.csv'
            if f.exists():
                df = pd.read_csv(f, sep=None, engine='python')

                # Verify columns exist
                if not set(all_cols).issubset(df.columns):
                    print(f"Warning: {f} is missing required columns. Skipping.", flush=True)
                    continue

                # Track source and original row index
                df['_source_file'] = str(f)
                df['_orig_idx'] = df.index + 2

                # Keep only the necessary columns + tracking columns
                df = df[all_cols + ['_source_file', '_orig_idx']]
                dfs.append(df)
            else:
                print(f"Note: {f} not found. Skipping.", flush=True)

        if not dfs:
            print("Error: No valid data.csv files found in the provided paths.", flush=True)
            return

        # 2. Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # -------------------------------------------------------------------
        # MINIMAL ROBUSTNESS EDIT (Step 2.5): 
        # Coerce parameters to numeric if possible to prevent 
        # type mismatch bugs (e.g. 1.0 vs "1.0") breaking deduplication.
        # -------------------------------------------------------------------
        for col in param_cols:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='ignore')
            # If they remain strings (e.g. categorical params), strip whitespace
            if combined_df[col].dtype == object:
                combined_df[col] = combined_df[col].astype(str).str.strip()

        # 3. Helper function to extract comparable numbers (minimize on real part)
        def get_comparable_value(x):
            try:
                c = parse_complex(x)
                if np.isnan(c.real):
                    return np.inf
                return float(c.real)
            except Exception:
                return np.inf

        def get_imag_abs(x):
            try:
                c = parse_complex(x)
                if np.isnan(c.imag):
                    return np.inf
                return float(abs(c.imag))
            except Exception:
                return np.inf

        # Create temporary columns for sorting/minimizing
        combined_df['_target_val'] = combined_df[target_col].apply(get_comparable_value)
        combined_df['_target_imag_abs'] = combined_df[target_col].apply(get_imag_abs)

        # 4. Deterministic tie-breaking:
        sort_cols = list(param_cols) + ['_target_val', '_target_imag_abs', '_source_file', '_orig_idx']
        combined_df = combined_df.sort_values(by=sort_cols, ascending=True, kind='mergesort')

        # 5. Keep the first occurrence per parameter combination
        final_df = combined_df.drop_duplicates(subset=param_cols, keep='first').copy()

        # 6. Final sort by parameter columns and drop temporary columns
        final_df = final_df.sort_values(by=param_cols, ascending=True)
        final_cols_to_save = list(all_cols)  
        final_df = final_df[final_cols_to_save]

        # 7. Save to output
        output_path.mkdir(parents=True, exist_ok=True)
        temp_out = str(output_path / "data.tmp")
        final_df.to_csv(temp_out, index=False)
        os.replace(temp_out, output_path / 'data.csv')

        print(f"Successfully minimized and merged {len(final_df)} unique parameter rows into '{output_path}'.", flush=True)

    except Exception as e:
        print(f"An unexpected error occurred: {e}", flush=True)

# -----------------------------------------------------------------------------------------------------------------

def delete_contents_from_log(filename, start_markers, end_marker=None):
    if not start_markers:
        return

    # Convert markers to a set of strings for fast, foolproof matching
    start_markers = {str(m) for m in start_markers}

    try:
        with open(filename, "r") as f:
            text = f.read()

        # Bulletproof Regex:
        # \*+         : Matches any number of asterisks
        # #\*#\*_     : Matches exactly #*#*_
        # (\d+)       : Captures the ID (group 1)
        # _#\*#\*+    : Matches _#*# followed by any number of asterisks
        # \n.*?       : Non-greedily matches the content inside (re.DOTALL makes . match newlines)
        # #+(?:\n|$)  : Matches the bottom hashes and its newline (or end of file)
        block_pattern = re.compile(
            r"\*+#\*#\*_(\d+)_#\*#\*+\n.*?#+(?:\n|$)",
            re.DOTALL
        )

        new_text = []
        last_end = 0

        # finditer iterates through the file, block by block
        for m in block_pattern.finditer(text):
            # 1. Keep any spacing/text that exists BEFORE this block
            new_text.append(text[last_end:m.start()])
            
            # 2. Keep the block itself ONLY if it's not in the deletion list
            if m.group(1) not in start_markers:
                new_text.append(m.group(0))
                
            last_end = m.end()

        # 3. Append anything left at the very end of the file
        new_text.append(text[last_end:])

        with open(filename, "w") as f:
            f.write("".join(new_text))

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")

# -----------------------------------------------------------------------------------------------------------------

def main_delete(DATA_PATH, system):
    MASTER_LOG_FILENAME = "logfile.txt"

    master_dir = DATA_PATH
    master_log_path = DATA_PATH / MASTER_LOG_FILENAME
    file_list = []
    
    if not os.path.isdir(master_dir):
        print(f"Error: The provided path is not a valid directory: {master_dir}")
        return file_list
    
    if not master_log_path.exists():
        print(f"⚠️ Warning: Master log file not found at \n'{master_log_path}'. \nSkipping deletion step to protect pre-existing files.", flush=True)
        return

    for filename in os.listdir(master_dir):
        if filename.endswith(".csv"):
            base_name, _ = os.path.splitext(filename)
            file_list.append(base_name)

    print(f"📖 Reading master log file in main_delete(): '{master_log_path}'")
    existing_entries = parse_master_log(master_log_path, system)
    print(f"Found {len(existing_entries)} existing entries.")

    file_set = set(file_list)
    entry_set = set(existing_entries.keys())

    print('file_list = ', file_list)
    print('entries = ', entry_set)

    entries_to_delete = entry_set - file_set 
    if entries_to_delete:
        # delete_contents_from_log(master_log_path, entries_to_delete, '#*#*_')
        delete_contents_from_log(master_log_path, entries_to_delete)
        for number in entries_to_delete:
            print(f"🚫 Successfully deleted '{number}' from logfile.")

    files_to_delete = file_set - entry_set
    for filename in files_to_delete:
        os.remove(clean_path(f'{master_dir}/{filename}.csv'))
        print(f"🚫 Successfully deleted '{filename}' from '{master_dir}'.")

    if not entries_to_delete and not files_to_delete:
        print("✅ Everything is perfectly synced. Nothing was deleted.")

# -----------------------------------------------------------------------------------------------------------------

def main_copy(TEMP_PATH, DATA_PATH, system):
    PARAMS_FILENAME = "parameters.txt"
    DATA_FILENAME = "data.csv"
    MASTER_LOG_FILENAME = "logfile.txt"

    source_params_path = TEMP_PATH / PARAMS_FILENAME
    source_data_path = TEMP_PATH / DATA_FILENAME
    master_log_path = DATA_PATH / MASTER_LOG_FILENAME
    
    if not source_params_path.exists() or not source_data_path.exists():
        print(f"❌ Error: Make sure '{source_params_path}' and '{source_data_path}' exist in folder '{TEMP_PATH}'.")
        return

    DATA_PATH.mkdir(exist_ok=True)
        
    with open(source_params_path, 'r') as f:
        new_params_content = f.read().strip()

    first_line = new_params_content.split('\n')[0].strip()
    if first_line != system:
        print(first_line)
        raise ValueError(f"It is not the \n**{system}** \nin the first line of \n**{source_params_path}** \nPlease check the file content.")

    print(f"📖 Reading master log file in main_copy(): {master_log_path}")
    existing_entries = parse_master_log(master_log_path, system)
    print(f"Found {len(existing_entries)} existing entries.")

    match_found = False
    for number, existing_params in existing_entries.items():
        if new_params_content == existing_params:
            print(f"🚫 Copy skipped! The parameters from source file '{source_data_path}' already exist under ID: {number}")
            match_found = True
            break
            
    if not match_found:
        print("✨ New parameter set detected.")
        
        existing_numbers = set(existing_entries.keys())
        new_number = None
        while True:
            candidate_number = str(random.randint(10000, 99999))
            if candidate_number not in existing_numbers:
                new_number = candidate_number
                break
        
        print(f"⭐ Generated new unique ID: {new_number}")
        
        dest_data_path = DATA_PATH / f"{new_number}.csv"
        
        shutil.copy(source_data_path, dest_data_path)
        print(f"✅ Copied '{source_data_path}' to '{dest_data_path}'")
        
        new_log_entry = (
            f"************************#*#*_{new_number}_#*#*************************\n"
            f"{new_params_content}\n"
            f"{'###############################################################'}\n"
        )
        
        try:
            with open(master_log_path, 'r+') as f:
                content = f.read()
                cleaned_content = content.rstrip('\n')
                f.seek(0)
                f.truncate()
                if cleaned_content:
                    f.write(cleaned_content + '\n' + new_log_entry)
                else:
                    f.write(new_log_entry)
        except FileNotFoundError:
            with open(master_log_path, 'w') as f:
                f.write(new_log_entry)
        
        print(f"✍️ Successfully appended new entry to '{master_log_path}'.")

# -----------------------------------------------------------------------------------------------------------------

def sync_simulation_data(temp_path, dest_data_path, system_name, prompt_user=True):
    """
    Wrapper function to sync simulation directories from a temp folder to a master data folder.
    Cleans paths automatically, prompts for confirmation, executes cleanup, and copies new data.
    """
    BULK_TEMP_PATH = clean_path(temp_path)
    DATA_PATH = clean_path(dest_data_path)
    
    print(f"----------- TEMP_PATH is {BULK_TEMP_PATH} -----------\n----------- Current DATA_PATH is {DATA_PATH} -----------", flush=True)


    if prompt_user:
        checked = input("Have you checked DATA_PATH and BULK_TEMP_PATH? Type yes to continue: ")
        if checked != "yes":
            raise ValueError("Please check DATA_PATH and BULK_TEMP_PATH before running the script.")

    main_delete(DATA_PATH, system_name)

    directories_file = BULK_TEMP_PATH / 'directories.txt'
    if directories_file.exists():
        with open(directories_file, 'r') as f:
            for line in f:
                serial_number, directory_name = line.strip().split('. ', 1)
                print(f'serial number : {serial_number}')
                main_copy(BULK_TEMP_PATH / directory_name, DATA_PATH, system_name)
                print('*********************************************')
    else:
        print(f"❓ Note: '{directories_file}' not found in '{BULK_TEMP_PATH}'. No new folders to copy.")

    print('*********************END***********************')


# ****************************************************************************************
# EXAMPLE USAGE
# ****************************************************************************************
if __name__ == "__main__":
    # Define your paths
    p1 = clean_path("folder1")
    p2 = clean_path("folder2")
    p3 = clean_path("folder3")
    out = clean_path("output_folder")

    # Define your GENERAL parameters and results here
    # You can change these lists based on user input
    user_defined_params = ['p1', 'p2', 'p3', 'p4', 'p5']
    user_defined_results = ['r1', 'r2', 'r3', 'r4', 'r5']
    
    # Or for your specific scientific case:
    # user_defined_params = ['t_sigma', 'mu', 'Temperature', 'V']
    # user_defined_results = ['Delta11', 'Delta10', 'Delta01', 'Delta00']

    # Call the function passing the lists
    merge_and_sort_csv(p1, p2, out, user_defined_params, user_defined_results)

    file_list = [p1, p2, p3]
    merge_and_minimize_csv(file_list, out, user_defined_params, user_defined_results, target_col='r5')
