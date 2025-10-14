def determine_run_folder(root_path, prefix='run_'):
    """
    Determine the next run folder number based on existing folders.
    
    Args:
        root_path (Path): Root path where runs are stored
        
    Returns:
        Path: Path to the new run folder
    """
    root_path.mkdir(exist_ok=True)
    
    # Find existing run folders
    existing_runs = [d for d in root_path.glob(f'{prefix}*') if d.is_dir()]
    run_numbers = []
    
    # Extract run numbers from folder names
    for run_dir in existing_runs:
        try:
            # Extract number after 'run_'
            run_number = int(run_dir.name.split('_')[1])
            run_numbers.append(run_number)
        except (IndexError, ValueError):
            # Skip folders that don't match the expected format
            print(f"Invalid run folder: {run_dir.name}")
            continue
    
    # Determine the next run number
    next_run_number = 1  # Default if no runs exist
    if run_numbers:
        next_run_number = max(run_numbers) + 1
    
    # Create the run folder path
    run_folder = root_path / f'run_{next_run_number}'
    run_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Using run folder: {run_folder}")
    return run_folder
