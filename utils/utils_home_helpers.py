
def check_home_progress(progress_dict):
    """
    Checks overall progress across all home levels.
    Returns a completion percentage and list of completed levels.
    """
    total_levels = len(progress_dict)
    completed = [k for k, v in progress_dict.items() if v]
    percent_complete = int((len(completed) / total_levels) * 100)
    return percent_complete, completed
