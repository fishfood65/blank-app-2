
import pytest
from datetime import date, timedelta
from utils.input_tracker import daterange, get_filtered_dates

def test_daterange_yields_all_dates():
    start = date(2024, 1, 1)
    end = date(2024, 1, 5)
    expected = [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5)
    ]
    result = list(daterange(start, end))
    assert result == expected

def test_get_filtered_dates_all_days():
    start = date(2024, 1, 1)
    end = date(2024, 1, 7)
    result = get_filtered_dates(start, end, "All Days")
    assert len(result) == 7
    assert result[0] == start and result[-1] == end

def test_get_filtered_dates_weekdays_only():
    start = date(2024, 1, 1)  # Monday
    end = date(2024, 1, 7)    # Sunday
    result = get_filtered_dates(start, end, "Weekdays Only")
    weekdays = [d for d in result if d.weekday() < 5]
    assert result == weekdays
    assert len(result) == 5  # Mon-Fri

def test_get_filtered_dates_weekend_only():
    start = date(2024, 1, 1)  # Monday
    end = date(2024, 1, 7)    # Sunday
    result = get_filtered_dates(start, end, "Weekend Only")
    weekends = [d for d in result if d.weekday() >= 5]
    assert result == weekends
    assert len(result) == 2  # Sat-Sun
