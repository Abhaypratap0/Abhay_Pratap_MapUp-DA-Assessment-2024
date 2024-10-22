from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code here.
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i + n]
        for j in range(len(group) - 1, -1, -1):
            result.append(group[j])
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    length_dict = {}
    for word in lst:
        length = len(word)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(word)
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def f_dict(d, parent_key='', sep='.'):
      items = []
      
      for k, v in d.items():
          
          new_key = f"{parent_key}{sep}{k}" if parent_key else k
          
          if isinstance(v, dict):
              items.extend(f_dict(v, new_key, sep=sep).items())
          
          elif isinstance(v, list):
              for i, item in enumerate(v):
                  list_key = f"{new_key}[{i}]"
                  if isinstance(item, dict):
                      items.extend(f_dict(item, list_key, sep=sep).items())
                  else:
                      items.append((list_key, item))
          
          else:
              items.append((new_key, v))
      
      return dict(items)
    return f_dict(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
            
    nums.sort()
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    return result


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Your code here
    dates = []
    
    i = 0
    while i < len(text):
        if i + 10 <= len(text) and text[i:i+2].isdigit() and text[i+2] == '-' and text[i+5] == '-' and text[i+6:i+10].isdigit():
            dates.append(text[i:i+10])
            i += 10
        elif i + 10 <= len(text) and text[i:i+2].isdigit() and text[i+2] == '/' and text[i+5] == '/' and text[i+6:i+10].isdigit():
            dates.append(text[i:i+10])
            i += 10
        elif i + 10 <= len(text) and text[i:i+4].isdigit() and text[i+4] == '.' and text[i+7] == '.' and text[i+8:i+10].isdigit():
            dates.append(text[i:i+10])
            i += 10
        else:
            i += 1

    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Your code here
    import polyline 
    from math import radians, sin, cos, sqrt, atan2
    
    # Haversine formula
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = radians(lat1), radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lon2 - lon1)
        
        a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c 
    coordinates = polyline.decode(polyline_str)
    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    distances = [0] 
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i - 1]
        lat2, lon2 = coordinates[i]
        dist = haversine(lat1, lon1, lat2, lon2)
        distances.append(dist)
    df['distance'] = distances
    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]  
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  
            final_matrix[i][j] = row_sum + col_sum - 2*rotated_matrix[i][j] 
    
    return final_matrix

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    grouped = df.groupby(['id', 'id_2'])
    
    incomplete_flags = []

    for (id_val, id2_val), group in grouped:
        covered_days = {day: [] for day in days_of_week} 
        
        for _, row in group.iterrows():
            start_day = row['startDay']
            start_time = time_to_seconds(row['startTime'])
            end_day = row['endDay']
            end_time = time_to_seconds(row['endTime'])
            
            start_index = days_of_week.index(start_day)
            end_index = days_of_week.index(end_day)
            
            if start_index == end_index:  
                covered_days[start_day].append((start_time, end_time))
            else:
                covered_days[start_day].append((start_time, 86399))
                
                for i in range(start_index + 1, end_index):
                    covered_days[days_of_week[i]].append((0, 86399))
                
                covered_days[end_day].append((0, end_time))
        
        full_24_hour_coverage = True
        for day, time_ranges in covered_days.items():
            if not time_ranges:
                full_24_hour_coverage = False
                break
            
            time_ranges.sort()
            current_end = 0
            
            for start, end in time_ranges:
                if start > current_end:  
                    full_24_hour_coverage = False
                    break
                current_end = max(current_end, end)
            
            if current_end < 86399:  
                full_24_hour_coverage = False
                break
        
        has_full_week_coverage = all(covered_days[day] for day in days_of_week)
        is_complete = has_full_week_coverage and full_24_hour_coverage
        incomplete_flags.append((id_val, id2_val, not is_complete))
    
    result_df = pd.DataFrame(incomplete_flags, columns=['id', 'id_2', 'incomplete'])
    result_df.set_index(['id', 'id_2'], inplace=True)
    
    return result_df['incomplete']
