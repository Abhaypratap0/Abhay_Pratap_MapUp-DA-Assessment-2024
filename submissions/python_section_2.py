import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    # Initialize the distance matrix with infinity
    dist_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    # Populate the distance matrix with distances from the DataFrame
    for _, row in df.iterrows():
        dist_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        dist_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    # Set the diagonal to zero (distance to itself)
    np.fill_diagonal(dist_matrix.values, 0)

    # Floyd-Warshall algorithm to calculate the shortest paths
    for intermediate in unique_ids:
        for start in unique_ids:
            for end in unique_ids:
                if dist_matrix.loc[start, end] > dist_matrix.loc[start, intermediate] + dist_matrix.loc[intermediate, end]:
                    dist_matrix.loc[start, end] = dist_matrix.loc[start, intermediate] + dist_matrix.loc[intermediate, end]

    return dist_matrix


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    # Convert the index and columns to coordinate pairs with melt
    unrolled = df.stack().reset_index()
    unrolled.columns = ['id_start', 'id_end', 'distance']

    # Filter out self-loops (id_start == id_end)
    unrolled = unrolled[unrolled['id_start'] != unrolled['id_end']]

    # Sort for better readability (optional)
    unrolled = unrolled.sort_values(by=['id_start', 'id_end']).reset_index(drop=True)

    return unrolled

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_distances = df[df['id_start'] == reference_id]

    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])  

    average_distance_reference = reference_distances['distance'].mean()

    lower_threshold = average_distance_reference * 0.9
    upper_threshold = average_distance_reference * 1.1

    average_distances = df.groupby('id_start', as_index=False)['distance'].mean()
    average_distances.columns = ['id_start', 'average_distance']


    filtered_ids = average_distances[(average_distances['average_distance'] >= lower_threshold) & 
                                     (average_distances['average_distance'] <= upper_threshold)]

    return filtered_ids.sort_values(by='id_start')


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates by multiplying the distance with the rate coefficients
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate

    return df

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define weekday and weekend lists
    weekdays_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends_list = ['Saturday', 'Sunday']

    # List to store new rows with time-based tolls
    expanded_rows = []

    # Loop through each row in the input DataFrame
    for _, toll_row in df.iterrows():
        start_id = toll_row['id_start']
        end_id = toll_row['id_end']
        route_distance = toll_row['distance']

        # Loop through weekdays with specific time intervals and discounts
        for weekday in weekdays_list:
            time_windows = [
                (time(0, 0), time(10, 0), 0.8),   # 00:00 to 10:00
                (time(10, 0), time(18, 0), 1.2),  # 10:00 to 18:00
                (time(18, 0), time(23, 59, 59), 0.8)  # 18:00 to 23:59:59
            ]

            for start_time, end_time, discount_factor in time_windows:
                # Create a new row with adjusted toll rates
                new_entry = {
                    'id_start': start_id,
                    'id_end': end_id,
                    'start_day': weekday,
                    'start_time': start_time,
                    'end_day': weekday,
                    'end_time': end_time,
                    'moto': toll_row['moto'] * discount_factor,
                    'car': toll_row['car'] * discount_factor,
                    'rv': toll_row['rv'] * discount_factor,
                    'bus': toll_row['bus'] * discount_factor,
                    'truck': toll_row['truck'] * discount_factor
                }
                expanded_rows.append(new_entry)

        # Loop through weekends with a constant discount factor
        for weekend in weekends_list:
            new_entry = {
                'id_start': start_id,
                'id_end': end_id,
                'start_day': weekend,
                'start_time': time(0, 0),  # 00:00
                'end_day': weekend,
                'end_time': time(23, 59, 59),  # 23:59:59
                'moto': toll_row['moto'] * 0.7,
                'car': toll_row['car'] * 0.7,
                'rv': toll_row['rv'] * 0.7,
                'bus': toll_row['bus'] * 0.7,
                'truck': toll_row['truck'] * 0.7
            }
            expanded_rows.append(new_entry)

    # Create a new DataFrame with the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df

  


# --------Main executions


df = pd.read_csv('/content/MapUp-DA-Assessment-2024/datasets/dataset-2.csv')

#Question 9

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(df)
# Print the resulting matrix
# print(distance_matrix)


#Question 10

# Unroll the distance matrix into id_start, id_end, and distance
unrolled_df = unroll_distance_matrix(distance_matrix)
# Print the resulting DataFrame
# print(unrolled_df.head(10))

#Question 11

reference_id = 1001400  
result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
# print(result_df)

#Question 12

# Calculate toll rates
toll_rate_df = calculate_toll_rate(unrolled_df)
# print(toll_rate_df.head(10))

#Question 13

time_df = calculate_time_based_toll_rates(toll_rate_df)
# print(time_df.head())