import os
import pandas as pd

# Path to the env+GPS data file
participant_data_path = r'C:\Users\xy2593\Desktop\CUSP-GX-9000 Guided Study\Data\ParticipantData'
env_gps_file = os.path.join(participant_data_path, 
                           'All_Participant_Process', 
                           'All_Participant_Labeld_XGB_with_EnvGPS.csv')

# Check if env+GPS file exists
if not os.path.exists(env_gps_file):
    print(f"Error: Could not find environmental data file {env_gps_file}")
else:
    # Read the env+GPS data
    print(f"Reading environmental and GPS data from {env_gps_file}")
    env_data = pd.read_csv(env_gps_file)
    
    # Print column names
    print("\nColumns in the environmental data file:")
    print(env_data.columns.tolist())
    
    # Check for participant column variations
    possible_participant_cols = [col for col in env_data.columns if 'particip' in col.lower()]
    if possible_participant_cols:
        print("\nPossible participant columns found:")
        for col in possible_participant_cols:
            print(f"- {col}")
            # Print unique values
            unique_values = env_data[col].unique()
            print(f"  Unique values (first 10): {unique_values[:10]}")
    else:
        print("\nNo columns with 'participant' in the name found")
    
    # If no participant column found, check the first few rows
    print("\nFirst 5 rows of the data:")
    print(env_data.head()) 