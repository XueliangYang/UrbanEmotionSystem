# Environmental Analysis Scripts

This directory contains scripts for analyzing environmental data collected during the Brooklyn study.

## Scripts

- **analyze_env_features.py**: Analyzes the relationship between environmental features and emotional states. It performs correlation analysis, random forest modeling, and creates visualizations like boxplots and violin plots for each environmental feature.

- **check_env_data.py**: Utility script to check the environmental data format, column names, and participant information.

- **match_labels_with_env_gps.py**: Matches emotion labels with environmental and GPS data based on timestamps.

## Output

The analysis results are stored in the `env_feature_analysis` folder in the `Usage_v6_Brooklyn` directory. These include:

- Feature importance plots
- Boxplots and violin plots for environmental variables
- ANOVA significance analysis
- Summary statistics

## Data Sources

The scripts use data from:
`C:\Users\xy2593\Desktop\CUSP-GX-9000 Guided Study\Data\ParticipantData\All_Participant_Process` 