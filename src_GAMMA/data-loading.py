#%%
import pandas as pd 

#%%
# Load the original demographics survey data with healthCode and professional-diagnosis columns and remove rows with NA values in professional-diagnosis
og_df = pd.read_csv('src_GAMMA/data/data_paired/csv/Demographics_Survey.csv')

og_df_copy = og_df.copy()

columns_to_keep = ['healthCode', 'professional-diagnosis']
filtered_df = og_df_copy[columns_to_keep]


na_count = filtered_df["professional-diagnosis"].isna().sum()
print(f"Number of NA values in professional-diagnosis: {na_count}")
filtered_df = filtered_df.dropna(subset=["professional-diagnosis"]).copy()

filtered_df["professional-diagnosis"] = filtered_df["professional-diagnosis"].astype(int)

print(f"After filtering: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")

label_counts = filtered_df["professional-diagnosis"].value_counts()
print("\nNumber of samples per class:")
print(label_counts)              # raw counts
print("\nPercentage distribution:")
print(label_counts / len(filtered_df) * 100)  # percentage

filtered_df.to_csv('src_GAMMA/data/data_paired/csv/Demographics_Survey_filtered.csv', index=False)

#%%