import pandas as pd
import numpy as np


def convert_notion_csv_to_anki(notion_csv_path, output_csv_path):
	# Remove headers
	df = pd.read_csv(notion_csv_path)
	column_names = df.columns

	# Remove white spaces
	df = df.dropna(how='all')

	# Remove remainig nan values (white spaces) with empty strings
	df = df.replace(np.nan, '', regex=True)

	# If the meaning of the word is not given then there is no point in doing this exercise, so we 
	# will ignore them as now
	df = df[df["Meaning"] != ""]

	# Keep unique sets of words and their POS in the table
	# (This is very inefficent but the aim was to quickly do it and because it's not a ercurring
	# this and the data will not increase to a huge number it should be fine)
	all_words = list(set(df['Word'].tolist()))
	clean_data = []
	for word in all_words:
		subset_df = df[df['Word'] == word]
		if subset_df.shape[0] == 1:
			clean_data.append(subset_df.iloc[0].tolist())
			continue
		if subset_df['POS'].unique().shape[0] == 1:
			clean_data.append(subset_df.iloc[0].tolist())
			continue
		unique_pos = list(subset_df['POS'].unique())
		for pos in unique_pos:
			subset_df_wrt_pos = subset_df[subset_df["POS"] == pos]
			subset_list_data = subset_df_wrt_pos.iloc[0].tolist()
			subset_list_data[0] = word + "-" + pos
			clean_data.append(subset_list_data)
	df = pd.DataFrame(clean_data)
	df.columns = column_names

	# Randomle shuffle it
	df = df.sample(frac = 1)

	# Save it without headers and index
	df.to_csv(output_csv_path, index=False, header=None)


if __name__ == "__main__":
	convert_notion_csv_to_anki("/Users/neeraj/Downloads/tmp.csv", "/Users/neeraj/Downloads/tmp_cvt.csv")