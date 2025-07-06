import csv
import pandas as pd
from scipy.stats import chi2_contingency

#######################################
##### Read in CSV (I used J Code) #####
#######################################

def remove_second_row_csv(input_path, output_path, custom_headers):
    # Read in CSV and remove the second row
    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if i != 1:  # Skip second row
                writer.writerow(row)

    # Modify CSV with Custom Headers
    if custom_headers:
        df = pd.read_csv(output_path, names=custom_headers, header=0)
    else:
        df = pd.read_csv(output_path)

    print(df.head())
    return df

#########################
##### Explore RQ #1 #####
#########################

def accuracy_percentage_by_dimensionality(df):
    # Ensure Dimensionality is numeric
    df['Dimensionality'] = pd.to_numeric(df['Dimensionality'], errors='coerce')

    # Filter for just 2D and 3D
    filtered = df[df['Dimensionality'].isin([2, 3])].copy()

    # Define keywords that indicate accuracy-related outcomes
    accuracy_keywords = ['accuracy', 'accurate', 'error', 'errors', 'mistake', 'mistakes', 'correct', 'correctly', 'pick', 'picking', 'performance']

    # Check if any of the keywords are in 'Takeaways' or 'Measures'
    filtered['MentionsAccuracy'] = filtered.apply(
        lambda row: any(k in str(row['Takeaways']).lower() for k in accuracy_keywords) or
                    any(k in str(row['Measures']).lower() for k in accuracy_keywords),
        axis=1
    )

    # Group by Dimensionality and MentionsAccuracy, then count
    grouped = filtered.groupby(['Dimensionality', 'MentionsAccuracy']).size().unstack(fill_value=0)

    # Add percentage column
    grouped['Total'] = grouped.sum(axis=1)
    grouped['Percent_Accuracy'] = (grouped[True] / grouped['Total'] * 100).round(1)

    print("\n=== Accuracy Mentions by Dimensionality (% and count) ===")
    print(grouped[['Total', True, 'Percent_Accuracy']].rename(columns={True: 'MentionsAccuracy_True'}))

    return grouped, filtered

#######################
##### RQ #1 Stats #####
#######################

def run_chi_square_test(df, group_col, binary_col, positive_value=True, verbose=True):
    contingency = pd.crosstab(df[group_col], df[binary_col])
    # Check that both groups (2.0 and 3.0) exist
    print("\nContingency table passed to chi2_contingency():")
    print(contingency)

    chi2, p, dof, expected = chi2_contingency(contingency.values)

    if verbose:
        print("\n=== Chi-Square Test ===")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"Degrees of Freedom: {dof}")
        print(f"P-value: {p:.4f}")
        print("\nObserved Frequencies:")
        print(contingency)
        print("\nExpected Frequencies:")
        print(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))

    return {
        'chi2': chi2,
        'p': p,
        'dof': dof,
        'observed': contingency,
        'expected': expected
    }

def test_read_csv_examples():
    # Read CSV

    print('\n=======================================')
    print('READ CSV\n')

    custom_headers = ['Student Code', 'Paper', 'Cues', 'DOI', 'Modality_1', 'Modality_2', 'Modality_3', 'Representation_1', 'Representation_2', 'Representation_3', 'Representation_4', 'Dimensionality', 'Reference_Frame_1', 'Reference_Frame_2', 'Reference_Frame_3', 'Trigger', 'Converyed_Info_Direction', 'Conveyed_Info_Position', 'Conveyed_Info_Depth', 'Conveyed_Info_Transition', 'Conveyed_Info_Alternatives', 'Purpose_1', 'Purpose_2', 'Phases', 'Markedness', 'Domain', 'Strongest_Response', 'Measures', 'Takeaways', '#_Of_Participants', 'Extra', '', 'x']
    df = remove_second_row_csv('Coding Sheet - J Code.csv', 'cleaned_file.csv', custom_headers=custom_headers)

    # RQ #1

    print('\n=======================================')
    print('START RQ #1')

    grouped, filtered = accuracy_percentage_by_dimensionality(df)

    print("\nUnique dimensionality values in filtered:")
    print(filtered['Dimensionality'].unique())

    print("\nContingency table preview:")
    print(pd.crosstab(filtered['Dimensionality'], filtered['MentionsAccuracy']))

    run_chi_square_test(filtered, group_col='Dimensionality', binary_col='MentionsAccuracy')

    # RQ #2
    print('\n=======================================')
    print('START RQ #2')

if __name__ == "__main__":
    test_read_csv_examples()