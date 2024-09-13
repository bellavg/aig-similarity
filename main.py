import argparse
import os

import pandas as pd
from aigverse import read_aiger_into_aig
from spectral import get_lap_spectral_dist, get_adj_spectral_dist
from distances import get_net_simile, get_deltacon0, get_ved, get_veo

# Map function names to actual function calls
FUNCTION_MAP = {
    "deltacon0": get_deltacon0,
    "netsimile":  get_net_simile,
    "lap_sd": get_lap_spectral_dist,
    "adj_sd": get_adj_spectral_dist,

    "veo": get_veo,
    "ved": get_ved,

}

AIG_TYPES = ['bdd', 'collapse', 'dsd', 'espresso', 'lut_bidec', 'sop', 'strash', 'default']


# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process two AIG types, a folder path, and a function.")
    # Adding arguments
    # Adding a list of AIG types as a single argument with at least 2 required
    parser.add_argument(
        "aig_types",
        type=str,
        choices=AIG_TYPES,
        help="List of AIG types (choose from predefined list)",
        nargs='*',
        default='default'
    )
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the AIG files",
                        nargs="?", default="data/aigs/")
    parser.add_argument("optimized_path", type=str, help="Path to the folder containing the optimized AIG files",
                        nargs="?", default="data/optimized/")
    parser.add_argument("save_path", type=str, help="Path to the folder where results should be saved",
                        nargs="?", default="data/results/")
    parser.add_argument("id_path", type=str, help="Path to the txt file with aig_ids to be used",
                        nargs="?", default="data/aigs/indices.txt")
    parser.add_argument("metric", type=str, choices=FUNCTION_MAP.keys(), help="Metric to apply")
    return parser.parse_args()


def get_results(args, aig_ids):
    if args.aig_types == 'default':
        args.aig_types = AIG_TYPES[:-1]
    # Dictionary to store results for each AIG type comparison
    results_dict = {f"{aig_type1},{aig_type2}": []
                    for i, aig_type1 in enumerate(args.aig_types)
                    for aig_type2 in args.aig_types[i + 1:]}

    # Retrieve the comparison function based on the metric provided by the user
    comparison_function = FUNCTION_MAP[args.metric]

    for filename in aig_ids:
        # Get paths to each AIG file for the given filename across different types
        aig_files = {aig_type: os.path.join(args.folder_path, aig_type, filename+".aig")
                     for aig_type in args.aig_types}

        # Compare each pair of AIG types
        for i, aig_type1 in enumerate(args.aig_types):
            for aig_type2 in args.aig_types[i + 1:]:
                # Read AIGER files into AIG networks
                aig1 = read_aiger_into_aig(aig_files[aig_type1])
                aig2 = read_aiger_into_aig(aig_files[aig_type2])
                comparison_result = comparison_function(aig1, aig2)

                # Save the comparison result in the dictionary
                comparison_key = f"{aig_type1},{aig_type2}"
                results_dict[comparison_key].append(comparison_result)
        print(f"AIG benchmark {filename} comparisons complete")

    return results_dict

#TODO test 2 of one kind new csv (DONE)
# test multiple aigtypes (more than 2),
# test existing csv and adding on,
# think about more than one metric at a time

def main():
    # Parse the arguments
    args = parse_arguments()
    # Open the file and read all lines into a list
    with open(args.id_path, 'r') as file:
        lines = file.readlines()
    # Remove newline characters if necessary
    aig_ids = sorted([line.strip() for line in lines])

    # check if results csv file already exists
    result_csv_path = os.path.join(args.save_path, f'{args.metric}_scores.csv')
    # Check if the file exists
    if os.path.exists(result_csv_path):
        # Reading a CSV file into a DataFrame
        results_df = pd.read_csv(result_csv_path)
    else:
        id_data = {"aig_ids": aig_ids}
        results_df = pd.DataFrame(id_data)

    aig_results = get_results(args, aig_ids)

    # Add the results to the DataFrame
    for key, results in aig_results.items():
        results_df[key] = results

    # Save the updated DataFrame to CSV
    results_df.to_csv(result_csv_path, index=False)


if __name__ == "__main__":
    main()