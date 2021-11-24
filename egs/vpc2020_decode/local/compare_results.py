# -*- coding: utf8 -*-
from collections import OrderedDict
from tabulate import tabulate
import csv
from argparse import ArgumentParser

def main():
    """
    Parameters to modify for customizing processing
    """

    parser = ArgumentParser()
    parser.add_argument('--output_filepath', type=str, default="compare_res", help="Path for output files. Don't include extension as multiple files will be created")
    args = parser.parse_args()

    # Dict of result files to process.
    # Key is test name and value is a dict with two keys :
    #   - "display" : Title of the test. It will be displayed on final table
    #   - "file_path" : path to the result file to process. Path is absolute or relative to the working directory
    # First file in this dict is the reference for value comparison between tests (variation percentage displayed is variation compared to first file result)
    results_path_dict = OrderedDict(
        # VPC2020_libri={"display": "VPC 2020 with Kaldi\nTrain : Libri-train-clean-360_anon", "file_path": "results_vpc2020_anon.txt"},
        # sidekit_anon_scratch_aug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360_anon_spk - Augmentation", "file_path": "results_anon_spk_scratch_aug.txt"},
        # sidekit_anon_scratch_noaug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360_anon_spk - No augmentation", "file_path": "results_anon_spk_scratch_no_aug.txt"},

        VPC2020_libri={"display": "VPC 2020 with Kaldi\nTrain : Libri-train-clean-360", "file_path": "results_vpc2020.txt"},
        #sidekit_vox_aug={"display": "VPC 2020 with Sidekit\nTrain : Voxceleb12 - Augmentation", "file_path": "results.txt"},
        sidekit_scratch_libri_aug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360 - Augmentation", "file_path": "results_scratch_aug.txt"},
        #sidekit_transfer_vox_libri_aug={"display": "VPC 2020 with Sidekit\nTrain : Voxceleb12, transfer Libri-train-clean-360 - Augmentation\nTraining duration : 14h54", "file_path": "results_transfer_aug.txt"},
        sidekit_scratch_libri_noaug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360 - No augmentation", "file_path": "results_scratch_noaug.txt"},
        #sidekit_transfer_vox_libri_noaug={"display": "VPC 2020 with Sidekit\nTrain : Voxceleb12, transfer Libri-train-clean-360 - No augmentation\nTraining duration : 6h47", "file_path": "results_transfer_noaug.txt"},
        #sidekit_anon_scratch_aug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360_anon - Augmentation", "file_path": "results_anon_spk_scratch_aug.txt"},
        #sidekit_anon_scratch_noaug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360_anon - No augmentation", "file_path": "results_anon_spk_scratch_no_aug.txt"},
        #sidekit_anon_transfer_aug={"display": "VPC 2020 with Sidekit\nTrain : Voxceleb12, transfer Libri-train-clean-360_anon - Augmentation", "file_path": "results_anon_spk_transfer_aug.txt"},
        #sidekit_anon_transfer_noaug={"display": "VPC 2020 with Sidekit\nTrain : Voxceleb12, transfer Libri-train-clean-360_anon - No augmentation", "file_path": "results_anon_spk_transfer_no_aug.txt"},
        sidekit_vad_scratch_aug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360 with VAD - Augmentation", "file_path": "results_vad_scratch_aug.txt"},
        sidekit_vad_scratch_noaug={"display": "VPC 2020 with Sidekit\nTrain : Libri-train-clean-360 with VAD - No augmentation", "file_path": "results_vad_scratch_no_aug.txt"},
        #sidekit_vad_transfer_aug={"display": "VPC 2020 with Sidekit\nTrain : Voxceleb12, transfer Libri-train-clean-360 with VAD - Augmentation", "file_path": "results_vad_transfer_aug.txt"},
        #sidekit_vad_transfer_noaug={"display": "VPC 2020 with Sidekit\nTrain : Voxceleb12, transfer Libri-train-clean-360 with VAD - No augmentation", "file_path": "results_vad_transfer_no_aug.txt"},

    )

    # Metrics to display on the final table.
    # Current accepted metrics (ATTENTION : case sensitive) : "EER", "linkability", "Cllr"
    metrics_to_display = ["EER"]

    # Out file path for compare results
    # Path without file extension. csv or txt extension will be added automatically
    out_filepath = args.output_filepath



    # Start processing result files
    results = OrderedDict() # Key = task, value = [[test name, EER, Cllr, Linkability]]
    display_tab_column_name = ["Dataset"]
    nb_res_file_processed = 0
    for test_name, test_data in results_path_dict.items():
        text_test_display = test_data["display"]
        file_path = test_data["file_path"]
        print("New result file : ", test_name, " - ", file_path)

        with open(file_path, "r") as result_file:
            skip_task = False
            while True:
                try:
                    cur_line = result_file.readline().strip()
                    if cur_line.strip() == "" or cur_line.startswith("ASR"):
                        # Stop iteration if a blank line is found or if ASR result part is reached
                        raise StopIteration
                    if skip_task:
                        # Skipping current task requested. Checking if a new task is reached.
                        # If yes, stop skipping and processing the task
                        # If not, go directly to next iteration to read next line
                        if cur_line.startswith("ASV"):
                            skip_task = False
                        else:
                            continue
                except StopIteration:
                    break # Stop loop as no more task are loaded

                # Line containing ASV => header line of the task
                if cur_line.startswith("ASV"):
                    # Extract cur_dataset name
                    head_line = cur_line[4:].strip()  # Remove ASV- or ASV:
                    split_head = head_line.split("-")
                    cur_dataset = split_head[0].strip() + " - " + split_head[1].strip()

                    # Checking if new dataset exists in first result file (reference).
                    # If current file is the first, ignore this checking as first file is used as reference
                    if nb_res_file_processed > 0 and cur_dataset not in results:
                        print("Current dataset not found in first result file : ", cur_dataset, ". Skipping current dataset")
                        # Read until next header
                        skip_task = True
                        # Go to next iteration for reading next line and skip the current task
                        continue

                    # Adding new task in result list if not already present in result list
                    results.setdefault(cur_dataset, {metric: [] for metric in metrics_to_display})

                # Retrieve EER result
                # Current line contains EER result and EER is a metric to keep
                if cur_line.startswith("EER") and "EER" in results[cur_dataset]:
                    eer_split = cur_line.split(":")
                    eer_val = eer_split[1].strip().replace("%", "")
                    # Calculate percentage difference with first result file. Only occurs if at least one file was processed
                    if nb_res_file_processed > 0:
                        first_val = results[cur_dataset]["EER"][0].replace("%", "").strip()
                        # Don't calculate difference if first result is not floating number => cast first_val to float will raise ValueError exception
                        try:
                            diff_percent = ((float(eer_val) - float(first_val)) * 100) / float(first_val)
                            diff_percent_display = "({:+.2f} %)".format(diff_percent)
                        except ValueError:
                            diff_percent_display = "(NaN%)"
                    else:
                        diff_percent_display = ""
                    eer_val_display = eer_val + " % " + diff_percent_display
                    results[cur_dataset]["EER"].append(eer_val_display)

                # Retrieve Cllr result
                # Current line contains Cllr result and Cllr is a metric to keep
                if cur_line.startswith("Cllr") and "Cllr" in results[cur_dataset]:
                    cllr_split = cur_line.split(":")
                    cllr_val = cllr_split[1].strip().replace(" ", "/")
                    cllr_val_display = cllr_val
                    results[cur_dataset]["Cllr"].append(cllr_val_display)

                # Retrieve Linkability result
                # Current line contains Linkability result and Linkability is a metric to keep
                if cur_line.startswith("linkability") and "linkability" in results[cur_dataset]:
                    linkability_split = cur_line.split(":")
                    linkability_val = linkability_split[1].strip()
                    results[cur_dataset]["linkability"].append(linkability_val)

        nb_res_file_processed += 1
        results = control_result_list(results, nb_res_file_processed)
        display_tab_column_name.append(text_test_display)

    tab_rows = []
    for cur_dataset, data in results.items():
        for res_name, res_data in data.items():
            if res_name in metrics_to_display:
                new_row = [cur_dataset + " - " + res_name]
                new_row.extend(data[res_name])
                tab_rows.append(new_row)

    # Write results in csv format
    csv_file_results = open(out_filepath + ".csv", "w", newline="")
    csv_writer = csv.writer(csv_file_results, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL)
    # Write header
    csv_writer.writerow([column.replace("\n", " - ") for column in display_tab_column_name])
    for row in tab_rows:
        csv_writer.writerow([itm for itm in row])
    csv_file_results.close()

    # Write results in text format
    compare_res_table = tabulate(tab_rows, headers=display_tab_column_name, tablefmt="orgtbl")
    print(compare_res_table)
    out_file = open(out_filepath + ".txt", "w")
    out_file.write(compare_res_table)
    out_file.close()

def control_result_list(result_list, nb_test_done):
    """
    Control if a result is missing in result_list. Useful if a dataset is not found in current file but found in result_list
    If a result is missing, adding "N/A" to the result list.

    Parameters
    ----------
    result_list: Dict of results already processed
    nb_test_done : Current number of tests done at this point

    Returns
    -------
    The result_list with "N/A" values if a test is missing
    """
    for dataset, data in result_list.items():
        for metricType, metricList in data.items():
            if len(metricList) < nb_test_done:
                result_list[dataset][metricType].append("N/A")

    return result_list



if __name__ == "__main__":
    main()
