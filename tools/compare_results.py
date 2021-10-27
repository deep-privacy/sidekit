# -*- coding: utf8 -*-
from collections import OrderedDict
from tabulate import tabulate

def main():
    results_path_dict = OrderedDict(
        kaldi="vpc2020/results_kaldi.txt",
        sidekit_voxceleb="vpc2020/results.txt",
        sidekit_librispeech="vpc2020/results_finetune_model2.txt"
    )

    results = OrderedDict() # Key = task, value = [[test name, EER, Cllr, Linkability]]
    for test_name, file_path in results_path_dict.items():
        print("New result file : ", test_name, " - ", file_path)

        with open(file_path, "r") as result_file:
            while True:
                # Read header of result task. It can raise StopIteration if no more task exists
                try:
                    head_line = result_file.readline()
                    if head_line.strip() == "" or "ASR" in head_line:
                        # Stop iteration if a blank line is found or if ASR result part is reached
                        raise StopIteration
                except StopIteration:
                    break # Stop loop as no more task are loaded
                # Extract dataset name from header
                head_line = head_line[4:].strip()  # Remove ASV- or ASV:
                split_head = head_line.split("-")
                dataset = split_head[0].strip() + " - " + split_head[1].strip()

                results.setdefault(dataset, {"EER": [], "Cllr": [], "Linkability": []})

                # Read EER
                eer_line = result_file.readline()
                eer_split = eer_line.split(":")
                eer_val = eer_split[1].replace("%", "").strip()
                if len(results[dataset]["EER"]) > 0:
                    first_val = results[dataset]["EER"][0].replace("%", "").strip()
                    diff_percent = ((float(eer_val) - float(first_val)) * 100) / float(first_val)
                    diff_percent_display = " ({:+.2f} %)".format(diff_percent)
                else:
                    diff_percent_display = ""
                eer_val_display = eer_val + " %" + diff_percent_display
                results[dataset]["EER"].append(eer_val_display)

                # Read Cllr
                cllr_line = result_file.readline()
                cllr_split = cllr_line.split(":")
                cllr_val = cllr_split[1].replace(" ", "/").strip()
                cllr_val_display = cllr_val
                results[dataset]["Cllr"].append(cllr_val_display)

                # Read linkability
                linkability_line = result_file.readline()
                if "linkability" in linkability_line:
                    linkability_split = linkability_line.split(":")
                    linkability_val = linkability_split[1].strip()
                else:
                    linkability_val = "N/A"
                results[dataset]["Linkability"].append(linkability_val)

    tab_column_name = ["Dataset"]
    for res_path_key in results_path_dict.keys():
        tab_column_name.append(res_path_key)

    tab_rows = []
    display_list = ["EER"]
    for dataset, data in results.items():
        for res_name, res_data in data.items():
            if len(display_list) == 0 or (res_name in display_list):
                new_row = [dataset + " - " + res_name]
                new_row.extend(data[res_name])
                tab_rows.append(new_row)

    compare_res_table = tabulate(tab_rows,headers=tab_column_name, tablefmt="orgtbl")
    print(compare_res_table)
    out_file = open("compare_res.txt", "w")
    out_file.write(compare_res_table)
    out_file.close()




if __name__ == "__main__":
    main()
