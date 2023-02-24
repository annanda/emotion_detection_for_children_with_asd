import glob
import pathlib
import os
import csv

from emotion_detection_system.conf import emotion_detection_system_folder

results_path = os.path.join(emotion_detection_system_folder, 'results', '20-02-23')

csv_path = os.path.join(results_path, 'baselines_results.csv')


def get_acc_from_line(line):
    if 'Accuracy: ' in line and 'Balanced' not in line:
        return float(line.split('Accuracy: ')[-1])


def get_b_acc_from_line(line):
    if 'Balanced Accuracy: ' in line:
        return float(line.split('Balanced Accuracy: ')[-1])


def get_colour_data(line, colour):
    if colour in line and not check_is_last_line_on_report(line):
        after_colour_str = line.split(colour)[1]
        data_str = after_colour_str.split('     ')[1]
        precision, recall, f1 = [float(s) for s in data_str.split('    ')]
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def check_is_last_line_on_report(line):
    return 'Confusion matrix:' in line


def get_information_from_file(file_path):
    has_read_results = False
    has_read_acc = False
    has_read_b_acc = False
    has_passed_all_class_report_lines = False
    acc = None
    b_acc = None
    colour_list = ['blue', 'green', 'red', 'yellow']
    colours_dict = {
        colour: {} for colour in colour_list
    }
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if '... RESULTS ...' in line:
                has_read_results = True
            elif has_read_results:
                if not has_read_acc:
                    acc_data = get_acc_from_line(line)
                    if acc_data is not None:
                        acc = acc_data
                if not has_read_b_acc:
                    b_acc_data = get_b_acc_from_line(line)
                    if b_acc_data is not None:
                        b_acc = b_acc_data

                for colour in ['blue', 'green', 'red', 'yellow']:
                    colour_data = colours_dict[colour]
                    if len(colour_data.keys()) == 0:
                        colour_dict = get_colour_data(line, colour)
                        if colour_dict is not None:
                            colours_dict[colour] = colour_dict

                if not has_passed_all_class_report_lines:
                    if check_is_last_line_on_report(line):
                        has_passed_all_class_report_lines = True
                        break
            else:
                continue
    return {
        'acc': acc,
        'b_acc': b_acc,
        'colours': colours_dict
    }


def write_rows_in_csv(writer):
    files = glob.glob(os.path.join(results_path, "*", "*.txt"))
    for file_path in files[:]:
        file_name = pathlib.Path(file_path).name
        data_included_slug = file_name.split("_20-02-23.txt")[0]
        scenario = get_scenario(data_included_slug)
        annotation_type = pathlib.Path(file_path).parent.name
        file_data = get_information_from_file(file_path)
        row = create_order_row(data_included_slug, scenario, annotation_type, file_data)
        writer.writerow(row)


def create_order_row(data_included_slug, scenario, annotation_type, file_data):
    precision_blue = file_data.get('colours', {}).get('blue', {}).get('precision', None)
    precision_green = file_data.get('colours', {}).get('green', {}).get('precision', None)
    precision_red = file_data.get('colours', {}).get('red', {}).get('precision', None)
    precision_yellow = file_data.get('colours', {}).get('yellow', {}).get('precision', None)

    recall_blue = file_data.get('colours', {}).get('blue', {}).get('recall', None)
    recall_green = file_data.get('colours', {}).get('green', {}).get('recall', None)
    recall_red = file_data.get('colours', {}).get('red', {}).get('recall', None)
    recall_yellow = file_data.get('colours', {}).get('yellow', {}).get('recall', None)

    f1score_blue = file_data.get('colours', {}).get('blue', {}).get('f1', None)
    f1score_green = file_data.get('colours', {}).get('green', {}).get('f1', None)
    f1score_red = file_data.get('colours', {}).get('red', {}).get('f1', None)
    f1score_yellow = file_data.get('colours', {}).get('yellow', {}).get('f1', None)

    row = [data_included_slug, scenario, annotation_type, file_data['acc'], file_data['b_acc'],
           precision_blue, precision_green, precision_red, precision_yellow, recall_blue, recall_green, recall_red,
           recall_yellow, f1score_blue, f1score_green, f1score_red, f1score_yellow]
    return row


def get_scenario(data_included_slug):
    scenario = None
    if 'all_data' in data_included_slug:
        scenario = data_included_slug.split('all_data_')[-1]
    elif 'participant' in data_included_slug:
        scenario_list = data_included_slug.split('_')[2:]
        scenario = '_'.join(scenario_list)
    elif 'session' in data_included_slug:
        scenario_list = data_included_slug.split('_')[3:]
        scenario = '_'.join(scenario_list)
    return scenario


def write_in_csv():
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        header_csv = ['Data_Included_Slug', 'Scenario', 'Annotation_Type', 'Accuracy', 'Accuracy_Balanced',
                      'Precision_Blue', 'Precision_Green', 'Precision_Red', 'Precision_Yellow', 'Recall_Blue',
                      'Recall_Green', 'Recall_Red', 'Recall_Yellow', 'F1score_Blue',
                      'F1score_Green', 'F1score_Red', 'F1score_Yellow']

        writer.writerow(header_csv)
        write_rows_in_csv(writer)


if __name__ == '__main__':
    write_in_csv()
