import glob
import pathlib
import os
import csv
import json

from emotion_detection_system.conf import emotion_detection_system_folder

results_path = os.path.join(emotion_detection_system_folder, 'json_results', 'data_experiments_nn_algorithm_300523')
batch_data_experiments = 'nn_algorithm_300523'
csv_path = os.path.join(results_path, f'{batch_data_experiments}_results.csv')


def write_rows_in_csv(writer):
    files = glob.glob(os.path.join(results_path, "*", "*.json"))
    for file_path in files[:]:
        with open(file_path, 'r') as json_file:
            data_file = json.load(json_file)
            row = create_order_row(data_file)
        writer.writerow(row)


def create_order_row(file_data):
    data_included_slug = file_data['Data_Included_Slug']
    participant_number = get_participant_number(file_data)
    session_number = get_session_number(file_data)
    scenario = file_data['Scenario']
    annotation_type = file_data['Annotation_Type']

    accuracy = format_print_value(file_data['Accuracy'])
    balanced_accuracy = format_print_value(file_data['Accuracy_Balanced'])

    precision_blue = format_print_value(file_data.get('Precision_blue', None))
    precision_green = format_print_value(file_data.get('Precision_green', None))
    precision_red = format_print_value(file_data.get('Precision_red', None))
    precision_yellow = format_print_value(file_data.get('Precision_yellow', None))

    recall_blue = format_print_value(file_data.get('Recall_blue', None))
    recall_green = format_print_value(file_data.get('Recall_green', None))
    recall_red = format_print_value(file_data.get('Recall_red', None))
    recall_yellow = format_print_value(file_data.get('Recall_yellow', None))

    f1score_blue = format_print_value(file_data.get('F1score_blue', None))
    f1score_green = format_print_value(file_data.get('F1score_green', None))
    f1score_red = format_print_value(file_data.get('F1score_red', None))
    f1score_yellow = format_print_value(file_data.get('F1score_yellow', None))

    row = [data_included_slug, participant_number, session_number, scenario, annotation_type,
           accuracy, balanced_accuracy,
           precision_blue, precision_green, precision_red, precision_yellow,
           recall_blue, recall_green, recall_red, recall_yellow,
           f1score_blue, f1score_green, f1score_red, f1score_yellow]
    return row


def format_print_value(value):
    if isinstance(value, float):
        return f"{value:.4}"
    else:
        return value


def get_participant_number(file_data):
    participant = None

    if 'all_data' in file_data['Data_Included_Slug']:
        participant = 'All data'
    elif 'participant' in file_data['Data_Included_Slug']:
        participant = f"Participant_0{file_data['Participant']}"

    return participant


def get_session_number(file_data):
    if 'session' in file_data['Data_Included_Slug']:
        session = f"Session_0{file_data['Participant']}_0{file_data['Session']}"
    else:
        session = None
    return session


def write_in_csv():
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        header_csv = ['Data_Included_Slug', 'Participant', 'Session', 'Scenario', 'Annotation_Type',
                      'Accuracy',
                      'Accuracy_Balanced',
                      'Precision_Blue', 'Precision_Green', 'Precision_Red', 'Precision_Yellow', 'Recall_Blue',
                      'Recall_Green', 'Recall_Red', 'Recall_Yellow', 'F1score_Blue',
                      'F1score_Green', 'F1score_Red', 'F1score_Yellow']

        writer.writerow(header_csv)
        write_rows_in_csv(writer)


if __name__ == '__main__':
    write_in_csv()
    print('done!')
