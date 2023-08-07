import pandas as pd
import os

from emotion_detection_system.conf import main_folder

# undersampling_data_results = pd.read_csv(
#     os.path.join(main_folder, 'emotion_detection_system', 'json_results', '280223',
#                  'undersampling_results.csv'))

# class_weight_data_results = pd.read_csv(
#     os.path.join(main_folder, 'emotion_detection_system', 'json_results', '060323',
#                  'class_weight_results.csv'))

oversampling_data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'data_experiments_oversampling_random_050423',
                 'oversampling_random_050423_results.csv'))

rfe_data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'data_experiments_rfe_030423',
                 'rfe_030423_results.csv'))

nn_oversampling_data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'data_experiments_nn_algorithm_300523',
                 'nn_algorithm_300523_results.csv'))

nn_bl_data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'data_experiments_nn_algorithm_bl_310723',
                 'nn_algorithm_bl_310723_results.csv'))

svm_undersampling_data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'data_experiments_svm_undersampling_010823',
                 'svm_undersampling_010823_results.csv'))

nn_undersampling_data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'data_experiments_nn_undersampling_010823',
                 'nn_undersampling_010823_results.csv'))

# BASELINE_DATA = pd.read_csv(
#     os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'baselines_results_added_columns.csv'))

BASELINE_DATA = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'data_experiments_baseline_040423',
                 'baseline_040423_results.csv'))

sessions_list = ['Session_01_01',
                 'Session_02_01',
                 'Session_02_02',
                 'Session_03_01',
                 'Session_03_02',
                 'Session_04_01',
                 'Session_04_02']

participants_list = ['Participant_01', 'Participant_02', 'Participant_03', 'Participant_04']


def calculate_difference_percentage(value_current, value_baseline):
    # print(f'> {comp_name}: {value_current}')
    # perc = ((value_current / value_baseline)) * 100
    # print(f'{perc} %')

    # print(f'diff = {value_baseline - value_current}')
    diff_perc = ((value_current / value_baseline) - 1) * 100
    # print(f'diff perc = {diff_perc} (how much \% of improvement or detriment)')
    # print(f'x times better/worse = {value_baseline / value_current}')

    # return perc, diff_perc
    return diff_perc


def compare_against_baseline(resulting_df, baseline_df, scenario, annotation_type, data_included):
    """
    Takes the current dataset and compare it with the baseline results
    :param resulting_df: dataframe with the interested data already filtered by a query.
    :param baseline_df: baseline dataframe with the interested data already filtered by a query.
    :param scenario: list of scenarios (strings) with any of the possible scenarios
    :param annotation_type: list of strings: parents and/or specialist
    :param data_included: string: 'participants', 'sessions' or 'all_data'
    :return:  None (it prints a message)
    """
    scenario_to_print = f"{data_included}_{scenario}_{annotation_type}"

    avg_acc = resulting_df[['Accuracy']].mean()
    std_acc = resulting_df[['Accuracy']].std()
    avg_acc_bl = baseline_df[['Accuracy']].mean()

    diff_avg = calculate_difference_percentage(avg_acc['Accuracy'], avg_acc_bl['Accuracy'])

    print("..........")
    print(
        f"Average Accuracy in {scenario_to_print} models: {avg_acc.to_string(index=False)}"
        f"(+-{std_acc.to_string(index=False)})\n"
        f"Baseline value in {scenario_to_print} models: {avg_acc_bl.to_string(index=False)}\n"
        f"Difference from Baseline: {diff_avg:.2f}%")

    avg_b_acc = resulting_df[['Accuracy_Balanced']].mean()
    std_b_acc = resulting_df[['Accuracy_Balanced']].std()
    b_acc_avg_bl = baseline_df[['Accuracy_Balanced']].mean()

    diff_b_acc = calculate_difference_percentage(avg_b_acc['Accuracy_Balanced'],
                                                 b_acc_avg_bl['Accuracy_Balanced'])
    print("..........")
    print(
        f"Average Balanced Accuracy in {scenario_to_print} models: {avg_b_acc.to_string(index=False)} "
        f"(+-{std_b_acc.to_string(index=False)})\n"
        f"Baseline value in {scenario_to_print} models: {b_acc_avg_bl.to_string(index=False)}\n"
        f"Difference from Baseline: {diff_b_acc:.2f}%")

    # Best Results ACC & B_ACC

    max_acc = resulting_df.query('Accuracy == Accuracy.max()').iloc[0]
    max_acc_bl = baseline_df.query('Accuracy == Accuracy.max()').iloc[0]
    diff_max_acc = calculate_difference_percentage(max_acc['Accuracy'], max_acc_bl['Accuracy'])

    print("..........")
    print("Best Values for the current Scenario:")
    print("..........")
    max_acc_value = max_acc['Accuracy']
    max_acc_data_included_slug = max_acc['Data_Included_Slug']
    max_acc_annotation_type = max_acc['Annotation_Type']
    acc_result_baseline = get_information(baseline_df, max_acc_data_included_slug, max_acc_annotation_type)

    max_acc_value_bl = max_acc_bl['Accuracy']
    max_acc_bl_data_included_slug = max_acc_bl['Data_Included_Slug']
    max_acc_bl_annotation_type = max_acc_bl['Annotation_Type']
    acc_result_scenario = get_information(resulting_df, max_acc_bl_data_included_slug, max_acc_bl_annotation_type)

    print(
        f"Best Accuracy: {max_acc_value} ({max_acc_data_included_slug}_{max_acc_annotation_type})\n"
        f"Accuracy of the best model in baseline: {acc_result_baseline['Accuracy'].to_string(index=False)} "
        f"({acc_result_baseline['Data_Included_Slug'].to_string(index=False)}_{acc_result_baseline['Annotation_Type'].to_string(index=False)})\n")
    print(
        f"Best Accuracy (baseline): {max_acc_value_bl} "
        f"({max_acc_bl_data_included_slug}_{max_acc_bl_annotation_type})\n"
        f"Accuracy of the best baseline model in this scenario: {acc_result_scenario['Accuracy'].to_string(index=False)} "
        f"({acc_result_scenario['Data_Included_Slug'].to_string(index=False)}_{acc_result_scenario['Annotation_Type'].to_string(index=False)})\n")
    print(f"Difference of Best Acc Value between Scenario and Baseline: {diff_max_acc:.2f}%")

    max_b_acc = resulting_df.query('Accuracy_Balanced == Accuracy_Balanced.max()')
    max_b_acc_bl = baseline_df.query('Accuracy_Balanced == Accuracy_Balanced.max()')
    diff_max_b_acc = calculate_difference_percentage(max_b_acc['Accuracy_Balanced'].iloc[0],
                                                     max_b_acc_bl['Accuracy_Balanced'].iloc[0])

    max_b_acc_value = max_b_acc['Accuracy_Balanced'].to_string(index=False)
    max_b_acc_data_included_slug = max_b_acc['Data_Included_Slug'].to_string(index=False)
    max_b_acc_annotation_type = max_b_acc['Annotation_Type'].to_string(index=False)
    b_acc_result_baseline = get_information(baseline_df, max_b_acc_data_included_slug, max_b_acc_annotation_type)

    max_b_acc_value_bl = max_b_acc_bl['Accuracy_Balanced'].to_string(index=False)
    max_b_acc_bl_data_included_slug = max_b_acc_bl['Data_Included_Slug'].to_string(index=False)
    max_b_acc_bl_annotation_type = max_b_acc_bl['Annotation_Type'].to_string(index=False)
    b_acc_result_scenario = get_information(resulting_df, max_b_acc_bl_data_included_slug, max_b_acc_bl_annotation_type)

    print("..........")
    print(
        f"Best Balanced Accuracy: {max_b_acc_value} ({max_b_acc_data_included_slug}_{max_b_acc_annotation_type})\n"
        f"Balanced Accuracy of the best model in baseline: {b_acc_result_baseline['Accuracy_Balanced'].to_string(index=False)} "
        f"({b_acc_result_baseline['Data_Included_Slug'].to_string(index=False)}_{b_acc_result_baseline['Annotation_Type'].to_string(index=False)})\n"
    )

    print(f"Best Balanced Accuracy (baseline): {max_b_acc_value_bl} "
          f"({max_b_acc_bl_data_included_slug}_{max_b_acc_bl_annotation_type})\n"
          f"Best Accuracy of the best baseline model in this scenario: {b_acc_result_scenario['Accuracy_Balanced'].to_string(index=False)} "
          f"({b_acc_result_scenario['Data_Included_Slug'].to_string(index=False)}_{b_acc_result_scenario['Annotation_Type'].to_string(index=False)})\n"
          )

    print(f"Difference of Best Balanced Acc Values between Scenario and Baseline: {diff_max_b_acc:.2f}%")


def get_information(data_to_look, data_included_slug, annotation_type):
    result = data_to_look.query(
        f"Data_Included_Slug == '{data_included_slug}' & Annotation_Type == '{annotation_type}'")
    return result


def calculate_best_f1_score(resulting_df, baseline_df):
    # Current Batch
    max_f1score_blue = resulting_df.query('F1score_Blue == F1score_Blue.max()')
    max_f1score_green = resulting_df.query('F1score_Green == F1score_Green.max()')
    max_f1score_red = resulting_df.query('F1score_Red == F1score_Red.max()')
    max_f1score_yellow = resulting_df.query('F1score_Yellow == F1score_Yellow.max()')

    # Baseline
    max_f1score_blue_bl = baseline_df.query('F1score_Blue == F1score_Blue.max()')
    max_f1score_green_bl = baseline_df.query('F1score_Green == F1score_Green.max()')
    max_f1score_red_bl = baseline_df.query('F1score_Red == F1score_Red.max()')
    max_f1score_yellow_bl = baseline_df.query('F1score_Yellow == F1score_Yellow.max()')

    # Differences from the baseline
    diff_f1_blue = calculate_difference_percentage(max_f1score_blue['F1score_Blue'].iloc[0],
                                                   max_f1score_blue_bl['F1score_Blue'].iloc[0])

    diff_f1_green = calculate_difference_percentage(max_f1score_green['F1score_Green'].iloc[0],
                                                    max_f1score_green_bl['F1score_Green'].iloc[0])

    diff_f1_red = calculate_difference_percentage(max_f1score_red['F1score_Red'].iloc[0],
                                                  max_f1score_red_bl['F1score_Red'].iloc[0])

    diff_f1_yellow = calculate_difference_percentage(max_f1score_yellow['F1score_Yellow'].iloc[0],
                                                     max_f1score_yellow_bl['F1score_Yellow'].iloc[0])

    print("..........")
    print("..........")
    print("Best Values of F1Score:")
    print(
        f"Best F1Score (Blue): {max_f1score_blue['F1score_Blue'].to_string(index=False)} ({max_f1score_blue['Data_Included_Slug'].to_string(index=False)}_{max_f1score_blue['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_f1score_blue_bl['F1score_Blue'].to_string(index=False)} ({max_f1score_blue_bl['Data_Included_Slug'].to_string(index=False)}_{max_f1score_blue_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff_f1_blue:.2f}%")
    print("..........")
    print(
        f"Best F1Score (Green): {max_f1score_green['F1score_Green'].to_string(index=False)} ({max_f1score_green['Data_Included_Slug'].to_string(index=False)}_{max_f1score_green['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_f1score_green_bl['F1score_Green'].to_string(index=False)} ({max_f1score_green_bl['Data_Included_Slug'].to_string(index=False)}_{max_f1score_green_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff_f1_green:.2f}%")
    print("..........")
    print(
        f"Best F1Score (Red): {max_f1score_red['F1score_Red'].to_string(index=False)} ({max_f1score_red['Data_Included_Slug'].to_string(index=False)}_{max_f1score_red['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_f1score_red_bl['F1score_Red'].to_string(index=False)} ({max_f1score_red_bl['Data_Included_Slug'].to_string(index=False)}_{max_f1score_red_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff_f1_red:.2f}%")
    print("..........")
    print(
        f"Best F1Score (Yellow): {max_f1score_yellow['F1score_Yellow'].to_string(index=False)} ({max_f1score_yellow['Data_Included_Slug'].to_string(index=False)}_{max_f1score_yellow['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_f1score_yellow_bl['F1score_Yellow'].to_string(index=False)} ({max_f1score_yellow_bl['Data_Included_Slug'].to_string(index=False)}_{max_f1score_yellow_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff_f1_yellow:.2f}%")


def calculate_aggregated_f1_score(resulting_df, baseline_df):
    avg_f1_blue = resulting_df[["F1score_Blue"]].mean()
    std_f1_blue = resulting_df[["F1score_Blue"]].std()

    avg_f1_green = resulting_df[["F1score_Green"]].mean()
    std_f1_green = resulting_df[["F1score_Green"]].std()

    avg_f1_red = resulting_df[["F1score_Red"]].mean()
    std_f1_red = resulting_df[["F1score_Red"]].std()

    avg_f1_yellow = resulting_df[["F1score_Yellow"]].mean()
    std_f1_yellow = resulting_df[["F1score_Yellow"]].std()

    avg_f1_blue_bl = baseline_df[["F1score_Blue"]].mean()
    avg_f1_green_bl = baseline_df[["F1score_Green"]].mean()
    avg_f1_red_bl = baseline_df[["F1score_Red"]].mean()
    avg_f1_yellow_bl = baseline_df[["F1score_Yellow"]].mean()

    diff_f1_blue = calculate_difference_percentage(avg_f1_blue['F1score_Blue'],
                                                   avg_f1_blue_bl['F1score_Blue'])

    diff_f1_green = calculate_difference_percentage(avg_f1_green['F1score_Green'],
                                                    avg_f1_green_bl['F1score_Green'])

    diff_f1_red = calculate_difference_percentage(avg_f1_red['F1score_Red'],
                                                  avg_f1_red_bl['F1score_Red'])

    diff_f1_yellow = calculate_difference_percentage(avg_f1_yellow['F1score_Yellow'],
                                                     avg_f1_yellow_bl['F1score_Yellow'])

    print("..........")
    print("..........")
    print(
        f"Average F1score (blue) models: {avg_f1_blue.to_string(index=False)} "
        f"(+-{std_f1_blue.to_string(index=False)})\n"
        f"Baseline value: {avg_f1_blue_bl.to_string(index=False)}\n"
        f"Difference from Baseline: {diff_f1_blue:.2f}%")

    print("..........")
    print(
        f"Average F1score (green) models: {avg_f1_green.to_string(index=False)} "
        f"(+-{std_f1_green.to_string(index=False)})\n"
        f"Baseline value: {avg_f1_green_bl.to_string(index=False)}\n"
        f"Difference from Baseline: {diff_f1_green:.2f}%")

    print("..........")
    print(
        f"Average F1score (red) models: {avg_f1_red.to_string(index=False)} "
        f"(+-{std_f1_red.to_string(index=False)})\n"
        f"Baseline value: {avg_f1_red_bl.to_string(index=False)}\n"
        f"Difference from Baseline: {diff_f1_red:.2f}%")

    print("..........")
    print(
        f"Average F1score (yellow) models: {avg_f1_yellow.to_string(index=False)} "
        f"(+-{std_f1_yellow.to_string(index=False)})\n"
        f"Baseline value: {avg_f1_yellow_bl.to_string(index=False)}\n"
        f"Difference from Baseline: {diff_f1_yellow:.2f}%")


def get_subset_data(data_to_apply, scenario, annotation_type, subdataset_case):
    """

    :param data_to_apply:
    :param scenario:
    :param annotation_type:
    :param subdataset_case:
    :return:
    """
    if subdataset_case == 'participants':
        resulting_df = data_to_apply.query(
            f"Participant in {participants_list} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
        baseline_df = BASELINE_DATA.query(
            f"Participant in {participants_list} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
    elif subdataset_case == 'sessions':
        resulting_df = data_to_apply.query(
            f"Session in {sessions_list} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
        baseline_df = BASELINE_DATA.query(
            f"Session in {sessions_list} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
    elif subdataset_case == 'all_data':
        resulting_df = data_to_apply.query(
            f"Participant == 'All data' & Scenario in {scenario} & Annotation_Type in {annotation_type}")
        baseline_df = BASELINE_DATA.query(
            f"Participant == 'All data' & Scenario in {scenario} & Annotation_Type in {annotation_type}")

    return resulting_df, baseline_df


def calculate_best_acc(data_to_apply):
    max_acc = data_to_apply.query('Accuracy == Accuracy.max()')
    max_acc_bl = BASELINE_DATA.query('Accuracy == Accuracy.max()')
    diff = calculate_difference_percentage(max_acc['Accuracy'].iloc[0], max_acc_bl['Accuracy'].iloc[0])

    max_b_acc = data_to_apply.query('Accuracy_Balanced == Accuracy_Balanced.max()')
    max_b_acc_bl = BASELINE_DATA.query('Accuracy_Balanced == Accuracy_Balanced.max()')
    diff_b_acc = calculate_difference_percentage(max_b_acc['Accuracy_Balanced'].iloc[0],
                                                 max_b_acc_bl['Accuracy_Balanced'].iloc[0])
    print("Best Values for the Data Experiment Batch:")
    print(
        f"Best Accuracy: {max_acc['Accuracy'].to_string(index=False)} ({max_acc['Data_Included_Slug'].to_string(index=False)}_{max_acc['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_acc_bl['Accuracy'].to_string(index=False)} ({max_acc_bl['Data_Included_Slug'].to_string(index=False)}_{max_acc_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff:.2f}%")

    print("..........\n")
    print(
        f"Best Balanced Accuracy: {max_b_acc['Accuracy_Balanced'].to_string(index=False)} ({max_b_acc['Data_Included_Slug'].to_string(index=False)}_{max_b_acc['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_b_acc_bl['Accuracy_Balanced'].to_string(index=False)} ({max_b_acc_bl['Data_Included_Slug'].to_string(index=False)}_{max_b_acc_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff_b_acc:.2f}%")


def get_model_group(row):
    if row['Participant'] == 'All data':
        return 'GM'
    elif 'Participant' in str(row['Participant']):
        return 'PSM'
    elif 'Session' in str(row['Session']):
        return 'SSM'


if __name__ == '__main__':
    # Example of configuration values
    scenario = ['va_late_fusion', 'va_early_fusion']
    annotation_type = ['parents']
    subset_data = 'sessions'
    resulting_df, baseline_df = get_subset_data(oversampling_data_results,
                                                scenario=scenario,
                                                annotation_type=annotation_type,
                                                subdataset_case=subset_data)
    compare_against_baseline(resulting_df, baseline_df, scenario, annotation_type, subset_data)
    # calculate_aggregated_f1_score(resulting_df, baseline_df)
    # calculate_best_f1_score(resulting_df, baseline_df)

    # compare_against_baseline_sessions(oversampling_data_results, scenario=['v'],
    #                                   annotation_type=['parents', 'specialist'])
    # compare_against_baseline_participants(oversampling_data_results, scenario=['v'],
    #                                       annotation_type=['parents', 'specialist'])

    # nn_bl_data_results['model_group'] = nn_bl_data_results.apply(lambda row: get_model_group(row), axis=1)
    # print('hi')
