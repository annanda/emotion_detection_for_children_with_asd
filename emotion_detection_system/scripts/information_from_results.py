import pandas as pd
import os

from emotion_detection_system.conf import main_folder

date_experiment = '280223'
file_name = 'undersampling_results.csv'

data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', f'{date_experiment}',
                 file_name))

BASELINE_DATA = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', 'baselines_results_added_columns.csv'))


def calculate_difference_percentage(value_current, value_baseline):
    # print(f'> {comp_name}: {value_current}')
    perc = ((value_current / value_baseline)) * 100
    # print(f'{perc} %')

    # print(f'diff = {value_baseline - value_current}')
    diff_perc = ((value_current / value_baseline) - 1) * 100
    # print(f'diff perc = {diff_perc} (how much \% of improvement or detriment)')
    # print(f'x times better/worse = {value_baseline / value_current}')

    # return perc, diff_perc
    return diff_perc


def compare_against_baseline(scenario, annotation_type, participant=None, session=None, metric=None):
    """
    Takes the current dataset and compare it with the baseline results
    :param scenario: list of scenarios (strings) with any of the possible scenarios
    :param annotation_type: list of strings: parents and/or specialist
    :param metric: string or None: f1score, precision, recall if None: calculates values for acc and b_acc
    :return:  None (it prints a message)
    """
    if participant:
        resulting_df = data_results.query(
            f"Participant in {participant} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
        baseline_df = BASELINE_DATA.query(
            f"Participant in {participant} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
    elif session:
        resulting_df = data_results.query(
            f"Session in {session} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
        baseline_df = BASELINE_DATA.query(
            f"Session in {session} & Scenario in {scenario} & Annotation_Type in {annotation_type}")
    else:
        resulting_df = data_results.query(f"Scenario in {scenario} & Annotation_Type in {annotation_type}")
        baseline_df = BASELINE_DATA.query(f"Scenario in {scenario} & Annotation_Type in {annotation_type}")

    if not metric:
        avg_acc = resulting_df[['Accuracy']].mean()
        std_acc = resulting_df[['Accuracy']].std()
        avg_acc_bl = baseline_df[['Accuracy']].mean()

        scenario_to_print = f"{scenario}_{annotation_type}_{participant}" if participant else f"{scenario}_{annotation_type}"

        diff_avg = calculate_difference_percentage(avg_acc['Accuracy'], avg_acc_bl['Accuracy'])

        print("##########\n")
        print(
            f"Average Accuracy in {scenario_to_print} models: {avg_acc.to_string(index=False)}"
            f"(+-{std_acc.to_string(index=False)})\n"
            f"Baseline value: {avg_acc_bl.to_string(index=False)}\n"
            f"Difference from Baseline: {diff_avg:.2f}%")

        avg_b_acc = resulting_df[['Accuracy_Balanced']].mean()
        std_b_acc = resulting_df[['Accuracy_Balanced']].std()
        b_acc_avg_bl = baseline_df[['Accuracy_Balanced']].mean()

        diff_b_acc = calculate_difference_percentage(avg_b_acc['Accuracy_Balanced'],
                                                     b_acc_avg_bl['Accuracy_Balanced'])
        print("##########\n")
        print(
            f"Average Balanced Accuracy in {scenario_to_print} models: {avg_b_acc.to_string(index=False)} "
            f"(+-{std_b_acc.to_string(index=False)})\n"
            f"Baseline value: {b_acc_avg_bl.to_string(index=False)}\n"
            f"Difference from Baseline: {diff_b_acc:.2f}%")

        # Best Results ACC & B_ACC

        max_acc = resulting_df.query('Accuracy == Accuracy.max()')
        max_acc_bl = baseline_df.query('Accuracy == Accuracy.max()')
        diff_max_acc = calculate_difference_percentage(max_acc['Accuracy'].iloc[0], max_acc_bl['Accuracy'].iloc[0])

        print("##########\n")
        print("Best Values for the current Scenario:")
        print(
            f"Best Accuracy: {max_acc['Accuracy'].to_string(index=False)} ({max_acc['Data_Included_Slug'].to_string(index=False)}_{max_acc['Annotation_Type'].to_string(index=False)})\n"
            f"Compared to baseline: {max_acc_bl['Accuracy'].to_string(index=False)} ({max_acc_bl['Data_Included_Slug'].to_string(index=False)}_{max_acc_bl['Annotation_Type'].to_string(index=False)})\n"
            f"Difference from Baseline: {diff_max_acc:.2f}%")

        max_b_acc = resulting_df.query('Accuracy_Balanced == Accuracy_Balanced.max()')
        max_b_acc_bl = baseline_df.query('Accuracy_Balanced == Accuracy_Balanced.max()')
        diff_max_b_acc = calculate_difference_percentage(max_b_acc['Accuracy_Balanced'].iloc[0],
                                                         max_b_acc_bl['Accuracy_Balanced'].iloc[0])

        print("##########\n")
        print(
            f"Best Balanced Accuracy: {max_b_acc['Accuracy_Balanced'].to_string(index=False)} ({max_b_acc['Data_Included_Slug'].to_string(index=False)}_{max_b_acc['Annotation_Type'].to_string(index=False)})\n"
            f"Compared to baseline: {max_b_acc_bl['Accuracy_Balanced'].to_string(index=False)} ({max_b_acc_bl['Data_Included_Slug'].to_string(index=False)}_{max_b_acc_bl['Annotation_Type'].to_string(index=False)})\n"
            f"Difference from Baseline: {diff_max_b_acc:.2f}%")


    else:
        print('Not Accuracy metric')


def calculate_best_acc():
    max_acc = data_results.query('Accuracy == Accuracy.max()')
    max_acc_bl = BASELINE_DATA.query('Accuracy == Accuracy.max()')
    diff = calculate_difference_percentage(max_acc['Accuracy'].iloc[0], max_acc_bl['Accuracy'].iloc[0])

    max_b_acc = data_results.query('Accuracy_Balanced == Accuracy_Balanced.max()')
    max_b_acc_bl = BASELINE_DATA.query('Accuracy_Balanced == Accuracy_Balanced.max()')
    diff_b_acc = calculate_difference_percentage(max_b_acc['Accuracy_Balanced'].iloc[0],
                                                 max_b_acc_bl['Accuracy_Balanced'].iloc[0])
    print("##########\n")
    print("Best Values for the Data Experiment Batch:")
    print(
        f"Best Accuracy: {max_acc['Accuracy'].to_string(index=False)} ({max_acc['Data_Included_Slug'].to_string(index=False)}_{max_acc['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_acc_bl['Accuracy'].to_string(index=False)} ({max_acc_bl['Data_Included_Slug'].to_string(index=False)}_{max_acc_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff:.2f}%")

    print("##########\n")
    print(
        f"Best Balanced Accuracy: {max_b_acc['Accuracy_Balanced'].to_string(index=False)} ({max_b_acc['Data_Included_Slug'].to_string(index=False)}_{max_b_acc['Annotation_Type'].to_string(index=False)})\n"
        f"Compared to baseline: {max_b_acc_bl['Accuracy_Balanced'].to_string(index=False)} ({max_b_acc_bl['Data_Included_Slug'].to_string(index=False)}_{max_b_acc_bl['Annotation_Type'].to_string(index=False)})\n"
        f"Difference from Baseline: {diff_b_acc:.2f}%")


def calculate_best_f1_score(data_to_apply):
    # Current Batch
    max_f1score_blue = data_to_apply.query('F1score_Blue == F1score_Blue.max()')
    max_f1score_green = data_to_apply.query('F1score_Green == F1score_Green.max()')
    max_f1score_red = data_to_apply.query('F1score_Red == F1score_Red.max()')
    max_f1score_yellow = data_to_apply.query('F1score_Yellow == F1score_Yellow.max()')

    # Baseline
    max_f1score_blue_bl = BASELINE_DATA.query('F1score_Blue == F1score_Blue.max()')
    max_f1score_green_bl = BASELINE_DATA.query('F1score_Green == F1score_Green.max()')
    max_f1score_red_bl = BASELINE_DATA.query('F1score_Red == F1score_Red.max()')
    max_f1score_yellow_bl = BASELINE_DATA.query('F1score_Yellow == F1score_Yellow.max()')

    # Differences from the baseline
    diff_f1_blue = calculate_difference_percentage(max_f1score_blue['F1score_Blue'].iloc[0],
                                                   max_f1score_blue_bl['F1score_Blue'].iloc[0])

    diff_f1_green = calculate_difference_percentage(max_f1score_green['F1score_Green'].iloc[0],
                                                    max_f1score_green_bl['F1score_Green'].iloc[0])

    diff_f1_red = calculate_difference_percentage(max_f1score_red['F1score_Red'].iloc[0],
                                                  max_f1score_red_bl['F1score_Red'].iloc[0])

    diff_f1_yellow = calculate_difference_percentage(max_f1score_yellow['F1score_Yellow'].iloc[0],
                                                     max_f1score_yellow_bl['F1score_Yellow'].iloc[0])

    print("##########\n")
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


def calculate_aggregated_f1_score():
    pass


# avg_f1_blue = data_results[["F1score_Blue"]].mean()
# avg_f1_green = data_results[["F1score_Green"]].mean()
# avg_f1_red = data_results[["F1score_Red"]].mean()
# avg_f1_yellow = data_results[["F1score_Yellow"]].mean()

if __name__ == '__main__':
    # Example of configuration values
    # scenario = ['va_late_fusion', 'va_early_fusion']
    # annotation_type = ['parents', 'specialist']
    # participant = ['All data']
    # session = None

    # calculate_best_acc()
    calculate_best_f1_score(data_results)
    # compare_against_baseline(scenario=['va_late_fusion', 'va_early_fusion'], annotation_type=['parents', 'specialist'])
    # compare_against_baseline(scenario=['va_late_fusion', 'va_early_fusion'], annotation_type=['parents'])
    # compare_against_baseline(scenario=['va_late_fusion', 'va_early_fusion'], annotation_type=['specialist'])

    # compare_against_baseline(scenario=['va_late_fusion'], annotation_type=['parents', 'specialist'])
    # compare_against_baseline(scenario=['va_early_fusion'], annotation_type=['parents', 'specialist'])
    #
    # compare_against_baseline(scenario=['va_late_fusion'], annotation_type=['parents'])
    # compare_against_baseline(scenario=['va_early_fusion'], annotation_type=['parents'])
    #
    # compare_against_baseline(scenario=['va_late_fusion'], annotation_type=['specialist'])
    # compare_against_baseline(scenario=['va_early_fusion'], annotation_type=['specialist'])

    # compare_against_baseline(scenario=['v'], annotation_type=['parents', 'specialist'])
    # compare_against_baseline(scenario=['v'], annotation_type=['specialist'])
    # compare_against_baseline(scenario=['v'], annotation_type=['parents'])
    #
    # compare_against_baseline(scenario=['a'], annotation_type=['parents', 'specialist'])
    # compare_against_baseline(scenario=['a'], annotation_type=['specialist'])
    # compare_against_baseline(scenario=['a'], annotation_type=['parents'])