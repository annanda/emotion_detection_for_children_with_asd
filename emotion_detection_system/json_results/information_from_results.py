import pandas as pd
import os

from emotion_detection_system.conf import main_folder

date_experiment = '280223'
file_name = 'undersampling_results.csv'

data_results = pd.read_csv(
    os.path.join(main_folder, 'emotion_detection_system', 'json_results', f'{date_experiment}',
                 file_name))

################
# MAX (ACC & B_ACC) VALUES IN GENERAL
max_acc_row = data_results.query('Accuracy == Accuracy.max()')
max_b_acc_row = data_results.query('Accuracy_Balanced == Accuracy_Balanced.max()')
print(
    f"Best Accuracy: {max_acc_row['Accuracy'].to_string(index=False)} ({max_acc_row['Data_Included_Slug'].to_string(index=False)}_{max_acc_row['Annotation_Type'].to_string(index=False)})")

print(
    f"Best Balanced Accuracy: {max_b_acc_row['Accuracy_Balanced'].to_string(index=False)} ({max_b_acc_row['Data_Included_Slug'].to_string(index=False)}_{max_b_acc_row['Annotation_Type'].to_string(index=False)})")
################
# MULTIMODAL IN GENERAL (ACC & B_ACC)
best_acc_va_row = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')").query(
    'Accuracy == Accuracy.max()')
best_b_acc_va_row = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')").query(
    'Accuracy_Balanced == Accuracy_Balanced.max()')
avg_va_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[['Accuracy']].mean()
std_va_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[['Accuracy']].std()

print(
    f"Best VA Accuracy: {best_acc_va_row['Accuracy'].to_string(index=False)} ({best_acc_va_row['Data_Included_Slug'].to_string(index=False)}_{best_acc_va_row['Annotation_Type'].to_string(index=False)})")
print(
    f"Best VA Balanced Accuracy: {best_b_acc_va_row['Accuracy_Balanced'].to_string(index=False)} ({best_b_acc_va_row['Data_Included_Slug'].to_string(index=False)}_{best_b_acc_va_row['Annotation_Type'].to_string(index=False)})")
print(
    f"Avg Accuracy VA models: {avg_va_acc_models.to_string(index=False)} (+-{std_va_acc_models.to_string(index=False)})")

################

avg_va_b_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[
    ['Accuracy_Balanced']].mean()
std_va_b_acc_models = data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion')")[
    ['Accuracy_Balanced']].std()

print(
    f"Avg Balanced Accuracy VA models: {avg_va_b_acc_models.to_string(index=False)} (+-{std_va_b_acc_models.to_string(index=False)})")

avg_va_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy']].mean()
std_va_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (parents): {avg_va_acc_models_parents.to_string(index=False)} (+-{std_va_acc_models_parents.to_string(index=False)})")

avg_va_b_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].mean()
std_va_b_acc_models_parents = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].std()

print(
    f"Avg Balanced Accuracy VA models (parents): {avg_va_b_acc_models_parents.to_string(index=False)} (+-{std_va_b_acc_models_parents.to_string(index=False)})")

avg_va_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy']].mean()
std_va_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (specialist): {avg_va_acc_models_specialist.to_string(index=False)} (+-{avg_va_acc_models_specialist.to_string(index=False)})")

avg_va_b_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].mean()
std_va_b_acc_models_specialist = \
    data_results.query("Scenario in ('va_late_fusion', 'va_early_fusion') & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].std()

################
# MULTIMODAL EARLY VS LATE FUSION (ACC & B_ACC)

# LATE FUSION
avg_acc_va_models_late_fusion = data_results.query("Scenario == 'va_late_fusion'")[
    ['Accuracy']].mean()
std_acc_va_models_late_fusion = data_results.query("Scenario == 'va_late_fusion'")[
    ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (late fusion): {avg_acc_va_models_late_fusion.to_string(index=False)} (+-{avg_acc_va_models_late_fusion.to_string(index=False)})")

avg_b_acc_va_models_late_fusion = data_results.query("Scenario == 'va_late_fusion'")[
    ['Accuracy_Balanced']].mean()
std_b_acc_va_models_late_fusion = data_results.query("Scenario == 'va_late_fusion'")[
    ['Accuracy_Balanced']].std()

avg_acc_va_models_late_fusion_parents = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy']].mean()
std_acc_va_models_late_fusion_parents = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (late fusion - parents): {avg_acc_va_models_late_fusion_parents.to_string(index=False)} (+-{avg_acc_va_models_late_fusion_parents.to_string(index=False)})")

avg_b_acc_va_models_late_fusion_parents = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].mean()
std_b_acc_va_models_late_fusion_parents = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].std()

avg_acc_va_models_late_fusion_specialist = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy']].mean()
std_acc_va_models_late_fusion_specialist = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (late fusion - specialist): {avg_acc_va_models_late_fusion_specialist.to_string(index=False)} (+-{std_acc_va_models_late_fusion_specialist.to_string(index=False)})")

avg_b_acc_va_models_late_fusion_specialist = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].mean()
std_b_acc_va_models_late_fusion_specialist = \
    data_results.query("Scenario == 'va_late_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].std()

# EARLY FUSION
avg_acc_va_models_early_fusion = data_results.query("Scenario == 'va_early_fusion'")[
    ['Accuracy']].mean()
std_acc_va_models_early_fusion = data_results.query("Scenario == 'va_early_fusion'")[
    ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (early fusion): {avg_acc_va_models_early_fusion.to_string(index=False)} (+-{std_acc_va_models_early_fusion.to_string(index=False)})")

avg_b_acc_va_models_early_fusion = data_results.query("Scenario == 'va_early_fusion'")[
    ['Accuracy_Balanced']].mean()
std_b_acc_va_models_early_fusion = data_results.query("Scenario == 'va_early_fusion'")[
    ['Accuracy_Balanced']].std()

avg_acc_va_models_early_fusion_parents = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy']].mean()
std_acc_va_models_early_fusion_parents = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (early fusion - parents): {avg_acc_va_models_early_fusion_parents.to_string(index=False)} (+-{std_acc_va_models_early_fusion_parents.to_string(index=False)})")

avg_b_acc_va_models_early_fusion_parents = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].mean()
std_b_acc_va_models_early_fusion_parents = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'parents'")[
        ['Accuracy_Balanced']].std()

avg_acc_va_models_early_fusion_specialist = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy']].mean()
std_acc_va_models_early_fusion_specialist = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy']].std()

print(
    f"Avg Accuracy VA models (early fusion - specialist): {avg_acc_va_models_early_fusion_specialist.to_string(index=False)} (+-{std_acc_va_models_early_fusion_specialist.to_string(index=False)})")

avg_b_acc_va_models_early_fusion_specialist = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].mean()
std_b_acc_va_models_early_fusion_specialist = \
    data_results.query("Scenario == 'va_early_fusion' & Annotation_Type == 'specialist'")[
        ['Accuracy_Balanced']].std()

################
# UNIMODAL VIDEO VS AUDIO (ACC & B_ACC)

# VIDEO
avg_acc_v_models = data_results.query("Scenario == 'v'")[
    ['Accuracy']].mean()
std_acc_v_models_early_fusion = data_results.query("Scenario == 'v'")[
    ['Accuracy']].std()

print(
    f"Avg Accuracy V models: {avg_acc_v_models.to_string(index=False)} (+-{std_acc_v_models_early_fusion.to_string(index=False)})")

avg_b_acc_v_models_early_fusion = data_results.query("Scenario == 'v'")[
    ['Accuracy_Balanced']].mean()
std_b_acc_v_models_early_fusion = data_results.query("Scenario == 'v'")[
    ['Accuracy_Balanced']].std()

# AUDIO
avg_acc_a_models = data_results.query("Scenario == 'a'")[
    ['Accuracy']].mean()
std_acc_a_models_early_fusion = data_results.query("Scenario == 'a'")[
    ['Accuracy']].std()

print(
    f"Avg Accuracy A models: {avg_acc_a_models.to_string(index=False)} (+-{std_acc_a_models_early_fusion.to_string(index=False)})")

avg_b_acc_a_models_early_fusion = data_results.query("Scenario == 'a'")[
    ['Accuracy_Balanced']].mean()
std_b_acc_a_models_early_fusion = data_results.query("Scenario == 'a'")[
    ['Accuracy_Balanced']].std()

################
# CLASSES CLASSIFICATION IN GENERAL (F1 SCORES)

best_f1_blue_row = data_results.query('F1score_Blue == F1score_Blue.max()')
best_f1_green_row = data_results.query('F1score_Green == F1score_Green.max()')
best_f1_red_row = data_results.query('F1score_Red == F1score_Red.max()')
best_f1_yellow_row = data_results.query('F1score_Yellow == F1score_Yellow.max()')

avg_f1_blue = data_results[["F1score_Blue"]].mean()
avg_f1_green = data_results[["F1score_Green"]].mean()
avg_f1_red = data_results[["F1score_Red"]].mean()
avg_f1_yellow = data_results[["F1score_Yellow"]].mean()

print(max_acc_row)
print(max_b_acc_row)
