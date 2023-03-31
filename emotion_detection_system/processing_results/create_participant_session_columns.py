import pandas as pd

dataframe_data = pd.read_csv(
    "/Users/annanda/PycharmProjects/emotion_detection_system/emotion_detection_system/json_results/baselines_results.csv")


def define_participant(row_information):
    data_slug = row_information['Data_Included_Slug']
    if 'participant' in data_slug:
        part = data_slug.split("_")[:2]
        part = "_".join(part).capitalize()
        return part
    elif 'all_data' in data_slug:
        return 'All data'
    else:
        return None


def define_session(row_information):
    data_slug = row_information['Data_Included_Slug']
    if 'session' in data_slug:
        part = data_slug.split("_")[:3]
        part = "_".join(part).capitalize()
        return part
    else:
        return None


if __name__ == '__main__':
    dataframe_data['Participant'] = dataframe_data.apply(define_participant, axis=1)
    dataframe_data['Session'] = dataframe_data.apply(define_session, axis=1)
    dataframe_data.to_csv(
        "/Users/annanda/PycharmProjects/emotion_detection_system/emotion_detection_system/json_results/baselines_results_added_columns.csv",
        index=False)
    print('done!')
