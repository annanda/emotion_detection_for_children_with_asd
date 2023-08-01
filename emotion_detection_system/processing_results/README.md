# Scripts to help processing results from a data experiment setup

## Quick-start

1. Change the variables `results_path` and `batch_data_experiments` in the file `process_results_json.py` to reflect the
   name of the data experiment.
2. Run `process_results_json.py`.
   A CSV file with the results will be created in the folder `json_results/<DATA_EXPERIMENT_SLUG>`. This file can be
   read on Jupyter notebook to better visualise the results per parameters analysed, e.g., modalities, annotation,
   fusion layer, between others.