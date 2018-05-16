# run summarization
sh run_master_summarize.sh

# run mobile/daily and initial indicator experiments
sh run_in_serial.sh master_summary_dataframe.pkl ./data/experiment_definitions/daily/
sh run_in_serial.sh master_summary_dataframe.pkl ./data/experiment_definitions/initial/

# generate CSV results
python3 standalone_report_generator.py --data-directory ./experiment_results/ --experiment-name basic
