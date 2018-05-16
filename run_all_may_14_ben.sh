
#Run IGTB 
bash run_in_serial.sh master_summary_dataframe-20180514-143842.pkl experiment_definitions_300/initial/

#Run DGTB
bash run_in_serial.sh master_summary_dataframe-20180514-143842.pkl experiment_definitions_300/daily/

#Run CSV generation
#python standalone_csv_generator.py --data-directory experiment_output/ --experiment-name may14 --no-stamp

#Copy results folder
#cp results results-may14-final/



bash run_in_serial.sh master_summary_dataframe-20180514-190154.pkl experiment_definitions_300/initial/

bash run_in_serial.sh master_summary_dataframe-20180514-190154.pkl experiment_definitions_300/daily/

python standalone_csv_generator.py --data-directory experiment_output/ --experiment-name may14 --no-stamp

python standalone_csv_generator.py --data-directory experiment_output-May15-1 --experiment-name may14 --no-stamp