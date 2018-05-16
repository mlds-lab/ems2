
#export EMS_DUMP_DIR="master_dump"
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
mkdir -p reports
# jupyter nbconvert --execute --to html --template ./clean_output.tpl results_report.ipynb --output reports/results_report-$timestamp.html --ExecutePreprocessor.kernel_name=python
# jupyter nbconvert --execute --to html --template ./clean_output.tpl master_data_report.ipynb --output reports/master_data_report-$timestamp.html --ExecutePreprocessor.kernel_name=python

jupyter nbconvert --execute --to html --template ./clean_output.tpl results_report.ipynb --ExecutePreprocessor.timeout=300 --output reports/results_report-$timestamp.html --ExecutePreprocessor.kernel_name=python
jupyter nbconvert --execute --to html --template ./clean_output.tpl master_data_report.ipynb --ExecutePreprocessor.timeout=300  --output reports/master_data_report-$timestamp.html --ExecutePreprocessor.kernel_name=python