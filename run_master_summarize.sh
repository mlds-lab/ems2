#!/bin/bash

. ./ems_config.sh

streams="master_streams_vm.json"
subjects="master_participants.json"
data_file="master_data_file/master_summary_dataframe.pkl"
parallel="by-subject"
localtime="true"

#Read optional mode from command line
if (( $# != 1 )); then
    mode="scratch"
else
	mode=$1
fi

python3.6 master_summarization_driver.py $streams $subjects $data_file $mode $parallel $localtime

