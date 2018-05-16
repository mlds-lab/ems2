
data_file=$1
edd_dir=$2

. ./ems_config.sh

for filename in $edd_dir/*; do
    echo $(date) ' starting ' $filename
    python3.6 experiment_engine4.py --data-file ./master_data_file/$data_file --edd-dir $edd_dir --edd-name ${filename##*/} --no-spark
done