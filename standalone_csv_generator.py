
import os
import numpy as np
import argparse
import pickle
import time
import userid_map
import csv_utility
import pandas as pd


def data_frame_to_csv_daily(df, score_column, score_name, prefix="", results_folder="results/"):
    """
    This function writes the prediction results from data frames to for DGTB to csv files.

    Args:
        df (data frame): data with prediction results
        score_column (string): name of the column with prediction scores
        score_name (string): score name
        prefix (string): path to write the csv file to and the prefix for file name
    """

    results_folder += 'daily/' + score_column + '/'

    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    ids     = np.array(df.index.get_level_values('Participant'))
    umn_ids = userid_map.perform_map(ids, 'data/mperf_ids.txt')
    vals    = np.array(df[score_column].fillna(0))

    dates   = [x.strftime("%-m/%-d/%Y") for x in df.index.get_level_values('Date')]

    csv_utility.write_csv_daily(results_folder + "%s"%(prefix), umn_ids, dates, np.array([""]*len(dates)), score_name, vals)


def generate_csv_from_pickle_file_daily(data_directory, pkl_file, indicator_name, results_dir):
    """
    This function loads data frames from the pkl files to be written into csv files (for DGTB).

    Args:
        data_directory (str): path for pkl file
        pkl_file (str): name of the pkl file
        indicator_name (str): name of the target
        results_dir (str): path for csv files

    """

    data_frames_dict = pickle.load(open(data_directory + pkl_file, 'rb'))
    df_te = data_frames_dict["df_te"]
    data_frame_to_csv_daily(df_te, "target", indicator_name, prefix="ground_truth_", results_folder=results_dir)
    data_frame_to_csv_daily(df_te, "prediction", indicator_name, prefix="prediction_", results_folder=results_dir)

    df_tr = data_frames_dict["df_tr"]
    df_all = pd.concat([data_frames_dict["df_te"],data_frames_dict["df_tr"]],axis=0)
    
    data_frame_to_csv_daily(df_all, "target", indicator_name, prefix="all_ground_truth_", results_folder=results_dir)
    data_frame_to_csv_daily(df_all, "prediction", indicator_name, prefix="all_prediction_", results_folder=results_dir)


def data_frame_to_csv_initial(df, score_column, score_name, prefix="", results_folder="results/"):
    """
    This function writes the prediction results from data frames to for IGBT to csv files.

    Args:
        df (data frame): data cases with prediction results
        score_column (string): name of the column with prediction scores
        score_name (string): score name
        prefix (string): path to write the csv file to and the prefix for file name
    """

    results_folder += 'initial/' + score_column + '/'

    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    ids     = np.array(df.index.get_level_values('Participant'))
    umn_ids = userid_map.perform_map(ids, 'data/mperf_ids.txt')
    vals    = np.array(df[score_column].fillna(0))

    csv_utility.write_csv_initial(results_folder + "%s"%(prefix), umn_ids, np.array([""]*len(ids)), np.array([""]*len(ids)), score_name, vals)


def generate_csv_from_pickle_file_initial(data_directory, pkl_file, indicator_name, results_dir):
    """
    This function loads data frames from the pkl files to be written into csv files (for IGBT).

    Args:
        data_directory (str): path for pkl file
        pkl_file (str): name of the pkl file
        indicator_name (str): name of the target
        results_dir (str): path for csv files

    """

    data_frames_dict = pickle.load(open(data_directory + pkl_file, 'rb'))
    df_te = data_frames_dict["df_te"]
    data_frame_to_csv_initial(df_te, "target", indicator_name, prefix="ground_truth_", results_folder=results_dir)
    data_frame_to_csv_initial(df_te, "prediction", indicator_name, prefix="prediction_", results_folder=results_dir)

    df_tr = data_frames_dict["df_tr"]
    df_all = pd.concat([data_frames_dict["df_te"],data_frames_dict["df_tr"]],axis=0)
    
    data_frame_to_csv_initial(df_all, "target", indicator_name, prefix="all_ground_truth_", results_folder=results_dir)
    data_frame_to_csv_initial(df_all, "prediction", indicator_name, prefix="all_prediction_", results_folder=results_dir)


def main(args):

    data_directory = args.data_directory
    experiment_name = args.experiment_name
    no_stamp = args.no_stamp

    # can be used for timestamped folders:
    if not no_stamp:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        results_dir = "results-" + timestr + "/"

    else:
        results_dir = "results/"


    # producing separate csv file matching UMN's format:
    for fname in os.listdir(data_directory):

        if (experiment_name in fname) and (".d.pkl" in fname):
            indicator_name = fname[len(experiment_name + "-"):-len(".pkl")]
            generate_csv_from_pickle_file_daily(data_directory, fname, indicator_name, results_dir)

        elif (experiment_name in fname) and (".pkl" in fname):
            indicator_name = fname[len(experiment_name + "-"):-len(".pkl")]
            generate_csv_from_pickle_file_initial(data_directory, fname, indicator_name, results_dir)

    print("Outputting Daily CSVs")
    output_filename = results_dir + "daily/prediction/merged_prediction.csv"
    csv_utility.merge_csv_daily(output_filename, results_dir + "daily/prediction/prediction_")

    output_filename = results_dir + "daily/target/merged_target.csv"
    csv_utility.merge_csv_daily(output_filename, results_dir + "daily/target/ground_truth_")

    output_filename = results_dir + "daily/prediction/all_merged_prediction.csv"
    csv_utility.merge_csv_daily(output_filename, results_dir + "daily/prediction/all_prediction_")

    output_filename = results_dir + "daily/target/all_merged_target.csv"
    csv_utility.merge_csv_daily(output_filename, results_dir + "daily/target/all_ground_truth_")


    print("Outputting Initial CSVs")
    # producing single csv files matching IARPA's format:
    output_filename = results_dir + "initial/prediction/merged_prediction.csv"
    csv_utility.merge_csv_initial(output_filename, results_dir + "initial/prediction/prediction_")

    output_filename = results_dir + "initial/target/merged_target.csv"
    csv_utility.merge_csv_initial(output_filename, results_dir + "initial/target/ground_truth_")

    # producing single csv files matching IARPA's format:
    output_filename = results_dir + "initial/prediction/all_merged_prediction.csv"
    csv_utility.merge_csv_initial(output_filename, results_dir + "initial/prediction/all_prediction_")

    output_filename = results_dir + "initial/target/all_merged_target.csv"
    csv_utility.merge_csv_initial(output_filename, results_dir + "initial/target/all_ground_truth_")






if __name__ == "__main__":
    """
    Start of execution.  Creates the argparse.ArgumentParser() that main() uses to load data
    and write the CSVs.
    """
    parser = argparse.ArgumentParser(description="generates CSV reports from experiment results")
    parser.add_argument("--data-directory", help="path to data files")
    parser.add_argument("--experiment-name", help="name of experiment")
    parser.add_argument("--no-stamp", action='store_const', const=True, help="add --no-stamp flag to write to an un-timestamped results folder")
    args = parser.parse_args()

    main(args)