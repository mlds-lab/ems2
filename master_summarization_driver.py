import sys
import io
import os
from io import StringIO
import warnings
import numpy as np
import json
# import matplotlib.pyplot as plt
# import mosaic_utils as mu
import datetime as dt
# from features import extract_features
import cc_data_retriever as data_retriever
from random import randint
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
# import summarizer as summarizer
from minio import Minio
from minio.error import ResponseError
from pyspark import SparkContext, SparkConf
from cerebralcortex.cerebralcortex import CerebralCortex

import master_summarizer 

import csv_utility
import userid_map

from operator import add
import socket
import sys

import environment_config as ec

import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAG = "EMS-FLAG"
ENVIRONMENT = socket.gethostname()

sc = ec.get_spark_context(ENVIRONMENT)

def main():
    """
    Driver for the summarization pipeline in the EMS.

    """

    if not len(sys.argv) == 7:
        print("\n\n\nUsage: master_summarization_driver.py: [stream file].json [participant file].json [master summary file path].pkl [mode] [parallelization type]\n")
        print("  * [stream file].json:  path to fson file listing  streams to summarize")
        print("  * [participant file].json:  path to json file listing participant uuids to summarize")
        print("  * [master summary file path].pkl:  path to save master summary (time stamp will be added)")
        print("  * [mode]:  can be 'scratch', 'replace_streams', 'increment_streams','increment_subjects' ")
        print("  * [parallelization type]: can be 'by-subject' or 'single' ")
        print("  * [localtime]: ('true' or 'false') shift marker and labels to subjects' home time zone")
        print("\nYou supplied the argument list:")
        print(sys.argv)
        
        return

    #Get command line arguments
    filename_streams  = sys.argv[1]
    filename_subjects = sys.argv[2]
    master_file       = sys.argv[3]
    mode              = sys.argv[4]
    job_type          = sys.argv[5]
    localtime         = sys.argv[6]
    
    print("Running summarizer with %s %s %s %s %s"%(filename_streams,filename_subjects,master_file,mode,job_type))

    #Check for streams file
    if(os.path.isfile(filename_streams)): 
        with open(filename_streams) as f:
            streams_dict = json.load(f)
            streams = streams_dict['streams']
    else:
        print("Stream file %s does not exist"%(filename_streams))
        exit()

    #Check for subjects file
    if(os.path.isfile(filename_subjects)): 
        with open(filename_subjects) as f:
            subjects_dict = json.load(f)
            subjects      = subjects_dict["subjects"]      
    else:
        print("Subjects file %s does not exist"%(filename_subjects))
        exit()        
    
    #Make sure output director exists
    os.makedirs(os.path.dirname(master_file), exist_ok=True)
    
    #Call the summarizer
    if localtime == 'true':
        master_summarizer.parallel_summarize_master(sc, streams, subjects,  mode, master_file, job_type, localtime=True)

    elif localtime == 'false':
        master_summarizer.parallel_summarize_master(sc, streams, subjects,  mode, master_file, job_type, localtime=False)

    else:
        print("unrecognized value for localtime: {}".format(localtime))

if __name__ == '__main__':
    main()
