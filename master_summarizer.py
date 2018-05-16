import sys
import os
import os.path
import io
import math
from io import StringIO
from io import BytesIO
import warnings
import numpy as np
import json
# import matplotlib.pyplot as plt
# import mosaic_utils as mu
import datetime as dt
import time
# from features import extract_features
import cc_data_retriever as data_retriever
from random import randint
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from minio import Minio
from minio.error import ResponseError
from cerebralcortex.cerebralcortex import CerebralCortex
from pyspark import SparkContext
from operator import add
# import dev as dev
import socket
import pandas as pd
import traceback
import csv
import datetime

import code

import environment_config as ec

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# LOCALTIME = False

FLAG = "EMS-FLAG"

ENVIRONMENT = socket.gethostname()

cc = ec.get_cc(ENVIRONMENT)
mC = ec.get_minio_client(ENVIRONMENT)


def parallel_summarize_master(sc, streams, subjects,  mode,  master_file, parallelism="all", localtime=False):
    """
    Builds a set of dictionaries that get serialized and passed to Spark jobs to summarize data for
    and experiment.

    Args:
        sc (SparkContext): The SparkContext object to be used to run summarization jobs
        streams (List): Set of all streams to add to the data frame.
        subjects (List): Set of subjects to pull data for.
        mode (str): Start from scratch or add streams/subjects to existing data frame.
        master_file (str): Name of master data file that will be created. 
        parallelism (str): One of the available parallelization schemes
        localtime (bool): Whether or not to shift marker and label streams to subjects' home time zone.
    """

    #sc.setLogLevel(0)

    master_log = []
    job_list = []
    df_list=[]
    all_stream_metadata={}
        
    #Extract streams names from stream list of dictionaries
    stream_names = [x["name"] for x in streams]
        
    #master_file = "master_summary_dataframe.pkl"
    if(os.path.isfile(master_file)):
        master_df = pickle.load( open(master_file, "rb" ))
        print("Found existing master summary data frame for %d subjects and %d features"%(master_df.shape[0],master_df.shape[1]))

        master_log.append(make_log("", "", "", "INFO", "found existing master summary data frame for {} subjects and {} features".format(master_df.shape[0], master_df.shape[1])))

        existing_streams  = list(master_df.columns)
        existing_subjects = list(master_df.index.levels[0]) 
        
        #master file exists, so consider an incremental mode
        if(mode=="scratch" or mode=="test"):
            #Re-compute the master summary dataframe from scratch for the, 
            #throwing out all olf data
            print("Mode=scratch: Re-computing from scratch")

            master_log.append(make_log("", "", "", "INFO", "mode=scratch: re-computing from scratch"))

        elif(mode=="replace_streams"):
            #Re-compute the summaries for the input streams 
            #Will also compute from scratch for any new streams
            #.Only runs on existing subjects
            print("Mode=replace_streams: Replacing old stream computations for existing users")

            master_log.append(make_log("", "", "", "INFO", "mode=replace_streams: replacing old stream computations for existing users"))

            subjects = existing_subjects

        elif(mode=="increment_streams"):
            #Compute summaries for the input streams,
            #skipping streams that already exist. Only
            #operates of existing subjects
            
            #Drop computation of streams that already exist 
            new_stream_names  = list(set(stream_names)-set(existing_streams))
            new_streams = []
            for s in streams:
                if s["name"] in new_stream_names:
                   new_streams.append(s) 
            streams = new_streams
            
            subjects = existing_subjects
            if(len(streams)==0):
                print("All streams have already been computed. No incremental additions.")

                master_log.append(make_log("", "", "", "INFO", "all streams have already been computed: no incremental additions"))

                exit()
            else:
                print("Incrementing streams: ", streams)

                master_log.append(make_log("", "", "", "INFO", "incrementing streams {}".format(streams)))

        elif(mode=="increment_subjects"):
            #Compute summaries for the input subjects,
            #skipping subjects that already exist
            #Only operates on existing streams
            subjects = list(set(subjects)-set(existing_subjects))
            streams  = existing_streams
            if(len(subjects)==0):
                print("All subjects have already been computed. No incremental additions.")

                master_log.append(make_log("", "", "", "INFO", "all streams have already been computed: no incremental additions"))

                exit()
            else:
                print("Incrementing subjects: ", subjects)

                master_log.append(make_log("", "", "", "INFO", "incrementing subjects {}".format(subjects)))
            
        else:
            print("Error: Summarization  mode is not defined")

            master_log.append(make_log("", "", "", "ERROR", "summarization mode is not defined"))

            exit()
    else:
        if mode not in ["test","scratch"]:
            print("Mode is not test or scratch, but master data file does not exist to increment or  replace")
    
    if(mode=="test"):
        #Test mode. Use five good user.
        #5 streams only for debug purposes.
        streams = streams[:20]
        
        subjects = ["622bf725-2471-4392-8f82-fcc9115a3745",
        "d3d33d63-101d-44fd-b6b9-4616a803225d",
        "c1f31960-dee7-45ea-ac13-a4fea1c9235c",
        "7b8358f3-c96a-4a17-87ab-9414866e18db",
        "8a3533aa-d6d4-450c-8232-79e4851b6e11"]

    
    # build up dictionary, write to string, pass to write_..._for_subs...()
    out_list=[]
    for i in range(0, len(subjects)):

        job_dict = {}
        job_dict["subject"] = subjects[i]
        job_dict["streams"] = streams
        job_dict["localtime"] = localtime
        json_string=json.dumps(job_dict)
                
        if(parallelism=="single"):
            out_list.append(parallel_summarize_worker(json_string))
        else:
            job_list.append(json_string)
        
    if(parallelism=="by-subject"):
        summ_rdd = sc.parallelize(job_list, len(job_list))
        job = summ_rdd.map(parallel_summarize_worker)
        out_list = job.collect()

    df_list_data, meta_data_list, subject_logs = zip(*out_list)

    #Combine all meta data dictionaries into 
    #one dictionary. Keys are stream/field ids
    #values are meta data elements
    all_meta_data = {}
    for m in meta_data_list:
        all_meta_data.update(m)

    # process logs -- append to master log, write to CSV, etc.
    master_log.extend(subject_logs) # FIXME: this should already produce a flattened list, shouldn't need next line
    master_log = [item for sublist in master_log for item in sublist]

    # write master log to CSV
    if not os.path.isdir("run_logs"):
        os.makedirs("run_logs")

    with open("run_logs/{}_{}.csv".format("master_summarizer", datetime.datetime.now().strftime("%m-%d-%y_%H:%M")), "w") as f:
        writer = csv.writer(f)
        writer.writerows(master_log)

    #df_data=df_list_data
    df_data = pd.concat(df_list_data , axis=0, keys=subjects)
    df_data.index.levels[1].name = "Date"
    df_data.index.levels[0].name = "Participant"
    
    if(mode=="scratch" or mode=="test"):
        #Re-compute the master summary dataframe from scratch for the, 
        #throwing out all olf data
        master_df = df_data
    elif(mode=="replace_streams"):
        #Re-compute the summaries for the input streams 
        #Will also compute from scratch for any new streams.
        #Only runs on existing subjects
        
        #Drop existing streams 
        stream_intersect = list(set(existing_streams).intersection(stream_names)) 
        
        #Replace old streams and add new streams 
        master_df = master_df.drop(labels=stream_intersect)
        master_df = pd.concat([master_df, df_data], axis=1)
        
    elif(mode=="increment_streams"):
        #Compute summaries for the input streams,
        #skipping streams that already exist
        master_df = pd.concat([master_df, df_data], axis=1)
    elif(mode=="increment_subjects"):
        #Compute summaries for the input subjects,
        #skipping subjects that already exist
        master_df = pd.concat([master_df, df_data], axis=0)
    
    #Write to disk
    
    #Add current timestamp to master file name
    fname,fext = os.path.splitext(master_file)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if(mode=="test"):
        master_file = fname + "-test-" + timestr + fext
    else:
        master_file = fname + "-" + timestr + fext
        
    pickle.dump( {"dataframe": master_df, "metadata":all_meta_data}, open( master_file, "wb" ), protocol=2 )

    try:

        #Write to minio
        object_name = "summary.dataframe"
        bucket_name = "master.summary"
        if not mC.bucket_exists(bucket_name):
            mC.make_bucket(bucket_name)
        bytes = BytesIO()

        # TODO: check with Ben on this change: df --> df_data
        pickle.dump( df_data, bytes, protocol=2 )

        bytes.flush()
        bytes.seek(0)
        mC.put_object(bucket_name, object_name, bytes, len(bytes.getvalue()))
        bytes.close()            
        
        
    except Exception as e:
        print("      ! Warning: Could not save to minio")
        print("-"*50)
        print(traceback.format_exc())
        print(e)
        print("-"*50)
    
        
def parallel_summarize_worker(json_string):

    """
    Handles the actual summarization and generation of the dataframe.  Can be run in parallel as a Spark
    job or single-threaded.

    Args:
        json_string (dict): The parameters for the summarization job: a subject, set of streams, and a localtimestring.

    Returns:
        df (pandas.DataFrame): A dataframe containing all summarized data. 
        all_stream_metadata (pandas.DataFrame): A dataframe containing all collected metadata.
        log_messages (List): A list of messages logged during summarization, to be later written to file.
    """

    log_messages = []

    obj_string = StringIO(json_string)
    obj = json.load(obj_string)
    sub = obj["subject"]
    streams = obj["streams"]    
    localtime = obj["localtime"]   

    offset = 0

    if localtime:
        offset = get_subject_offset(sub)

    offset_timedelta = datetime.timedelta(milliseconds=offset)

    u=sub
    user_streams = cc.get_user_streams(sub)
                
    all_stream_data={}
    all_stream_dfs=[]
    all_stream_metadata={}
        
    #Get the marker streams
    all_days = None
    for s in streams:
        
        #print("    * Attempting to fetch stream %s for user %s"%(s["name"],sub))
        #We'll try to fetch the stream and catch any exceptions
        try:
            
            #Split field out on & character
            stream_with_field = s["name"]
            stream, field = data_retriever.stream_and_field(stream_with_field)

            log_messages.append(make_log(sub, stream, field, "INFO", "attempting to fetch stream for user"))

            # print("stream: {}, field: {}".format(stream, field))

            #If there is no field, set index to -1, else
            #Attempt to retrive field
            if(field == ""):
                field_index = -1
            else:
                #Try to get the index of the field
                #This can fail because the field doesn't exist,
                #the meta data are non-compliant, etc.
                #Will return None if code could detect that the field does not exist
                #Should trigger an exception if there is an uncontrolled failure,
                #which is OK. The stream will be skipped for the user.
                #All values will later be imputed.
                field_index = data_retriever.get_field_index_clean(cc, sub, stream, field)
                
                #Try to catch  incorrect singleton streams specified with value field
                if field_index is None and field=="value":
                    print("      ! Warning: Field name 'value' not found in meta data for stream %s and user %s. Attempting to treat as singleton..."%(stream,sub))
                    log_messages.append(make_log(sub, stream, field, "WARNING", "Field name 'value' not found in meta data. Attempting to treat as singleton"))
                    field_index = -1

            # print("field_index: {}".format(field_index))
    
            #If field_index is none, meta data exists in proper format,
            #but the field name does not exist. We can't get data, so
            #that's it for this stream
            if field_index is None:
                print("      ! Warning: Field name %s not found in meta data for stream %s and user %s"%(field,stream,sub))

                # log = [sub, label_name, "WARNING", "field does not exist for user"]
                log_messages.append(make_log(sub, stream, field, "WARNING", "field does not exist for user"))
        
            else:
                #At this point we should have a valid stream index
                #Check to see if the stream exists for the user 
                if not stream in user_streams:  
                    #Stream doesn't exist. We're done
                    #Skip to the next stream
                    
                    #print("      ! Warning: Stream %s does not exist for user %s"%(stream,sub))

                    log_messages.append(make_log(sub, stream, field, "WARNING", "stream does not exist for user"))
                else:
                    
                    #If we get here the stream exists for the user
                    #and we should have a valid field index.
                    #Streams can be split into chunks
                    #so we need to make sure that we get all chunks     
                    all_data_pairs=[]
                    all_times = []
                    all_end_times = []
                    all_values = []                   
    
                    #Stream_ids holds all the ids for this user and stream name
                    stream_ids = cc.get_stream_id(sub, stream)
                    for id in stream_ids:
                        #Attempt to get the stream, and extract the times and data points
                        stream_id = id["identifier"]
                        this_data = data_retriever.load_cc_data(cc, sub, stream, field, all_days=None)

                        #If we get here, we got the stream data for this chunk, now get the time points
                        #For this chunk and add to what we have
                        if(len(this_data)>0):
                            # all_times = all_times + [x._start_time  for x in this_data ] 
                            # all_end_times = all_end_times + [x._end_time  for x in this_data ]

                            # print("applying offset of {} for subject {}".format(offset_timedelta, sub))

                            all_times = all_times + [x._start_time + offset_timedelta  for x in this_data ] 
                            all_end_times = all_end_times + [x._end_time + offset_timedelta  for x in this_data ]
                                                   
                            if(field_index>=0):
                                if(isinstance(this_data[0]._sample, list)):
                                    if(len(this_data[0]._sample)>field_index):
                                        all_values = all_values + [x._sample[field_index]  for x in this_data ]
                                    else:
                                        log_messages.append(make_log(sub, stream, field, "WARNING", "Field index is greater than the length of the data item"))
                                        print("      ! Warning: Field name %s for stream %s has index %d but data item length is %s"%(field,stream,field_index, len(this_data[0]._sample)))
                                        continue
                                else:
                                    log_messages.append(make_log(sub, stream, field, "WARNING", "Stream includes a field name, but data  point is not a list of values. Grabbing values."))
                                    print("      ! Warning: Field name %s defined for stream %s but data item is not a list. Grabbing values."%(field,stream))
                                    all_values = all_values + [x._sample  for x in this_data ] 
                                    
                            else:
                                all_values = all_values + [x._sample  for x in this_data ]   

            
                    #It'spossible that no data were found at all, check
                    if(len(all_values)==0):
                        #No data, skip creating a data frame for this stream.
                        #Pandas will handle assembling the streams                        
                        #print("      ! Warning: No data in field name %s for stream %s and user %s"%(field,stream,sub))
                        log_messages.append(make_log(sub, stream, field, "WARNING", "no data in field name for stream"))
                        
                    else:                        
                        #Analyze the values
                        if isinstance(all_values[0], str):
                            #Stream has string values -- split into columns by unique strings
                            all_times_array = np.array(all_times)
                            all_end_times_array = np.array(all_end_times) #added

                            all_values_array = np.array(all_values)
                            unique_str = np.unique(all_values)
                            for val_str in unique_str:
                                ind = val_str==all_values_array
                                these_vals = np.ones(np.sum(ind))
                                these_times = all_times_array[ind]
                                these_end_times = all_end_times_array[ind] #added

                                # durations in seconds --> minutes --> hours
                                these_durations = [(end - start).seconds / 60 / 60 for start, end in zip(these_times, these_end_times)]


                                col_name = "%s(%s)"%(stream_with_field,val_str)

                                # print("{} these_durations: {}".format(col_name, these_durations))

                                # df = pd.DataFrame(data=these_vals, index=these_times, columns=[col_name]).resample('D').sum()
                                df = pd.DataFrame(data=these_durations, index=these_times, columns=[col_name]).resample('D').sum()

                                all_stream_dfs.append(df)                                                          
                        
                        else:        
                            #If we have data, use it to  create a data  frame object for this stream
                            df = pd.DataFrame(data=all_values, index=all_times, columns=["%s"%(stream_with_field)]).resample('D').mean()
                            #Add the data  frame to the list of data frames
                            all_stream_dfs.append(df)
                        
                        #print("      + Successful fetch of field name %s for stream %s and user %s"%(field,stream,sub))

                        log_messages.append(make_log(sub, stream, field, "SUCCESS", "successful fetch of field name for stream"))


                        #Got data, now try to get some meta data
                        try:
                            metadata = data_retriever.get_metadata(cc, sub, stream, field)
                            if(metadata is not None):
                                all_stream_metadata[s["name"]] = metadata
                                #print("      + Successful fetch of meta data for field name %s of stream %s and user %s"%(field,stream,sub))
                                log_messages.append(make_log(sub, stream, field, "SUCCESS", "successful fetch of metadata"))
                            else:
                                print("      ! No meta data for field name %s of stream %s and user %s"%(field,stream,sub))
                                log_messages.append(make_log(sub, stream, field, "WARNING", "failed to fetch metadata"))
                    
                        except Exception as e:
                            print("      ! Error: Exception raised feteching metadata for field %s of stream %s and user %s"%(field,stream,sub))
                            print("-"*50)
                            print(traceback.format_exc())
                            print(e)
                            print("-"*50)   

                            log_messages.append(make_log(sub, stream, "ERROR", e, traceback.format_exc()))


        except Exception as e:
            #If anything we didn't catch above happens,
            #we catch it here, dump the stack 
            #but continue on to the next stream.
            print("      ! Warning: Exception raised feteching data in field name %s for stream %s and user %s"%(field,stream,sub))
            print("-"*50)
            print(traceback.format_exc())
            print(e)
            print("-"*50)   

            log_messages.append(make_log(sub, stream, "", e, traceback.format_exc()))

    #Combine all streams for this user into one DF
    if(len(all_stream_dfs)>0):    
        #If at least one data frame was produced, concat all frames together.
        #Pandas deals with aligning days when catting with a time
        #index. Pandas will also deal with  different columns defined
        #when catting across users 
        df = pd.concat(all_stream_dfs , axis=1)
    else:
        #Made it with no errors, but also no data.
        #Write an empty data frame
        print("      ! Warning: no data at all for user %s"%(sub))
        df=pd.DataFrame()

        log_messages.append(make_log(sub, "", "", "WARNING", "no data at all for user"))

    #Return the actual data frame for the user
    #These could get big, maybe better to store in minio
    #later
    
    return(df, all_stream_metadata, log_messages)

def get_subject_offset(subject):
    """
    Utility function: grabs some Qualtrics data for a participant, returns the UTC offset from the label point.

    Args:
        subject (str): The subject whose UTC offset is to be retrieved.

    Returns:
        datapoint.offset (int): The millisecond-precision offset of a subject's home time zone from UTC.
    """

    label_streams = [
        "org.md2k.data_qualtrics_ems.feature.v15.agreeableness.d&value",
        "org.md2k.data_qualtrics.feature.v15.alc.quantity.not_mitre.d&value",
        "org.md2k.data_qualtrics_ems.feature.v15.anxiety.d&value"
    ]

    for stream_name in label_streams:
        stream, field = data_retriever.stream_and_field(stream_name)
        data = data_retriever.load_cc_data(cc, subject, stream, field)

        if len(data) > 0:
            datapoint = data[0]
            
            if datapoint.offset is not None:

                # FIXME: this print is temporary -- disable for actual run
                print("found offset of {} in stream {} for subject {}".format(datapoint.offset, stream_name, subject))

                return datapoint.offset

            else:
                print("stream {} has no offset".format(stream_name))

        else:
            print("no data returned for label {}".format(stream_name))

    return 0

def make_log(subject, stream, field, info, message):
    """
    Utility function for structured logging.
    
    Args:
        subject (str): The subject the message relates to.
        stream (str): The related stream.
        field (str): The field to be logged.
        info (str): Either an error message from exceptions or a level indicator (e.g., "WARN").
        message (str): A detailed description of the event or a stack trace if exception was caught.

    Returns:
        A list of the above of arguments.
    """
    
    return [subject, stream, field, info, message]
    

