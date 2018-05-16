import numpy as np
import time
import json
from datetime import datetime
from datetime import date, timedelta

import traceback

#LOCALTIME = False
FLAG = "EMS-FLAG"

def load_data(cc, subject, marker, field, target=None, days=None, check=False, mC=None, run_id=None):
    """
    Pass-through function.  Accounts for the possibility of various data sources.

    :param CerebralCortex cc: CerebralCortex instance
    :param str subject: uuid of subject whose data is being retrieved
    :param str marker: Name of marker stream to retrieve
    :param str field: Name of field to return for compound DataPoint.sample types
    :param str target: Name of prediction target, can be used to ignore days in which label
        data isn't available (currently unused)
    :param List(str) days: Explicit list of days to retrieve data for

    :return: List of DataPoint objects
    :rtype: List(DataPoint)
    """
    return load_cc_data_clean(cc, subject, marker, field, target, days)

def load_cc_data_clean(cc, subject, marker, field, target=None, all_days=None, check=False, mC=None, run_id=None,local_flag=False):
    """
    Primary means of retrieving data from CerebralCortex.  Uses CerebralCortex functions
    to retrieve data for a particular subject-stream combination, combining data from
    one or more days into a single list.

    :param CerebralCortex cc: CerebralCortex instance
    :param str subject: uuid of subject whose data is being retrieved
    :param str marker: Name of marker stream to retrieve
    :param str field: Name of field to return for compound DataPoint.sample types
    :param str target: Name of prediction target, can be used to ignore days in which label
        data isn't available (currently unused)
    :param List(str) all_days: Explicit list of days to retrieve data for

    :return: List of DataPoint objects
    :rtype: List(DataPoint)
    """
    full_stream = []

    # print("+"*40 + all_days)

    if all_days is None or all_days == "all":
            # print("dr.load_cc_data: setting days to 'all'")
            # all_days = available_dates_for_stream(cc, marker_id)
            all_days = available_dates_for_user_and_stream_name(cc, subject, marker)

    if check:
        streams = cc.get_user_streams(subject)
        if len(streams) > 0 and marker not in streams:
            print("data retriever: stream {} not available for user {}".format(marker, subject))
            return full_stream

    #FIXME: this can also be handled higher up -- get stream uuids, pass to data retriever
    marker_ids = cc.get_stream_id(subject, marker)

    for id in marker_ids:
        marker_id = id["identifier"]

        for d in all_days:

            try:
                marker_stream = cc.get_stream(marker_id, subject, d, localtime=local_flag)
                full_stream.extend(marker_stream.data)

            except Exception as err:
                # logger.log_message_for_subject(mC, subject, err, run_id)
                print(err)

            # print("found {} values for stream {}".format(len(full_stream), marker))

    #FIXME: this indentation doesn't look right...
    return full_stream

def load_cc_data(cc, subject, marker, field, target=None, all_days=None, check=False, mC=None, run_id=None,local_flag=False):
    """
    Primary means of retrieving data from CerebralCortex.  Uses CerebralCortex functions
    to retrieve data for a particular subject-stream combination, combining data from
    one or more days into a single list.

    :param CerebralCortex cc: CerebralCortex instance
    :param str subject: uuid of subject whose data is being retrieved
    :param str marker: Name of marker stream to retrieve
    :param str field: Name of field to return for compound DataPoint.sample types
    :param str target: Name of prediction target, can be used to ignore days in which label
        data isn't available (currently unused)
    :param List(str) all_days: Explicit list of days to retrieve data for

    :return: List of DataPoint objects
    :rtype: List(DataPoint)
    """
    full_stream = []

    # print("+"*40 + all_days)

    if all_days is None or all_days == "all":
            # print("dr.load_cc_data: setting days to 'all'")
            # all_days = available_dates_for_stream(cc, marker_id)
            all_days = available_dates_for_user_and_stream_name(cc, subject, marker)

            # print("|"*20 + " all_days: {}".format(list(all_days)))

    if check:
        streams = cc.get_user_streams(subject)
        if len(streams) > 0 and marker not in streams:
            print("data retriever: stream {} not available for user {}".format(marker, subject))
            return full_stream

    # print("found marker {} in streams for user {}".format(marker, subject))

    #FIXME: this can also be handled higher up -- get stream uuids, pass to data retriever
    marker_ids = cc.get_stream_id(subject, marker)

    for id in marker_ids:
        marker_id = id["identifier"]

        # print("data retriever: load_cc_data: {}: {}".format(marker, marker_id))

        for d in all_days:

            # print("getting stream {} for subject {} and day {}".format(marker_id, subject, d))

            try:
                marker_stream = cc.get_stream(marker_id, subject, d, localtime=local_flag)
                full_stream.extend(marker_stream.data)

            except Exception as err:
                # logger.log_message_for_subject(mC, subject, err, run_id)
                print(err)

            # print("found {} values for stream {}".format(len(full_stream), marker))

        #FIXME: this indentation doesn't look right...
        return full_stream

def available_dates_for_user(cc, subject_id, streams=None):
    """
    Discovers all dates for which a user might potentially have collected data.  Loops through
    all available uuids for all available streams for the specified user; durations for each
    uuid are gotten from CerebralCortex.get_stream_duration().  These durations are converted to
    lists of explicit date strings using the dates_for_stream_between_start_and_end_times() utility
    function.

    :param CerebralCortex cc: CerebralCortex instance for accessing user streams
    :param uuid subject_id: uuid of subject whose stream durations will be queried for
    :param List(uuid) streams: explicit list of stream uuids to query for durations

    :return: List of all dates in which a might have collected data
    :rtype: List(str)
    """

    all_dates = []

    if not streams:
        streams = cc.get_user_streams(subject_id)

    for s in streams:

        if not (('data_analysis' in s) or ('data_qualtrics' in s)):
            # print("dismissing stream {} from date discovery".format(s))
            continue

        stream_ids = cc.get_stream_id(subject_id, s)

        # print("retriever.available_dates_for_user: user: {}, stream: {}, stream_ids: {}".format(subject_id, s, list(stream_ids)))

        for id in stream_ids:

            stream_id = id["identifier"]

            duration = cc.get_stream_duration(stream_id)

            # print("-" * 30)

            # print("stream {} with uuid {} for user {} duration: {}".format(s, stream_id, subject_id, duration))

            stream_dates = dates_for_stream_between_start_and_end_times(duration["start_time"], duration["end_time"])

            for sd in stream_dates:
                if not sd in all_dates:
                    all_dates.append(sd)

    return all_dates

def available_dates_for_user_and_stream_name(cc, user_id, stream_name, check=False):
    """
    Discovers all available dates for a stream based on its name and user ID.  A conveniece wrapper around
    available_dates_for_stream().

    Args:
        cc (CerebralCortex): Instance of CerebralCortex.
        user_id (str): UUID of the stream's owner.
        stream_name (str): Name (not UUID) of the stream whose dates are being discovered.
        check (boolean): Force a safety check for the stream within a user's available streams.

    Returns:
        dates (List): A list of all available dates for the stream.
    """

    dates = []

    if check:
        user_streams = cc.get_user_streams(user_id)

        if stream_name not in user_streams:
            print("data retriver: stream {} not available for user {}".format(stream_name, user_id))
            return dates

    stream_ids = cc.get_stream_id(user_id, stream_name)

    for id in stream_ids:
        stream_uuid = id["identifier"]

        for d in available_dates_for_stream(cc, stream_uuid):
            if d not in dates:
                dates.append(d)

    return dates

def available_dates_for_stream(cc, stream_id):
    """
    Discovers all available dates within a stream's duration.

    :param CerebralCortex cc: CerebralCortex instance
    :param str stream_id: uuid of stream to retrieve dates for

    :return: Explicit list of string representations of all dates within the given stream's duration
    :rtype: List(str)
    """
    all_days = []

    stream_duration = cc.get_stream_duration(stream_id)

    if stream_duration is None:
        print("no duration data available for stream ID " + str(stream_id))

    else:
        stream_start_time = stream_duration["start_time"]
        stream_end_time = stream_duration["end_time"]

        stream_start = datetime(stream_start_time.year, stream_start_time.month, stream_start_time.day)
        stream_end = datetime(stream_end_time.year, stream_end_time.month, stream_end_time.day)

        stream_interval = stream_end - stream_start

        number_of_days = stream_interval.days + 1 # add 1 to capture first and last days

        for i in range(0, number_of_days):
            all_days.append((stream_start + timedelta(days=i)).strftime("%Y%m%d"))

    return all_days

def dates_for_stream_between_start_and_end_times(stream_start, stream_end):
    """
    Generates list of explicit dates within a stream's duration

    :param datetime stream_start: Start of a stream's duration
    :paramt datetime stream_end: End of a stream's duration

    :return: Explicit list of string representations of dates between two datetime objects
    :rtype: List(str)
    """

    dates = []

    stream_interval = stream_end - stream_start

    number_of_days = stream_interval.days + 1 # add 1 to capture first and last days

    for i in range(0, number_of_days):
        dates.append((stream_start + timedelta(days=i)).strftime("%Y%m%d"))

    return dates
    
def get_field_index_clean(cc, subject, stream_name, field_name):
    """
    Discovers the index in an ordered list of a particular field within a DataPoint.sample.
    
    Args:
        cc (CerebralCortex): Instance of CerebralCortex.
        user_id (str): UUID of the stream's owner.
        stream_name (str): Name (not UUID) of the stream whose dates are being discovered.
        field_name (str): Name of the field to be located within the ordered list.

    Returns:
        i (int): Index of the field within the ordered list.  (Or None if the index can't be found.)
    """

    #Check to see if using numeric field index
    try:
       i=int(field_name)
       return(i)
    except:
        pass

    try:
        stream_ids = cc.get_stream_id(subject, stream_name)
        if(len(stream_ids)==0):
            return  None

        metadata = cc.get_stream_metadata(stream_ids[0]["identifier"])              
        for m in metadata:
            descriptor = json.loads(m["data_descriptor"])
            for i in range(0, len(descriptor)):
                if ("name" in descriptor[i]) and (descriptor[i]["name"] == field_name):
                    return i

    except Exception as e:
        print("      + Warning: Exception raised feteching index for field name %s for stream %s"%(field_name,stream_name))
        print("-"*50)
        print(traceback.format_exc())
        print(e)
        print("-"*50) 
        return None

    return None

def get_metadata(cc, subject, stream_name, field_name):
    """
    Retrieves the data descriptor from the metadata for a particular user's stream.

    Args:
        cc (CerebralCortex): Instance of CerebralCortex.
        user_id (str): UUID of the stream's owner.
        stream_name (str): Name (not UUID) of the stream whose dates are being discovered.
        field_name (str): Name of the field to be located within the ordered list.

    Returns:
        md (List or dict): The data descriptor portion of the metadata.
    """

    stream_ids = cc.get_stream_id(subject, stream_name)
    if(len(stream_ids)==0):
        return  None
    metadata = cc.get_stream_metadata(stream_ids[0]["identifier"])
    
    #If field name is an int, try to get
    #meta data at that index
    try:
        i=int(field_name)
        descriptor=json.loads(metadata[0]["data_descriptor"])
        return descriptor[i]
    except:
        pass        
    
    if(field_name==""):
        #if isinstance(metadata, (list,)):
        md = json.loads(metadata[0]["data_descriptor"])
        if isinstance(md, (list,)):
            return(md[0])# should be a singleton list if it's a list
        else:
            return(md) #If not a list, should just be a dict
        #else:
        #    return json.loads(metadata["data_descriptor"])
    else:  
        for m in metadata:
            descriptor = json.loads(m["data_descriptor"])
            for i in range(0, len(descriptor)):
                if ("name" in descriptor[i]) and (descriptor[i]["name"] == field_name):
                    return descriptor[i]

    return None

def stream_and_field(stream_with_field):
    """
    Parses a field name that's been appended to the end of  a stream name.

    Args:
        stream_with_field (str): Stream name with field name appended.

    Returns:
        stream_name (str): Name of the stream with the field and separator stripped off.
        field_name (str): Name of the field with stream and separator stripped off.
    """

    stream_and_field = stream_with_field.split("&")

    if len(stream_and_field) == 1:
        return stream_with_field, ""

    stream_name = stream_and_field[0]
    field_name = stream_and_field[1]

    return stream_name, field_name