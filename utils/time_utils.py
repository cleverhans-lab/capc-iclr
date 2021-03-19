from datetime import datetime


def get_timestamp():
    dateTimeObj = datetime.now()
    # timestampStr = dateTimeObj.strftime("%Y-%B-%d-(%H:%M:%S.%f)")
    timestampStr = dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S-%f")
    return timestampStr


def log_timing(stage: str, log_file: str):
    with open(log_file, 'a') as f:
        f.write(f"{get_timestamp()}: {stage}\n")