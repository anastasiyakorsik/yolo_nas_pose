import time

from tqdm import tqdm
from typing import Type

from input_processing import InputProcessor
from output_processing import OutputProcessor, prediction_processing
                

def inference_video(model, filename, conf = 0.6):
    """
    Inferencing video
    Arguments:
        model: Yolo-NAS Pose model
        filename (str): Name of video file
        conf (float) [OPTIONAL]: Confidence threshold
    Returns:
        Processed model predictions for video (dict)
        Inference time (float)
    """
    try:
        preds = []
        start_time = time.time()

        model_prediction = model.predict(filename, conf=conf)
        raw_prediction = [res for res in tqdm(model_prediction._images_prediction_gen, total=model_prediction.n_frames, desc="Processing Video")]

        for i in range(len(raw_prediction)):
            prediction = {}
            key = f"frame_#{i}"
            prediction["markup_frame"] = key
            prediction["chain_markups"] = prediction_processing(raw_prediction[i])
            preds.append(prediction)

        end_time = time.time()

        inference_time = end_time - start_time

        return preds, inference_time
    except Exception as err:
        print(f"ERROR - Exception occured in inference_video() {err=}, {type(err)=}")
        raise

def processing_videos(model, filenames, conf = 0.6):
    """
    Inferencing videos from dataset

    Arguments:
        model: YOLO-NAS Pose model
        filenames (list): List of videos filenames
        conf (float) [OPTIONAL]: Confidence threshold

    Returns:
        Dictionary with videos inference results
    """
    try: 
        res = []
        print("INFO - Start inferencing videos")
        for filename in filenames:
            inner_res = {}
            print(f"INFO - Inferencing video {filename}")
            preds, inf_time = inference_video(model, filename, conf)

            print(f"inference_time: {inf_time}")

            inner_res["file_name"] = filename
            inner_res["file_chains"] = preds

            res.append(inner_res)
        return res

    except Exception as err:
        print(f"ERROR - Exception occured in process_videos() {err=}, {type(err)=}")
        raise

def inference_mode(output_json_name, model, config: Type[InputProcessor]):
    """
    Inference program mode. Starts processing videos from dataset 
    and creates output JSON configuration file

    Arguments:
        output_json_name: Output JSON file name
        model: YOLO-NAS Pose downloaded model
        config: Input configuration object
    """
    try:
        # Starting inference process for videos
        result = processing_videos(model, config.filenames, 0.6)
        # print(result)

        # Creating output configuration file
        out_res = OutputProcessor(output_json_name, result)
        out_res.produce_config()
        out_res.serialize_output()

        print(f"INFO - Inference results saved and recorded into {output_json_name}")
    except Exception as err:
        print(f"ERROR - Exception occured in inference.inference_mode() {err=}, {type(err)=}")
        raise
