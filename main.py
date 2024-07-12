import sys
import torch

from super_gradients.training import models
from inference import inference_mode
from input_processing import InputProcessor

def main():
    try:
        if len(sys.argv) < 2:
            raise ValueError('ERROR - Please provide path to configuration file')
        
        config_file = sys.argv[1]

        print("INFO - Getting model:")
        model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        config = InputProcessor(config_file)
        config.deserialize_input()

        if config.mode == "inference":
            output_json_name = f"yolo-nas-pose-{config.mode}.json"
            inference_mode(output_json_name, model, config)
        elif config.mode == "additional_training":
            output_json_name = f"yolo-nas-pose-{config.mode}.json"
            print(output_json_name)
        else:
            print(f"ERROR - Incorrect program mode in config_file {config_file}")
            
    except Exception as err:
        print(f"ERROR - Exception occured in main() {err=}, {type(err)=}")
        raise


if __name__ == "__main__":
    main()