import json

class OutputProcessor:
    """Object processing input configuration JSON file and deserializing into InputProcessing object

    Attributes:
        config_file: Produced configuration file
        data: Serializable model predictions
        config: Serializable config object
    """
    def __init__(self, config_file, data):
        """
        Arguments:
            config_file: Path to produced configuration file
            data: Serializable predictions containing data produced by model
        """
        self.config_file = config_file
        self.data = data
    
    def produce_config(self):
        """
        Collect output JSON 
        """
        config = {}
        config["files"] = self.data

        self.config = config

    def serialize_output(self):
        """ Creates JSON output config file """
        with open(self.config_file, "w") as write_file:
            json.dump(self.config, write_file, indent=4)


def prediction_processing(raw_frame_pred):
    """
    Processing raw model prediction for one frame in video into dict
    Arguments:
        raw_frame_pred: Model prediction for one frame in video
    Returns:
        Processed frame prediction
    """
    try:
        poses = raw_frame_pred.prediction.poses
        scores = raw_frame_pred.prediction.scores
        bboxes = raw_frame_pred.prediction.bboxes_xyxy
        edge_links = raw_frame_pred.prediction.edge_links

        boxes = []
        for i in range(len(bboxes)):
            box = {}
            box_coords = {}

            bbox = bboxes[i]

            box_coords["score"] = float(scores[i])
            box_coords["x"] = float(bbox[0])
            box_coords["y"] = float(bbox[1])
            box_coords["width"] = float(bbox[2] - bbox[0])
            box_coords["height"] = float(bbox[3] - bbox[1])

            box[f"bbox_{i}"] = box_coords
            boxes.append(box)

        nodes = []
        for i in range(len(poses)):
            
            person = poses[i]

            for j in range(len(person)):

                pose = person[j]
                node = {}
                node_coords = {}
                node_coords["x"] = float(pose[0])
                node_coords["y"] = float(pose[1])
                node_coords["score"] = float(pose[2])

                node[f"node_{j}"] = node_coords
                nodes.append(node)

        edges = []
        for i in range(len(edge_links)):
            edge = {}
            edge_nodes = {}

            link = edge_links[i]

            edge_nodes["from"] = int(link[0])
            edge_nodes["to"] = int(link[1])

            edge[f"edge_{i}"] = edge_nodes
            edges.append(edge)  

        markup_vector = {}
        markup_vector["nodes"] = nodes
        markup_vector["edges"] = edges        

        pr_frame_pred = {}
        pr_frame_pred["markup_path"] = boxes
        pr_frame_pred["markup_vector"] = markup_vector


        return pr_frame_pred
    except Exception as err:
        print(f"ERROR - Exception occured in prediction_processing() {err=}, {type(err)=}")
        raise