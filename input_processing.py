import json

class InputProcessor:
    """Object processing input configuration JSON file and deserializing into InputProcessor object

    Attributes:
        config_file: Received configuration file
        config: Deserialized config object

        src_dir: Source input data directory
        filenames: List of video filenames in src_dir

        dest_dir: Destination output data directory

        mode: Program mode

        data: Received data
    """
    def __init__(self, config_file):
        """
        Arguments:
            config_file: Path to configuration file
        """
        self.config_file = config_file

    def deserialize_input(self):
        """
        Deserializing input JSON file
        """
        with open(self.config_file, "r") as read_file:
            self.config = json.load(read_file)
    
        self.src_dir = self.config["src_dir"]
        self.filenames = self.config["filenames"]

        self.dest_dir = self.config["dest_dir"]

        self.mode = self.config["mode"]

        self.data = self.config["data"]


