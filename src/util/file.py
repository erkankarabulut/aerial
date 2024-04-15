import yaml


def get_algorithm_parameters():
    with open("src/parameters.yaml") as stream:
        try:
            parameters = yaml.safe_load(stream)
            return parameters
        except yaml.YAMLError as exc:
            print("An error occurred while reading the 'parameters.yaml' file:", exc)
            return None
