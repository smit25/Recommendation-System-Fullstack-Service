import yaml

def read_versions_from_yaml(filepath = "../version.yaml"):
    """
    Reads 'previous_version' and 'current_version' from a YAML file.

    Args:
        filepath (str): The path to the YAML file.

    Returns:
        tuple or None: A tuple containing (previous_version, current_version),
                       or None if an error occurs or the keys are not found.
    """
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            if data and 'previous_version' in data and 'current_version' in data:
                return data['previous_version'], data['current_version']
            else:
                return None
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def parse_version(version: str):
    return list(map(lambda x: int(x), version.split(".")))

if __name__ == "__main__":
    print(read_versions_from_yaml("./version.yaml"))
    print(parse_version("4.1.0"))