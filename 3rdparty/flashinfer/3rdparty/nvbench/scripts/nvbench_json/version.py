file_version = (1, 0, 0)

file_version_string = "{}.{}.{}".format(file_version[0],
                                        file_version[1],
                                        file_version[2])


def check_file_version(filename, root_node):
    try:
        version_node = root_node["meta"]["version"]["json"]
    except KeyError:
        print("WARNING:")
        print("  {} is written in an older, unversioned format. ".format(filename))
        print("  It may not read correctly.")
        print("  Reader expects JSON file version {}.".format(file_version_string))
        return

    # TODO We could do something fancy here using semantic versioning, but
    # for now just warn on mismatch.
    if version_node["string"] != file_version_string:
        print("WARNING:")
        print("  {} was written using a different NVBench JSON file version."
              .format(filename))
        print("  It may not read correctly.")
        print("  (file version: {} reader version: {})"
              .format(version_node["string"], file_version_string))
