from multiprocessing import Pool, Lock
import xml.etree.ElementTree as ET
import argparse


import util
import consts

_WRITE_LOCK = Lock()


def print_node(node):
    print(ET.tostring(node).decode())


def has_one_part(xml: ET.Element):
    try:
        pl = xml.find("part-list")
        return len(list(iter(pl))) == 1
    except:
        return False


def has_one_staff(xml: ET.Element):
    return xml.find("./part/measure/note/staff") is None


def write_if_one_staff(path: str):
    try:
        xml = util.parse_xml(path)
        if has_one_part(xml) and has_one_staff(xml):
            _WRITE_LOCK.acquire()
            with open(consts.TRAINING_FILES_LIST, "a") as f:
                f.write(path + "\n")
            _WRITE_LOCK.release()
    except:
        pass


def run(dir: str):
    # Prepare the output directory.
    util.ensure_dir_exists(consts.MISC_FILES_ROOT)
    util.clear_if_exists(consts.TRAINING_FILES_LIST)

    paths = util.get_all_files_in_directory(dir, extensions=consts.ANY_XML_FILE)
    with Pool(consts.PARALLELISM) as p:
        p.map(write_if_one_staff, paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--xml_folder_root", type=str, required=True)
    args = parser.parse_args()
    run(args.xml_folder_root)
