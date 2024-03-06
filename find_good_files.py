# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from multiprocessing import Pool, Lock
import xml.etree.ElementTree as ET
import argparse
from absl import app, flags

flags.DEFINE_string(
    "xml_folder_root",
    None,
    "The maximum number of paths to process. If None, all paths will be processed.",
)

flags.mark_flag_as_required("xml_folder_root")


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
        xml = util.ParseXml(path)
        if has_one_part(xml) and has_one_staff(xml):
            _WRITE_LOCK.acquire()
            with open(consts.TRAINING_FILES_LIST, "a") as f:
                f.write(path + "\n")
            _WRITE_LOCK.release()
    except:
        pass


def Main(argv):
    # Prepare the output directory.
    util.EnsureDirExists(consts.MISC_FILES_ROOT)
    util.ClearIfExists(consts.TRAINING_FILES_LIST)

    paths = util.GetAllFilesInDirectory(
        flags.FLAGS.xml_folder_root, extensions=consts.ANY_XML_FILE
    )
    with Pool(consts.PARALLELISM) as p:
        p.map(write_if_one_staff, paths)


if __name__ == "__main__":
    app.run(Main)
