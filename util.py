# Copyright 2023 Google LLC

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from contextlib import contextmanager
import gzip
import logging
import os
import tempfile
from typing import List, Optional, Union, Tuple
import xml.etree.ElementTree as ET
import math
import shutil

import consts


def EnsureDirExists(path: str):
    os.makedirs(path, exist_ok=True)


def ClearIfExists(path: str):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
        EnsureDirExists(path)


def GetAllFilesInDirectory(directory, extensions: Union[Tuple[str], str] = tuple()):
    paths = []
    extensions = extensions if type(extensions) is tuple else (extensions,)
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if len(extensions) > 0:
                if all(not filename.endswith(e) for e in extensions):
                    continue
            paths.append(os.path.join(root, filename))
    return paths


def ReadFileAsText(path: str) -> Optional[str]:
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                data = f.read()
            return data
        with open(path, "r") as f:
            data = f.read()
        return data
    except Exception as e:
        logging.debug(f"Error reading file ({path}) as text: {e}")
        return ""


def ParseXml(path: str) -> Optional[ET.Element]:
    xml = ReadFileAsText(path)
    try:
        return ET.fromstring(xml)
    except ET.ParseError as e:
        logging.debug(f"Error parsing xml ({path}): {e}")
        return None


def GetTokensFromXmlRoot(root) -> List[str]:
    if root is None:
        return []

    tokens = []
    stack = [root]

    while stack:
        parent = stack.pop()

        # Remove attributes we don't care about.
        for name, _ in list(parent.attrib.items()):
            if not KeepAttribute(name):
                del parent.attrib[name]

        # Get tokens for this node.
        tokens.extend(GetTokens(parent))

        # Find children while removing certain ones we don't care about.
        # Don't remove these while iterating otherwise the iterator will break.
        # Iterate backwards to make sure we go left to right.
        children_to_remove = []
        for child in reversed(list(iter(parent))):
            if KeepElement(child.tag):
                stack.append(child)
            else:
                children_to_remove.append(child)
        for child in children_to_remove:
            parent.remove(child)

    return tokens


def GetUniqueTokens(path: str):
    return set(GetTokensFromXml(path))


def GetTokensFromXml(path: str) -> List[str]:
    return GetTokensFromXmlRoot(ParseXml(path))


def GetTokens(node: ET.Element) -> List[str]:
    tokens = []
    attrs = []
    for k, v in list(node.attrib.items()):
        attrs.append(f'{k}="{v}"')
    attrs = " ".join(attrs)
    attrs = " " + attrs if attrs else ""
    text = node.text.strip() if node.text else ""
    children = iter(node)
    if text or children:
        n = f"<{node.tag}{attrs}>"
        tokens.append(n)
        if text:
            if text.lstrip("-").isnumeric():
                text = str(TranslateNum(int(text)))
            tokens.append(text)
    else:
        tokens.append(f"<{node.tag}{attrs} />")
    return tokens


def TranslateNum(num: int):
    """Round to nearest power of 2 unless in whitelisted nums.

    This prevents an explosion in the number of 'content' tokens.
    It'd probably be better if the model could view various numeric
    values in the XML as something more than just tokens. But as
    long as we're treating everything as a token, we might as well
    do something like this to constrain the possibilities.
    """
    # Round to nearest power of 2 unless in whitelisted nums.
    if num in consts.WHITELISTED_NUMS:
        return num
    sign = -1 if num < 0 else 1
    return (2 ** round(math.log2(abs(num)))) * sign


def KeepElement(tag_name: str):
    return (
        tag_name not in consts.TAG_NAMES_TO_IGNORE
        and tag_name in consts.ALLOWED_TAG_NAMES
    )


def KeepAttribute(name: str):
    return name not in consts.ATTRIBUTES_TO_IGNORE


@contextmanager
def GetTempDir():
    temp_dir = tempfile.mkdtemp(prefix="xamil", dir="/tmp")
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
