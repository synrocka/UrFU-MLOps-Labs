import os
import shutil

if os.path.exists("test"):
    shutil.rmtree("./test")

if os.path.exists("train"):
    shutil.rmtree("./train")