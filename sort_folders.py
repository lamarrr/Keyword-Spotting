# move all to raw folder
# move all files in __background_noises_ to noises
# create unknown folder

# Train:Test:Evaluation, 80:5:15
# Randomly cut background_noises to 1000
# Cut Unknown randomly to 10,000 * 5  aug-> 50,000

# use all known classes
# for unknown classes remove up to LEN_UNKNOWN

import glob
import os
import random
import tqdm
from matplotlib import pyplot

BASE_DIR = "speech_commands"
RAW_DIR = "raw"
TARGET_DIR = "sorted"
LEN_UNKNOWN = 12000

print("Started")
classes = [
    "nora", "marvin", "on", "off", "down", "left", "right", "up", "stop",
    "background_noise"
]

# unknown folder not included
all_folders = glob.glob(os.path.join(BASE_DIR, RAW_DIR, "*"))
print("Folders:", all_folders)
assert len(all_folders) > 0, "No Subfolders found in %s" % RAW_DIR

print("Total Folders:", len(all_folders))

print("Creating Folders")
os.mkdir(os.path.join(BASE_DIR, TARGET_DIR))
unknown_dir = os.path.join(BASE_DIR, TARGET_DIR, "unknown")
os.mkdir(unknown_dir)
for _class in classes:
    directory = os.path.join(BASE_DIR, TARGET_DIR, _class)
    os.mkdir(directory)

print("Done")

print("Total Folders: %d" % len(all_folders))

file_counts = dict(tuple((f.split(os.path.sep)[-1], 0) for f in all_folders))

print("Sorting Folders")

uname = "unknown_%d.wav"
count = 0

for folder in tqdm.tqdm(all_folders):
    name = folder.split(os.path.sep)[-1]
    if not (name in classes):

        source = os.path.join(folder, "*.wav")
        sources = glob.glob(source)
        comms = []
        for wav in sources:
            count += 1
            target_path = os.path.join(BASE_DIR, TARGET_DIR, "unknown",
                                       uname % count)
            os.system("mv %s %s" % (wav, target_path))

        os.system("rm -r %s" % folder)

    else:
        target_dir = os.path.join(BASE_DIR, TARGET_DIR, name)
        source = os.path.join(folder, "*.wav")
        os.system("mv %s %s" % (source, target_dir))
        os.system("rm -r %s" % folder)

print("Disposing Files from unknown folder")
# dispose files at random from unknown
unknown_wavs = glob.glob(os.path.join(unknown_dir, "*.wav"))
print("Known Wavs: ", len(unknown_wavs))

print("Disposing Randomly...")
random.shuffle(unknown_wavs)

dispose_wavs = unknown_wavs[LEN_UNKNOWN:]
print("To Dispose: ", len(dispose_wavs))

for wav in tqdm.tqdm(dispose_wavs):
    os.system("rm %s" % wav)

print("Done")
print("Removing RAW folder")
os.system("rm -r %s/%s" % (BASE_DIR, RAW_DIR))
print("Done")

print("All Folders Sorted")
