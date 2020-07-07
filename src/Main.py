import zipfile
import py7zr
import argparse
import os
import glob
import shutil
from Reader import Reader

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

# parser = argparse.ArgumentParser()
# parser.add_argument('zipfile', help='Path to config file')
# args = parser.parse_args()

def main():
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #
    # with open(args.zipfile) as file:
    #     print(file)
    #
    # if args.zipfile[-4:] == '.zip':
    #     with zipfile.ZipFile(args.zipfile,"r") as zip_ref:
    #         zip_ref.extractall('extracted')
    # elif args.zipfile[-3:] == '.7z':
    #     archive = py7zr.SevenZipFile(args.zipfile, mode='r')
    #     archive.extractall('extracted')
    #     archive.close()
    # else:
    #     print("Cannot read the zip file")

    dirName = 'extracted';
    listOfFiles = getListOfFiles(dirName)
    reader = Reader()

    for elem in listOfFiles:
        print(elem)
        reader.predict(elem)

    files = glob.glob('extracted/*')

    print('---------------------NOW DELETE------------------------')

    for elem in listOfFiles:
        os.remove(elem)
    for f in files:
        print(f)
        try:
            shutil.rmtree(f)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == "__main__":
    main()
