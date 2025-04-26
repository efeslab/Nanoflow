import sys
import re
import os


# get all folders in the current directory
folders = [f for f in os.listdir('.') if os.path.isdir(f)]
for folder in folders:
    filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]
    for filename in filenames:
        path = folder + '/' + filename
        file = open(path, 'r')
        lines = file.readlines()
        file.close()
        print(path)
        file = open(path, 'w')

        # replace consequtive space with ','
        for line in lines:
            modified_line = ','.join(line.split())
            modified_line_without_parentheses = re.sub(r'\(.*?\)', '', modified_line)
            file.write(modified_line_without_parentheses+ "\n")