import os
directory = 'simon'
files = [os.path.join(directory, f) for f in os.listdir(directory)]

output_file = open('all_tex_files.txt','w')

for file in files:
    with open(file, 'r') as input_file:
        for line in input_file.readlines():
            output_file.write(line)
