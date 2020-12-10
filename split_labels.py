import os

txt_path = 'data/txt_labels'
files = list(sorted(os.listdir(os.path.join(txt_path))))


for file in files:
    f = open(os.path.join(txt_path, file), 'r')
    new_file = open(os.path.join('data/txt_num', file), 'w')
    txt = ''
    for line in f:
        id_label, x, y, w, h = line.split()
        if int(id_label) < 13:
            txt += line
    if not txt:
        print(file)
    new_file.write(txt)
    new_file.close()
    f.close()

