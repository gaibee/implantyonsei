import os
import shutil

def make_raw():
    path0 = r'dataset/picture'
    ls_path1 = ['test', 'train', 'val']
    ls_path2 = ['Buttress', 'Reverse_buttress', 'v-shaped']

    for path1 in ls_path1:
        for path2 in ls_path2:
            org_path = os.path.join(path0, path1, path2)

            ls_file = os.listdir(org_path)

            for file in ls_file:
                full_path = os.path.join(org_path, file)

                shutil.copy(full_path, r'dataset/raw')
                print(file)
                
print(len(os.listdir('dataset/raw')))