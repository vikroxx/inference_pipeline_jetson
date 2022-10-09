import os
import shutil
import argparse


def lets_duplicate_dir(orig_dir, cropouts_dir, final_dir):
    list_subdir = os.listdir(orig_dir)
    # print(list_subdir)
    print('Creating sub directories in final folder')
    for subdir in list_subdir:
        if not os.path.isdir(os.path.join(final_dir, subdir)):
            os.mkdir(os.path.join(final_dir, subdir))
        print(os.path.join(final_dir, subdir))
        files = (os.listdir(os.path.join(orig_dir, subdir)))
        if len(files):
            for file in files:
                shutil.copy(os.path.join(cropouts_dir, file), os.path.join(final_dir, subdir, file))


def main():
    parser = argparse.ArgumentParser(description='duplicate directory structure with original crop outs')
    parser.add_argument('--crops_dir', type=str, help='path of the original crops directory')
    parser.add_argument('--orig_dir', type=str, help='path of the directory to duplicate with original crops')

    args = parser.parse_args()
    crops_dir = args.crops_dir
    orig_dir = args.orig_dir
    final_dir = args.orig_dir + '_original'

    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    lets_duplicate_dir(orig_dir, crops_dir, final_dir)


if __name__ == '__main__':
    main()
