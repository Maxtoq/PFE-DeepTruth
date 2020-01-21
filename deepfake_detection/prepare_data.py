import os
import argparse


def init_output_dir(output_dir):
    if not os.path.exists(os.path.join(output_dir, 'train')):
        os.makedirs(os.path.join(output_dir, 'train'))
    if not os.path.exists(os.path.join(output_dir, 'test')):
        os.makedirs(os.path.join(output_dir, 'test'))

def check_source_dir(source_dir):
    if os.listdir(source_dir) != ['manipulated', 'originals']:
        print('ERROR: Source directory must contain only two directories: \'manipulated\' and \'originals\'.')
        exit(0)


def create_data(source_dir, output_dir, label, train_split=0.8, nb_sample=1):
    print(os.listdir(source_dir), label)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for training and testing a 3d Convolutional model.')
    parser.add_argument('-s', '--source', help='-s <source_dir_path> : source directory containing \'originals\' and \'manipulated\' directories',
                        default=None, type=str)
    parser.add_argument('-o', '--output', help='-o <output_dir_path> : output directory (must be empty)',
                        default=None, type=str)
    args = parser.parse_args()
    source_dir = args.source
    output_dir = args.output

    check_source_dir(source_dir)
    
    init_output_dir(output_dir)

    create_data(os.path.join(source_dir, 'originals'), output_dir, 0, nb_sample=4)
    create_data(os.path.join(source_dir, 'manipulated'), output_dir, 1)