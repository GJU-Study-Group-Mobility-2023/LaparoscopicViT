import os
import argparse

from data import Cholec80

def setup_argparser():
    parser = argparse.ArgumentParser(description='LaparoscopicViT')
    parser.add_argument('--data-rootdir', type=str, required=True, help='Data root directory')
    parser.add_argument('--verify_checksum', action='store_true', help='Verify integrity of downloaded data.')
    
    return parser


def main(args):
    outfile = os.path.join(args.data_rootdir, "cholec80.tar.gz")
    outdir = os.path.join(args.data_rootdir, "cholec80") 
    
    cholec80 = Cholec80(args)


    if not os.path.exists(os.path.join(outdir, outfile)):
        cholec80.download_data()
        cholec80.file_extraction()

    else:
        print('Cholec80 dataset is already downloaded.')


if __name__ == '__main__':
    parser = setup_argparser()
    parsed_args = parser.parse_args()
    # /Volumes/SSD - 2TB
    
    main(parsed_args)