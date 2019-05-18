import numpy as np
import pandas as pd
import argparse 
import wfdb
import h5py 

def data_preparation(reference):
    SAMPLING_RATE = 300
    WINDOW_SIZE = 30 * SAMPLING_RATE
    classes = ['A','N','O','~']

    data = np.zeros((reference.shape[0], WINDOW_SIZE, 1))
    labels = np.zeros((reference.shape[0], len(classes)), dtype=np.int32)

    for i, (rec,label) in enumerate(zip(reference['record'], reference['class'])):   
        record = wfdb.rdrecord(args.path+"%s" % rec)
        signal = record.p_signal
        
        # Select the first 30s
        signal = signal[:min(WINDOW_SIZE, signal.shape[0])]

        # Normalization
        signal = (signal - np.mean(signal)) / np.std(signal)
        data[i, :min(WINDOW_SIZE, signal.shape[0])] = signal
    
        # One hot encode 
        labels[i, classes.index(label)] = 1

    return data, labels


def main(args):

    reference = pd.read_csv(args.path+'REFERENCE.csv', nrows=args.num_rows, header=None)
    reference = reference.sample(frac=1.).reset_index(drop=True)
    reference.columns = ['record', 'class']

    print('Creating dataset...')
    data, labels = data_preparation(reference)

    # save as hdf5
    hf = h5py.File(args.h5_path, 'w')

    hf.create_dataset('data', data=data)
    hf.create_dataset('labels', data=labels)

    hf.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path',
        type=str,
        help='path to the WFDB data'
    )

    parser.add_argument(
        '--num-rows',
        type=int,
        default=10000,
        help='number of rows to read'
    )

    parser.add_argument(
        '--h5-path',
        type=str,
        help='path to save the dataset as hdf5 file'
    )

    args = parser.parse_args()
    main(args)



