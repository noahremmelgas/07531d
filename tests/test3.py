import os
import csv
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
import pandas as pd
from numba import njit
from tensorflow.keras import mixed_precision

# Set the mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

tf.config.optimizer.set_jit(True)


filenames = [['loc_30deg_1m', 1], ['loc_30deg_2m', 2], ['loc_30deg_3m', 3], ['loc_30deg_4m', 4], ['loc_30deg_5m', 5], 
             ['loc_minus60deg_1m', 1], ['loc_minus60deg_2m', 2], ['loc_minus60deg_3m', 3], ['loc_minus60deg_4m', 4], ['loc_minus60deg_5m', 5]]

def convert_to_array(filename):
    f = h5py.File('wifi_csi_data_loc/' + filename + '.mat', 'r')
    data = f.get('csi_complex_data')
    data = np.array(data)
    print(f"Loaded data for {filename}: shape {data.shape}")
    return data
    
@njit(cache=True)
def convert_to_complex(data):
        """Converts real/imaginary data to complex numbers using NumPy and Numba."""
        num_time_steps = len(data)
        num_subcarriers = 30
        num_antennas = 3

    # Preallocate output array for efficiency
        temp_time = np.zeros(( num_time_steps, num_subcarriers, num_antennas), dtype=np.complex128)

        for m_idx, m in enumerate(range(0, num_time_steps)):
            for n in range(num_subcarriers):
                for o in range(num_antennas):
                    p = data[m][n][o]
                    temp_time[m_idx, n, o] = p[0] + 1j * p[1]

        print(f"Converted data to complex: shape {np.shape(temp_time)}")
        return temp_time

def save_as_npz(filenames):
    for filename in filenames:
        data = convert_to_array(filename)
        data = convert_to_complex(data)

        npz_filename = f"npz_files/{filename}.npz"
        np.savez(npz_filename, data)
        print(f"Saved: {npz_filename}")


def add_labels(data, labels):
    real_part = data['arr_0'].real
    imag_part = data['arr_0'].imag
    combined_data = np.concatenate((real_part, imag_part), axis=-1)  # Shape: (time_steps, 30, 6)
    print(np.shape(combined_data))

    adjusted_labels = labels

    dataset = tf.data.Dataset.from_tensor_slices((combined_data, np.full(len(combined_data), adjusted_labels)))
    return dataset

def create_test_train_datasets(dataset, train_percentage = 0.5, test_percentage = 0.1):

    dataset_size = len(list(dataset))
    train_size = int(dataset_size * train_percentage)
    test_size = int(dataset_size * test_percentage)

    dataset = dataset.shuffle(buffer_size=dataset_size)
    print("Creating testing and training datasets")
    # Create train and test datasets
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    
    return train_dataset, test_dataset

def combine_datasets(filenames):
    combine_train_ds = None
    combine_test_ds = None
    print('Creating testing and training datasets')
    for filename, label in filenames:
        print(label)
        data = np.load(f"npz_files/CSI_EFF_{filename}.npz")
        dataset = add_labels(data, label)
        print('Labels added')
        train_ds, test_ds = create_test_train_datasets(dataset)
        
        if combine_train_ds is None:
            combine_train_ds = train_ds
            combine_test_ds = test_ds
        else:
            combine_train_ds = combine_train_ds.concatenate(train_ds)
            combine_test_ds = combine_test_ds.concatenate(test_ds)

    
    return combine_train_ds, combine_test_ds

def train_model(train_data, test_data, epochs = 100, batch_size = 64):

    model = keras.Sequential([
        keras.layers.Conv1D(16, kernel_size=3, activation='relu', input_shape=(30, 6)),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, dtype='float32')  # For binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mean_squared_error',
              metrics=['mae'])

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop after 5 epochs without improvement
    restore_best_weights=True
)


    model.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=test_data,
        callbacks=[lr_scheduler, early_stopping]
    )

    return model

def prediction(filename):
    input_data = np.load(f"npz_files/{filename}.npz")

    # Preprocess the input data to combine real and imaginary parts
    real_part = input_data['arr_0'].real
    imag_part = input_data['arr_0'].imag
    combined_data = np.concatenate((real_part, imag_part), axis=-1)  # Shape: (time_steps, 30, 6)

    # Add a batch dimension to match the model's expected input shape
    combined_data = np.expand_dims(combined_data, axis=0)  # Shape: (1, time_steps, 30, 6)

    # Load the trained model
    model = tf.keras.models.load_model('my_model.keras')
    result_array = np.array([])

    # Make predictions
    for i in range(100):
        result = model.predict(combined_data[0][i:i+1])
        result_array = np.append(result_array, result)
        
    print(np.mean(result_array))

def main():
    print('Creating datasets')
    train_ds, test_ds = combine_datasets(filenames)

    print('Saving datasets')
    train_ds.save('datasets/train_ds')
    test_ds.save('datasets/test_ds')
    
    print('Loading datasets')
    test_ds = tf.data.Dataset.load('datasets/test_ds')
    train_ds = tf.data.Dataset.load('datasets/train_ds')

    train_ds = train_ds.shuffle(buffer_size=train_ds.cardinality())
    test_ds = test_ds.shuffle(buffer_size=test_ds.cardinality())

    train_ds = train_ds.take(40000)
    test_ds = test_ds.take(10000)

    train_ds = train_ds.batch(64)
    test_ds = test_ds.batch(64)

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    result = train_model(train_ds, test_ds)

    result.save('my_model.keras')
    
    for filename, label in filenames:
        prediction(filename)
        




main()