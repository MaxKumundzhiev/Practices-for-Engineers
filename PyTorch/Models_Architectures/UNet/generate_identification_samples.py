# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import glob
import sys
import numpy as np
import elasticdeform
from utility_functions import opening_files
from utility_functions.sampling_helper_functions import densely_label, pre_compute_disks


def generate_slice_samples(dataset_dir, sample_dir, sample_size=(40, 160), spacing=(2.0, 2.0, 2.0),
                           no_of_samples=5, no_of_vertebrae_in_each=2, file_ext=".nii.gz"):
    sample_size = np.array(sample_size)
    ext_len = len(file_ext)

    paths = glob.glob(dataset_dir + "/**/*" + file_ext, recursive=True)

    np.random.seed(1)

    sample_size_np = np.array(sample_size, int)
    print("Generating " + str(no_of_samples * len(paths)) + " identification samples of size 8 x "
          + str(sample_size_np[0]) + " x " + str(sample_size_np[1]) + " for " + str(len(paths)) + " scans")

    for cnt, data_path in enumerate(paths):
        data_path_without_ext = data_path[:-ext_len]
        metadata_path = data_path_without_ext + ".lml"
        volume = opening_files.read_nii(data_path, spacing=spacing)

        labels, centroids = opening_files.extract_centroid_info_from_lml(metadata_path)
        centroid_indexes = np.round(centroids / np.array(spacing)).astype(int)

        disk_indices = pre_compute_disks(spacing)
        dense_labelling = densely_label(volume.shape, disk_indices, labels, centroid_indexes, use_labels=True)
        lower_i = np.min(centroid_indexes[:, 0])
        lower_i = np.max([lower_i - 15, 0]).astype(int)
        upper_i = np.max(centroid_indexes[:, 0])
        upper_i = np.min([upper_i + 15, volume.shape[0] - 1]).astype(int)

        cuts = []
        while len(cuts) < no_of_samples:
            cut = np.random.randint(lower_i, high=upper_i)
            sample_labels_slice = dense_labelling[cut - 4: cut + 4, :, :]
            if np.unique(sample_labels_slice).shape[0] > no_of_vertebrae_in_each:
                cuts.append(cut)

        name = (data_path.rsplit('/', 1)[-1])[:-ext_len]

        count = 0
        for i in cuts:

            volume_slice = volume[i-4:i+4, :, :]
            sample_labels_slice = dense_labelling[i, :, :]

            if volume_slice.shape[0] != 8:
                break
            '''
            [volume_slice, sample_labels_slice] = elasticdeform.deform_random_grid(
                [volume_slice, sample_labels_slice], sigma=7, points=3, order=0)
            '''
            [volume_slice, sample_labels_slice] = elasticdeform.deform_random_grid(
                [volume_slice, np.expand_dims(sample_labels_slice, axis=0)], sigma=7, points=3, order=0, axis=(1, 2))

            sample_labels_slice = np.squeeze(sample_labels_slice, axis=0)
            if volume_slice.shape[1] < sample_size[0]:
                dif = sample_size[0] - volume_slice.shape[1]
                volume_slice = np.pad(volume_slice, ((0, 0), (0, dif), (0, 0)),
                                      mode="constant", constant_values=-5)
                sample_labels_slice = np.pad(sample_labels_slice, ((0, dif), (0, 0)),
                                             mode="constant")

            if volume_slice.shape[2] < sample_size[1]:
                dif = sample_size[1] - volume_slice.shape[2]
                volume_slice = np.pad(volume_slice, ((0, 0), (0, 0), (0, dif)),
                                      mode="constant", constant_values=-5)
                sample_labels_slice = np.pad(sample_labels_slice, ((0, 0), (0, dif)),
                                             mode="constant")

            j = 0
            while True:
                random_area = volume_slice.shape[1:3] - sample_size
                random_factor = np.random.rand(2)
                random_position = np.round(random_area * random_factor).astype(int)
                corner_a = random_position
                corner_b = corner_a + sample_size

                cropped_combines_slice = volume_slice[:, corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]
                cropped_sample_labels_slice = sample_labels_slice[corner_a[0]:corner_b[0], corner_a[1]:corner_b[1]]

                care_about_labels = np.count_nonzero(cropped_sample_labels_slice)
                j += 1
                if care_about_labels > 500 or j > 100:
                    break

            count += 1
            name_plus_id = name + "-" + str(count)
            path = '/'.join([sample_dir, name_plus_id])
            sample_path = path + "-sample"
            labelling_path = path + "-labelling"
            np.save(sample_path, cropped_combines_slice)
            np.save(labelling_path, cropped_sample_labels_slice)
        print(str(cnt + 1) + " / " + str(len(paths)))


generate_slice_samples(dataset_dir=sys.argv[1],
                       sample_dir=sys.argv[2],
                       sample_size=(80, 320),
                       no_of_samples=100,
                       spacing=(1.0, 1.0, 1.0),
                       no_of_vertebrae_in_each=1)
