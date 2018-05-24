import subprocess, sys, logging, os
from itertools import islice
import SimpleITK as sitk
import numpy as np


def take(iterable, n):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def sort_file_names(file_names):
    return sorted(file_names, key=lambda dictionary: dictionary['image_file_names'][0])


def create_displacement_field_names(image_file_names, dataset_name):
    name_0, name_1 = image_file_names
    name_0 = os.path.basename(name_0)
    name_1 = os.path.basename(name_1)
    name_we_0, image_extension_we_0 = os.path.splitext(name_0)
    name_we_1, image_extension_we_1 = os.path.splitext(name_1)
    name_pair_1 = name_we_1 + "_to_" + name_we_0 + ".mha"
    name_pair_0 = name_we_0 + "_to_" + name_we_1 + ".mha"

    return (os.path.join(dataset_name, name_pair_1), os.path.join(dataset_name, name_pair_0))


# Some data sets come without world coordinate information for the label images. This is a problem
# when measuring overlap. Here, we copy information from the image to the label. We write in mhd
# format so we can inspect the header with a simple text editor
copy_information_from_images_to_labels_ext = '.mhd'
def copy_information_from_images_to_labels(image_file_names, label_file_names,
                                           displacement_field_file_names,
                                           output_directory, mhd_pixel_type):
        new_label_file_names = []
        for image_file_name, label_file_name, displacement_field_file_name \
                in zip(image_file_names, label_file_names, displacement_field_file_names):
            label_file_name_we, label_file_name_ext = os.path.splitext(label_file_name)
            dataset_output_directory = os.path.join(output_directory, 'tmp', 'labels_with_world_info',
                                                    os.path.dirname(displacement_field_file_name))
            output_file_name = os.path.join(dataset_output_directory,
                                            os.path.basename(label_file_name_we)
                                            + '_label_with_world_info'
                                            + copy_information_from_images_to_labels_ext)

            if not os.path.isdir(dataset_output_directory):
                os.makedirs(dataset_output_directory)

            # File info is read from corresponding image file
            image = sitk.ReadImage(image_file_name)

            # Write raw file with this info
            label = sitk.ReadImage(label_file_name)
            label.CopyInformation(image)

            if not os.path.isfile(output_file_name):
                sitk.WriteImage(sitk.Cast(label, sitk.sitkUInt8), output_file_name)
                print('Created label with world information %s.' % output_file_name)

            new_label_file_names.append(output_file_name)

        return tuple(new_label_file_names)


mask_ext = '.nii.gz'


def create_mask_by_thresholding(label_file_names, displacement_field_file_names,
                                output_directory, threshold, dilate, erode):
    mask_file_names = []
    for label_file_name, displacement_field_file_name \
            in zip(label_file_names, displacement_field_file_names):
        label_file_name_we, label_file_name_ext = os.path.splitext(label_file_name)
        dataset_output_directory = os.path.join(output_directory, 'tmp', 'masks',
                                                os.path.dirname(displacement_field_file_name))
        output_file_name = os.path.join(dataset_output_directory, os.path.basename(
            label_file_name_we) + '_mask' + mask_ext)

        if not os.path.isdir(dataset_output_directory):
            os.makedirs(dataset_output_directory)

        if not os.path.isfile(output_file_name):
            label = sitk.ReadImage(label_file_name)
            mask = label > threshold
            padding = (erode,)*mask.GetDimension()
            padded_mask = sitk.ConstantPad(mask, padding, padding)
            dilated_mask = sitk.BinaryDilate(padded_mask, dilate, sitk.sitkAnnulus, 0, 1, False) # pixels
            filled_mask = sitk.BinaryErode(dilated_mask, erode, sitk.sitkAnnulus, 0, 1, False)
            cropped_filled_mask = sitk.Crop(filled_mask, padding, padding)
            sitk.WriteImage(cropped_filled_mask, output_file_name)
            print('Created mask %s.' % output_file_name)

        mask_file_names.append(output_file_name)

    return tuple(mask_file_names)


def create_mask_by_size(image_file_name, mask_file_name):
    mask_directory = os.path.dirname(mask_file_name)

    if not mask_file_name.endswith(mask_ext):
        mask_file_name = os.path.splitext(mask_file_name)[0] + mask_ext

    if mask_directory is not None:
        os.makedirs(mask_directory, exist_ok=True)

    if not os.path.exists(mask_file_name):
        image = sitk.ReadImage(image_file_name)

        siz = image.GetSize()
        siz = siz[len(siz)-1:] + siz[:len(siz)-1]  # left-shift size
        mask = sitk.GetImageFromArray(np.ones(siz))
        mask.CopyInformation(image)

        os.makedirs(os.path.dirname(mask_file_name), exist_ok=True)
        sitk.WriteImage(sitk.Cast(mask, sitk.sitkUInt8), mask_file_name)

    return mask_file_name


def merge_dicts(*dicts):
    return { key: value for dict in dicts for key, value in dict.items() }


def read_pts(file_name):
    return np.loadtxt(file_name)


def read_csv(path_file):
    """ loading points from a CSV file as ndarray of floats

    :param str path_file:
    :return ndarray:

    >>> content = " ,X,Y\\n1,226.4,173.5\\n2,278,182\\n3,256.7,171.2"
    >>> _= open('sample_points.csv', 'w').write(content)
    >>> load_csv('sample_points.csv')
    array([[ 226.4,  173.5],
           [ 278. ,  182. ],
           [ 256.7,  171.2]])
    >>> os.remove('sample_points.csv')
    """
    with open(path_file, 'r') as fp:
        lines = fp.readlines()
    points = [list(map(float, l.rstrip().split(',')[1:])) for l in lines[1:]]
    return np.array(points)


def read_vtk(file_name):
    return np.loadtxt(file_name, skiprows=5)


def write_vtk(point_set, point_set_file_name):
    try:
        with open(point_set_file_name, 'w+') as f:
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Point set warp generated by SuperBench\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write("POINTS %i float\n" % point_set.shape[0])

            for point in point_set:
                for p in point:
                    f.write("%f " % p)

                f.write("\n")
    except Exception as e:
        raise Exception('Error writing vtk file: %s' % str(e))



def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def warp_point_set(superelastix, point_set, displacement_field_file_name):
    blueprint_file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'warp_point_set_%dd.json' % point_set.shape[1])

    input_point_set_file_name = os.path.splitext(displacement_field_file_name)[0] + '-input-point-set.vtk'
    output_point_set_file_name = os.path.splitext(displacement_field_file_name)[0] + '-output-point-set.vtk'

    write_vtk(point_set, input_point_set_file_name)

    try:
        stdout = subprocess.check_output([superelastix,
                                          '--conf', blueprint_file_name,
                                          '--in', 'InputPointSet=%s' % input_point_set_file_name,
                                          'DisplacementField=%s' % displacement_field_file_name,
                                          '--out', 'OutputPointSet=%s' % output_point_set_file_name,
                                          '--loglevel', 'trace',
                                          '--logfile', os.path.splitext(output_point_set_file_name)[0] + '.log'])
    except:
        raise Exception('\nFailed to warp %s. See %s' %
                        (input_point_set_file_name, os.path.splitext(output_point_set_file_name)[0] + '.log'))

    return read_vtk(output_point_set_file_name)


def warp_label_image(superelastix, label_file_name, displacement_field_file_name):
    output_label_file_name = os.path.splitext(displacement_field_file_name)[0] + '-output-label.mha'

    try:
        stdout = subprocess.check_output([superelastix,
                                          '--conf', os.path.join(get_script_path(), 'warp_label_image.json'),
                                          '--in', 'LabelImage=%s' % label_file_name,
                                          'DisplacementField=%s' % displacement_field_file_name,
                                          '--out', 'WarpedLabelImage=%s' % output_label_file_name,
                                          '--loglevel', 'trace',
                                          '--logfile', os.path.splitext(output_label_file_name)[0] + '.log'])
    except:
        logging.error('Failed to warp %s.' % label_file_name)


    return output_label_file_name
