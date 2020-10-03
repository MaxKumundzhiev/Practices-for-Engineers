import os
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig(level=logging.DEBUG)


def get_data():
    if not os.path.exists('../testSetPlaces205_resize.tar.gz'):
        logging.info('Downloading data.')
        cmd = 'wget http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz'
        os.system(cmd)

    if not os.path.exists('../testSet_resize'):
        logging.info('Unzipping data.')
        cmd = 'tar -xzf testSetPlaces205_resize.tar.gz'
        os.system(cmd)
    return True


def move_data():
    os.makedirs('../images/train/class/', exist_ok=True) # 40,000 images
    os.makedirs('../images/val/class/', exist_ok=True) # 1,000 images
    logging.info('Successfully created images/train/class/ and images/val/class/ folders.')

    images = os.listdir('../testSet_resize')
    logging.info(f'Amount of images: {len(images)}')

    for image_index, image in enumerate(tqdm(images)):
        if image_index < 1000:  # first 1000 will be val
            os.rename('testSet_resize/' + image, 'images/val/class/' + image)
        else:
            os.rename('testSet_resize/' + image, 'images/train/class/' + image)
    logging.info(f'Split data on train and validation.')
    return images


if __name__ == '__main__':
    get_data()
    logging.info(f'Downloaded and unzipped data successfully under {os.getcwd ()}.')

    # move_data()
    logging.info(f'Amount of Train {len(os.listdir("../images/train/class/"))}.')
    logging.info(f'Amount of Validation {len(os.listdir("../images/val/class/"))}.')

    # verify plotting image
    train_images = os.listdir("../images/train/class/")
    validation_images = os.listdir("../images/train/class/")

    full_path = os.path.join('../images/train/class/', train_images[1])
    print(full_path)
    img = mpimg.imread(f'{full_path}')
    imgplot = plt.imshow(img)
    plt.show()





