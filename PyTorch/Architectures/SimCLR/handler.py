# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

from settings import PATHS
import matplotlib.pyplot as plt

from utils.common.dataloader import FaceLandmarksDataset
from utils.external.dataset import show_landmarks

face_dataset = FaceLandmarksDataset(
    csv_file=PATHS['CSV_PATH'],
    root_dir=PATHS['ROOT_DIR']
)

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


