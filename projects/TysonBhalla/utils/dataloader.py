import numpy as np
import gzip

class DataLoader:

    def __init__(self, train_image, train_label, test_image, test_label, 
                 sizes = {"train": 60_000, "test": 10_000}, offsets = {"img": 16, "lbl": 8},
                 img_size = (28,28)):

        self.train_image = train_image
        self.train_label = train_label
        self.test_image = test_image
        self.test_label = test_label

        self.sizes = sizes.copy()
        self.offsets = offsets.copy()
        self.img_size = img_size

    
    def get_data(self, data="train", batch_size = 1):
         with gzip.open(self.train_image if data=="train" else self.test_image, 'rb') as image_data, gzip.open(self.train_label if data=="train" else self.test_label, 'rb') as label_data:

            image_data.seek(self.offsets['img'])
            label_data.seek(self.offsets['lbl'])

            count = 0
            while count < self.sizes[data]:
                images = []
                labels = []
                count += batch_size
                
                for i in range(batch_size):

                    image = []
                    for j in range(self.img_size[0]):
                        row = []
                        for k in range(self.img_size[1]):
                            row.append(int.from_bytes(image_data.read(1)))
                        image.append(row)
                    images.append(image)

                    labels.append(int.from_bytes(label_data.read(1)))

                yield (np.array(labels), np.array(images))
