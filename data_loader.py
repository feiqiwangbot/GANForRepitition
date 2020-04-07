import scipy
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.counter = 0
        
    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./dataset/' + self.dataset_name)

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, count = 0, batch_size=1, is_testing=False):
        #data_type = "train" if not is_testing else "val"
        #path_A = glob('./dataset/*' % (self.dataset_name, data_type))
        path_A = glob('./dataset/dataset/*')
        #path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))
        #print(len(path_A))
        self.n_batches = int(len(path_A)/ batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        #path_B = np.random.choice(path_B, total_samples, replace=False)

        #for i in range(self.n_batches-1):
        batch_A = path_A[count*batch_size:(count+1)*batch_size]
        #batch_B = path_B[i*batch_size:(i+1)*batch_size]
        imgs_A = []
        for img_A in batch_A:
            img_A = self.imread(img_A)
            #img_B = self.imread(img_B)
            img_A = scipy.misc.imresize(img_A, self.img_res)
            #img_B = scipy.misc.imresize(img_B, self.img_res)
            #if not is_testing and np.random.random() > 0.5:
            #       img_A = np.fliplr(img_A)
            #      img_B = np.fliplr(img_B)
            imgs_A.append(img_A)
            #imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)/127.5 - 1.
        #imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
