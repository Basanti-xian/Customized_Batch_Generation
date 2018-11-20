import matplotlib.pyplot as plt
import numpy as np
import os, random, cv2, time
from PIL import Image
from data_util import GeneratorEnqueuer
import tensorflow as tf

tf.app.flags.DEFINE_float('min_crop_side_ratio', 25,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')

FLAGS = tf.app.flags.FLAGS



def random_batch(data_train_image, data_train_label, batch_size):
    # Number of images (transfer-values) in the training-set.
    num_images = len(data_train_image)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = [data_train_image[j] for j in idx]
    y_batch = [data_train_label[j] for j in idx]

    return x_batch, y_batch


def load_data(img_path= 'Summer 2018 Pics/train'):
    total_imgs = []
    total_labels = []
    # img_path = 'Summer 2018 Pics'

    for subdir in os.listdir(img_path):
        subdir_path = os.path.join(img_path, subdir)
        if subdir == 'A':
            for f in os.listdir(subdir_path):
                ext = os.path.splitext(f)[1]
                if ext == '.png':
                    img = np.array(Image.open(os.path.join(subdir_path, f)))
                    total_imgs.append(np.array(img))
                    total_labels.append([1,0,0])

        elif subdir == 'B':
            for f in os.listdir(subdir_path):
                ext = os.path.splitext(f)[1]
                if ext == '.png':
                    img = np.array(Image.open(os.path.join(subdir_path, f)))
                    total_imgs.append(np.array(img))
                    total_labels.append([0,1,0])

        elif subdir == 'C':
            for f in os.listdir(subdir_path):
                ext = os.path.splitext(f)[1]
                if ext == '.png':
                    img = np.array(Image.open(os.path.join(subdir_path, f)))
                    total_imgs.append(np.array(img))
                    total_labels.append([0,0,1])


    tmp = list(zip(total_imgs, total_labels))
    random.shuffle(tmp)
    return zip(*tmp)


def crop_area(im, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)

    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)

        if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
            # area too small
            continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]

        return im

    return im


def generator(dir_path, input_size=512, batch_size=32,
              random_scale=np.array([0.5, 1])):

    image_list, label_list = load_data(dir_path)
    # print('{} training images'.format(image_list.shape[0]))
    index = np.arange(0, len(image_list))

    while True:
        np.random.shuffle(index)

        images = np.zeros((batch_size, input_size, input_size, 3))
        labels = np.zeros((batch_size, 3))
        idx = 0
        # images = []
        # labels = []

        for i in index:
            try:
                im = image_list[i]
                label = label_list[i]
                h, w, _ = im.shape

                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)

                # crop background
                im= crop_area(im)

                # pad and resize image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize=(input_size, input_size))

                # images.append(im[:, :, ::-1].astype(np.float32))
                # labels.append(label)

                if idx < batch_size:
                    images[idx] = (im[:, :, ::-1]/255).astype(np.float32)
                    labels[idx] = label
                    idx +=1
                elif idx == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, input_size, input_size, 3))
                    labels = np.zeros((batch_size, 3))
                    idx = 0

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue






def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



