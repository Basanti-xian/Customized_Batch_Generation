from Utils import load_data, crop_area
from keras.models import load_model
import numpy as np
import cv2
from collections import Counter



def test_crops(im, input_size, num_crops):

    random_scale = np.array([0.5, 1])
    images = []

    while True:
        try:
            h, w, _ = im.shape

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)

            # crop background
            im = crop_area(im)

            # pad and resize image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(input_size, input_size))

            images.append(im[:, :, ::-1].astype(np.float32))

            if len(images) == num_crops:
                return images

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue


def predict_crops(imgs, labels, num_crops):
    predict_labels = []
    predict_confidence = []

    for im in imgs:
        img_crops = test_crops(im, input_size=224, num_crops=num_crops)
        y_preds = model.predict(np.array(img_crops))
        preds = np.argmax(y_preds, axis=1)
        c = Counter(preds)
        p = c.most_common()[0]
        predict_confidence.append(p[1]/num_crops)
        predict_labels.append(p[0])

    print('Ground Truth: ', labels)
    print('Predictions: ', predict_labels)
    print('Prediction Confidence: ', predict_confidence)

    correct_prediction = np.multiply(np.equal(predict_labels, labels),1)
    accuracy = np.mean(correct_prediction)
    print("Accuracy: ", accuracy)


if __name__ == '__main__':

    model = load_model('checkpoints/vgg16_classification.h5')


    X_test, y_test = load_data('Summer 2018 Pics/test')
    y_test_cls = np.argmax(y_test, axis=1)

    # for i in range(5,15):
    #     print("When cropped to {} pieces: ".format(i))
    #     predict_crops(X_test, y_test_cls, i)

    predict_crops(X_test, y_test_cls, 10)

