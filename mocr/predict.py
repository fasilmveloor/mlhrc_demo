import base64
from cv2 import cv2
import numpy as np
from keras import models

CLASSES = ['്', '്വ', 'ർ', 'ൾ', '്ര', 'ൽ', 'ൗ', 'ൻ', '്യ', 'ൺ', 'ി', 'ഹ്ല', 'ൃ', 'ീ', 'ാ', 'ഹ്മ', 'ു', 'െ', 'ൂ', 'േ', 'സ്ഥ', 'സ്ല', 'ഹ', 'സ', 'സ്സ', 'സ്ധ', 'ഷ്ട', 'ഷ', 'ഹ്ന', 'സ്റ്റ', 'ള', 'ശ്ച', 'വ', 'ള്ള', 'ശ്ശ', 'ല്ല', 'ഴ', 'ശ', 'വ്വ', 'ശ്ല', 'റ്റ', 'യ', 'ര', 'മ്ല', 'മ്മ', 'റ', 'മ', 'യ്യ', 'ല', 'മ്പ', 'ബ്ദ', 'ഫ', 'ബ്ധ', 'ഫ്ള', 'ഭ', 'ബ്ബ', 'ബ്ള', 'ബ', 'പ്പ', 'പ്ല', 'ധ', 'ന്മ', 'ന്ഥ', 'ന്ധ', 'പ', 'ന', 'ന്ന', 'ന്റ', 'ന്ത', 'ത്ധ', 'ദ', 'ദ്ധ', 'ഥ', 'ത്ഭ', 'ത്സ', 'ത്ഥ', 'ത്മ', 'ദ്ദ', 'ത്ത', 'ണ്ഡ', 'ണ', 'ണ്മ', 'ണ്ണ', 'ഡ്ഠ', 'ഠ', 'ത', 'ഡ', 'ഡ്ഡ', 'ണ്ട', 'ഛ', 'ഞ്ച', 'ഞ', 'ജ', 'ച്ഛ', 'ജ്ജ', 'ട', 'ട്ട', 'ജ്ഞ', 'ഞ്ഞ', 'ങ്ക', 'ങ', 'ഗ്ഗ', 'ച്ച', 'ങ്ങ', 'ച', 'ഗ്ന', 'ഗ്മ', 'ഘ', 'ഗ്ല', 'ക്ല', 'ഗ', 'ക്ഷ', 'ക', 'ഏ', 'ഒ', 'ക്ക', 'ക്ത', 'ഖ', 'എ', 'ഉ', 'ഇ', 'ആ', 'ഋ', 'അ']

IMAGE_SIZE = 32
model = models.load_model('mocr\mlocr.h5')
classes = sorted(CLASSES)

def predict(img):
  image = data_uri_to_cv2_img(img)
  grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

  ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)

  contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  preprocessed_letters = []
  for c in contours :
    x, y, w, h = cv2.boundingRect(c)
    
    #creating a rectangle around the digit in the original image 
    #for displaying the digits fetched via contours
    cv2.rectangle(image, (x,y), (x+w, y+h), color = (16,222, 90), thickness = 3)
    
    #cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y+h , x:x+w]

    #resizing the digit to 18,18
    resized_digiit = cv2.resize(digit, (22,22))

    #padding the digits with 5 pixels of black color in each side
    padded_digit = np.pad(resized_digiit, ((5,5), (5,5)), "constant", constant_values = 0)

    #Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_letters.append(padded_digit)


  #preprocessed_letters.reverse()
  letters = []
  inp = np.array(preprocessed_letters)

  for letter in preprocessed_letters:

    l = predict_word(letter.reshape(32,32))
    letters.append(l)

  text = ""
  letters.reverse()
  text = text.join(letters)
  return text

#def predict(img):
    #img = process_image(img)
    #data = apply_nn(img)
    #return data

def predict_word(img):
    image_data = img
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    a = model.predict(dataset)[0]
    print(a)
    new = dict(zip(classes, a))
    res = sorted(new.items(), key= lambda x:x[1], reverse=True)
    print(res[0])
    return res[0][0]

def process_image(img):
    image = data_uri_to_cv2_img(img)

    # greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary
    (__, img_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # dilation

    # find contours
    ctrs, __ = cv2.findContours(
        img_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    char_ctr = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[2] * cv2.boundingRect(ctr)[3]),
                      reverse=True)[0]
    # Get bounding box
    x, y, w, h = cv2.boundingRect(char_ctr)

    # Getting ROI
    roi = img_bw[y:y + h, x:x + w]
    return skeletize(crop(roi, IMAGE_SIZE))


def crop(image, desired_size):
    """Crop and pad to req size"""
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def skeletize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeroes = size - cv2.countNonZero(img)
        if zeroes == size:
            done = True

    return skel


def apply_nn(data):
    image_data = data
    model = models.load_model('mocr\mlocr.h5')
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float64)
    a = model.predict(dataset)[0]

    classes = sorted(CLASSES)
    new = dict(zip(classes, a))
    res = sorted(new.items(), key = lambda x : x[1], reverse = True)
    return res[0][0]


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    alpha_channel = img[:, :, 3]
    rgb_channels = img[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate(
        (alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)
