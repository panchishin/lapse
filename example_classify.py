import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pylab as plt

tf.logging.set_verbosity(tf.logging.WARN)


def get_images(file_num):
    file_name ="data/file"+str(file_num).zfill(5)+".npy"
    return np.array( np.load(file_name) , dtype=np.float32 )/255.


def create_classifier_model():
    classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2"
    module = hub.Module(classifier_url)

    def call_module(images=images):
        return module(dict(images=images), signature="image_classification", as_dict=True)["MobilenetV2/Predictions"]

    classifier_layer = tf.keras.layers.Lambda(call_module, input_shape = [224,224,3])
    return tf.keras.Sequential([classifier_layer])


def get_imagenet_labels():
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    return np.array(open(labels_path).read().splitlines())


def top_k(results,k=5):
    return np.argsort(results, axis=-1)[:,-k:][::]



def show_one(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

imagenet_labels = get_imagenet_labels()

def show_30(images):
    global imagenet_labels
    top_index = top_k(results,k=1).reshape([-1])
    labels = imagenet_labels[top_index]
    for n in range(30):
      plt.subplot(5,6,n+1)
      plt.imshow(images[n])
      plt.title(labels[n] + " " + str(int(100*results[n,top_index[n]])))
      plt.axis('off')
    plt.show()

def show(file_num,first):
    images = get_images(file_num)
    start = 0 if first else 30
    end = start + 30
    show_30(images[start:end,:])



if __name__ == "__main__":

    classifier_model = create_classifier_model()
    classifier_model.summary()

    images = get_images(num)
    results = classifier_model.predict(images)

    show(1,False)
    show(100,False)