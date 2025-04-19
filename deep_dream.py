import tensorflow as tf
import numpy as np
from PIL import Image

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

def deprocess(img):
    img = img[0]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def calc_loss(img, model):
    layer_activations = model(img)
    loss = tf.reduce_mean(layer_activations)
    return loss

@tf.function
def deep_dream_step(img, model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    gradients = tape.gradient(loss, img)
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    img += gradients * step_size
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

def run_deep_dream(img, model, steps=100, step_size=0.01):
    for step in range(steps):
        img = deep_dream_step(img, model, step_size)
    return img

def save_and_show(img):
    img.save("dreamed_image.jpg")
    img.show()

if __name__ == "__main__":
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    layer_names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in layer_names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    original_img = load_image("sample.jpg")
    dreamed_img = run_deep_dream(original_img, dream_model)
    result = deprocess(dreamed_img)
    save_and_show(result)