import tensorflow as tf

# Load the existing model
model = tf.keras.models.load_model("model.h5")

# Export in SavedModel format
model.export("saved_model/")
print("Model successfully saved in 'saved_model/' format.")
