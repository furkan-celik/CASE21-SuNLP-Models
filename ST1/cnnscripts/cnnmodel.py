import tensorflow as tf
from sklearn.metrics import f1_score

# TensorFlow is not good at calculation of F1 Score. And calculating it inside batches is a bit problematic. 
# We decided to implement this as callback rather than metric, even though this prevents it to be used as monitor in earlystopping, this shows us true values.
# We calculate both training macro f1 and validation macro f1 to detect overfitting
class MacroF1Callback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    y_pred = self.model.predict(X_train)

    print("Train F1 Score: ", f1_score(y_train, y_pred > 0.5, average = "macro"))
    
    y_pred = self.model.predict(X_val)

    print("Test F1 Score: ", f1_score(y_val, y_pred > 0.5, average = "macro"))

def get_model(embedding_matrix, maxlen, PARAMS):
  model = tf.keras.models.Sequential([
                                      tf.keras.layers.Embedding(input_dim= embedding_matrix.shape[0],
                                                                output_dim= embedding_matrix.shape[1],
                                                                weights = [embedding_matrix],
                                                                input_length= maxlen,
                                                              trainable = PARAMS["train_embed"]),
                                      tf.keras.layers.Conv1D(128, 3, activation= "relu", padding= "same"),
                                      tf.keras.layers.GlobalMaxPool1D(),
                                      tf.keras.layers.Dropout(PARAMS["dropout"]),
                                      tf.keras.layers.Dense(64, activation= "relu", kernel_regularizer= tf.keras.regularizers.l2(1e-4)),
                                      tf.keras.layers.Dense(1, activation= "sigmoid")
                                      ]
  )

  model.compile(loss='binary_crossentropy', 
              optimizer='Adam', 
              metrics=['accuracy', "AUC"])
  
  return model