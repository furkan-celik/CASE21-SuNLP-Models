import tensorflow as tf

def getLSTMModel(embedding_matrix, maxlen, PARAMS):
    model = tf.keras.models.Sequential([
                tf.keras.layers.Embedding(input_dim= embedding_matrix.shape[0],
                                          output_dim= embedding_matrix.shape[1],
                                          weights = [embedding_matrix],
                                          input_length= maxlen,
                                         trainable = True),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(maxlen, dropout= PARAMS["dropout"], recurrent_dropout = PARAMS["seq_dropout"])),
                tf.keras.layers.Dropout(PARAMS["dropout"]),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(PARAMS["dropout"]),
                tf.keras.layers.Dense(1, activation='sigmoid')
                ]
    )

    model.compile(
        loss='binary_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(), 
        metrics=['accuracy', "AUC"]
    )

    return model