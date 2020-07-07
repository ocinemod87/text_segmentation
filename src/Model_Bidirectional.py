from tensorflow.keras.layers import (
    Layer,
    Dense,
    Input,
    LSTM,
    TimeDistributed,
    Bidirectional,
    GlobalMaxPool1D,
    BatchNormalization,
    MaxPooling1D,
    Flatten,
    Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

class Models:

    def Model_Bidirectional(nb_words, WV_DIM, wv_matrix, MAX_SEQUENCE_LENGTH):

        wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
        input_text = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = wv_layer(input_text)

        x = Bidirectional(LSTM(units=100,return_sequences=True, dropout=0.2),input_shape=(MAX_SEQUENCE_LENGTH,))(x)
        x = Bidirectional(LSTM(units=100,return_sequences=True, dropout=0.2),input_shape=(MAX_SEQUENCE_LENGTH,))(x)
        x = MaxPooling1D()(x)

        # Sentence-level LSTM
        x = Bidirectional(LSTM(units=100,return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(LSTM(units=100, dropout=0.2))(x)
        x = Dense(1, activation="sigmoid")(x)

        model = Model(input_text, x)


        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                      metrics=['accuracy'])

        return model

    def Model_Bidirectional_3(nb_words, WV_DIM, wv_matrix, MAX_SEQUENCE_LENGTH):

        wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
        input_text = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = wv_layer(input_text)

        x = Bidirectional(LSTM(units=200,return_sequences=True, dropout=0.2),input_shape=(MAX_SEQUENCE_LENGTH,))(x)
        x = Bidirectional(LSTM(units=200,return_sequences=True, dropout=0.2),input_shape=(MAX_SEQUENCE_LENGTH,))(x)
        x = MaxPooling1D()(x)

        # Sentence-level LSTM
        x = Bidirectional(LSTM(units=200,return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(LSTM(units=200, dropout=0.2))(x)
        x = Dense(3, activation="softmax")(x)

        model = Model(input_text, x)


        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                      metrics=['categorical_accuracy'])

        return model

    def Model_Bidirectional_300(nb_words, WV_DIM, wv_matrix, MAX_SEQUENCE_LENGTH):

        wv_layer = Embedding(nb_words,
                     WV_DIM,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)
        input_text = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = wv_layer(input_text)

        x = Bidirectional(LSTM(units=300,return_sequences=True, dropout=0.2),input_shape=(MAX_SEQUENCE_LENGTH,))(x)
        x = Bidirectional(LSTM(units=300,return_sequences=True, dropout=0.2),input_shape=(MAX_SEQUENCE_LENGTH,))(x)
        x = MaxPooling1D()(x)

        # Sentence-level LSTM
        x = Bidirectional(LSTM(units=300,return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(LSTM(units=300, dropout=0.2))(x)
        x = Dense(3, activation="softmax")(x)

        model = Model(input_text, x)


        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                      metrics=['categorical_accuracy'])

        return model
