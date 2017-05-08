import gc
import os
import pickle
import pprint

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.models import Model
from keras.layers import *
from keras.callbacks import EarlyStopping


# ---------- Config -----------

EXPERIMENT_ID = 'lystdo-fasttext'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

data_folder = os.path.abspath(os.path.join(os.curdir, os.pardir, 'data')) + os.path.sep
aux_data_folder = os.path.join(data_folder, 'aux') + os.path.sep
features_data_folder = os.path.join(data_folder, 'features') + os.path.sep

NUM_FOLDS = 5
NUM_RANDOM_SEARCH_ITERATIONS = 30
NUM_EPOCHS = 25

# Define random search structure
search_grid = [
    {
        'num_lstm': np.random.randint(128, 512),
        'num_dense': np.random.randint(50, 250),
        'lstm_dropout_rate': np.random.random_sample() / 2,
        'dense_dropout_rate': np.random.random_sample() / 2,
    }
    for i in range(NUM_RANDOM_SEARCH_ITERATIONS)
]


# -------- Read data ----------

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


embedding_matrix = load(aux_data_folder + 'embedding_weights_fasttext_filtered_no_stopwords.pickle')

X_q1 = load(features_data_folder + 'X_train_nn_fasttext_q1_filtered_no_stopwords.pickle')
X_q2 = load(features_data_folder + 'X_train_nn_fasttext_q2_filtered_no_stopwords.pickle')

y = load(features_data_folder + 'y_train.pickle')

EMBEDDING_DIM = embedding_matrix.shape[-1]
VOCAB_LENGTH = embedding_matrix.shape[0]
MAX_SEQUENCE_LENGTH = X_q1.shape[-1]

print('Embedding dim: ', EMBEDDING_DIM)
print('Vocab length:  ', VOCAB_LENGTH)
print('Max seq length:', MAX_SEQUENCE_LENGTH)

print('X_q1:', X_q1.shape)
print('X_q2:', X_q2.shape)
print('y_  :', y.shape)


# -------- Define models ----------

def create_model(params):
    embedding_layer = Embedding(
        VOCAB_LENGTH,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False,
    )
    lstm_layer = LSTM(
        params['num_lstm'],
        dropout=params['lstm_dropout_rate'],
        recurrent_dropout=params['lstm_dropout_rate'],
    )

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(params['dense_dropout_rate'])(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(params['num_dense'], activation='relu')(merged)
    merged = Dropout(params['dense_dropout_rate'])(merged)
    merged = BatchNormalization()(merged)

    output = Dense(1, activation='sigmoid')(merged)

    model = Model(
        inputs=[sequence_1_input, sequence_2_input],
        outputs=output
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer='nadam',
        metrics=['accuracy']
    )

    return model


def get_model_fingerprint(params):
    return EXPERIMENT_ID + '-lstm-{}-dense-{}-droplstm-{:.3f}-dropdense-{:.3f}'.format(
        params['num_lstm'],
        params['num_dense'],
        params['lstm_dropout_rate'],
        params['dense_dropout_rate'],
    )


# Do a K-Fold Split
kfold = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED
)


# -------- Perform Random Search ----------

histories = []
best_score = 1e9
best_params = None

# Begin Random Search.
for search_iter, current_params in enumerate(search_grid):

    print()
    print('-' * 30, f'Iteration {search_iter + 1} / {NUM_RANDOM_SEARCH_ITERATIONS}', '-' * 30)
    print(f'Trying parameter combination:')
    pprint.pprint(current_params)

    current_iter_val_scores = []

    # Begin K-Fold.
    for fold_num, (ix_train, ix_val) in enumerate(kfold.split(X_q1, y)):
        X_fold_train_q1 = np.vstack([X_q1[ix_train], X_q2[ix_train]])
        X_fold_train_q2 = np.vstack([X_q2[ix_train], X_q1[ix_train]])

        X_fold_val_q1 = np.vstack([X_q1[ix_val], X_q2[ix_val]])
        X_fold_val_q2 = np.vstack([X_q2[ix_val], X_q1[ix_val]])

        y_fold_train = np.concatenate([y[ix_train], y[ix_train]])
        y_fold_val = np.concatenate([y[ix_val], y[ix_val]])

        print()
        print(f'Fitting fold {fold_num + 1} of {kfold.n_splits}')
        print()

        model = create_model(current_params)
        history = model.fit(
            [X_fold_train_q1, X_fold_train_q2], y_fold_train,
            validation_data=([X_fold_val_q1, X_fold_val_q2], y_fold_val),

            batch_size=2048,
            epochs=NUM_EPOCHS,
            verbose=1,

            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.001,
                    patience=3,
                    verbose=1,
                    mode='auto',
                ),
            ],
        )

        best_val_score = min(history.history['val_loss'])
        print(f'Validation score: {best_val_score}')

        current_iter_val_scores.append(best_val_score)
        histories.append((current_params, best_val_score, history.history))
        save(histories, aux_data_folder + f'{EXPERIMENT_ID}-random-search-history.pickle')

    # End K-Fold
    # Save the trained model with the current parameter combination.
    current_iter_avg_score = np.mean(current_iter_val_scores)
    model_save_filename = '{}-random-search-{:.4f}-{}.keras'.format(
        EXPERIMENT_ID,
        current_iter_avg_score,
        get_model_fingerprint(current_params)
    )

    if current_iter_avg_score < best_score:
        best_score = current_iter_avg_score
        best_params = current_params

    print()
    print('CV score  :', current_iter_avg_score)
    print('Saving as :', model_save_filename)
    model.save(aux_data_folder + model_save_filename)

    K.clear_session()
    del model
    gc.collect()


# -------- Print final results ----------

print()
print('=' * 70)
print('Best CV score:', best_score)
print('Best params:')
pprint.pprint(best_params)
