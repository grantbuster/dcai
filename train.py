import os
import pandas as pd
import tensorflow as tf
import numpy as np
import sys


tf.random.set_seed(123)


def train(user_data, test_data, batch_size=8, epochs=100):
    print('Running on "{}"'.format(user_data))
    train = tf.keras.preprocessing.image_dataset_from_directory(
        user_data + '/train',
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    valid = tf.keras.preprocessing.image_dataset_from_directory(
        user_data + '/val',
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    total_length = ((train.cardinality() + valid.cardinality()) * batch_size).numpy()
    if total_length > 10_000:
        print(f"Dataset size larger than 10,000. Got {total_length} examples")
        sys.exit()

    test = tf.keras.preprocessing.image_dataset_from_directory(
        test_data,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    loss_0, acc_0 = model.evaluate(valid)
    print(f"loss {loss_0}, acc {acc_0}")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model_{}".format(os.path.basename(user_data)),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train,
        validation_data=valid,
        epochs=epochs,
        callbacks=[checkpoint],
    )

    model.load_weights("best_model_{}".format(os.path.basename(user_data)))

    loss, val_acc = model.evaluate(valid)
    print(f"final loss {loss}, final val_acc {val_acc}")

    test_loss, test_acc = model.evaluate(test)
    print(f"test loss {test_loss}, test acc {test_acc}")

    valid = tf.keras.preprocessing.image_dataset_from_directory(
        user_data + '/val',
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    df = pd.DataFrame()
    x = []
    for i, image in enumerate(valid.unbatch()):
        x.append(image[0])
        fp = valid.file_paths[i]
        truth = image[1].numpy().argmax() + 1
        df.at[fp, 'dataset'] = 'val'
        df.at[fp, 'truth'] = truth

    for i, image in enumerate(test.unbatch()):
        x.append(image[0])
        fp = test.file_paths[i]
        truth = image[1].numpy().argmax() + 1
        df.at[fp, 'dataset'] = 'test'
        df.at[fp, 'truth'] = truth

    print('Running predictions...')
    y_prob = model.predict(np.array(x))
    predictions = y_prob.argmax(axis=1) + 1
    df['prediction'] = predictions
    df.index.name = 'fp'
    df = df.reset_index()
    fp = './predictions_{}.csv'.format(os.path.basename(user_data), index=False)
    df.to_csv(fp)
    print('Finished writing predictions to: {}'.format(fp))
    return df, val_acc, test_acc


if __name__ == '__main__':
    user_data = str(sys.argv[1] + '/' + sys.argv[1])
    test_data = str(sys.argv[2] + '/' + 'label_book')
    train(user_data, test_data)
