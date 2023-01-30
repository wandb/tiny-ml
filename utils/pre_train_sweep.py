import os
from wandb.keras import WandbMetricsLogger
from tensorflow.keras.optimizers import Adam, Adamax, Nadam
import wandb
import tensorflow as tf
import numpy as np

def pre_train(config = None):

        run = wandb.init(config=config)

        EPOCHS = run.config.epochs or 100
        LEARNING_RATE = run.config.learning_rate or 0.005
        KERNEL_SIZE = run.config.kernel_size
        DROPOUT_P = run.config.dropout
        ACTIVATION = run.config.activation
        BATCH_SIZE = run.config.batch_size
        BETA_1 = run.config.beta_1
        BETA_2 = run.config.beta_2
        EPSILON = run.config.epsilon

        artifact = run.use_artifact('tiny-ml/wake_word_detection/npz-esc-50-files:v0', type='pre_processed_sound_data')
        artifact_dir = artifact.download()
        with np.load('./artifacts/npz-esc-50-files:v0/train.npz',allow_pickle=True) as data:
            train_x = data['x_data'].astype(np.float32)[:2000]
            train_y = data['y_data'].astype(np.uint8)[:2000]
        with np.load('./artifacts/npz-esc-50-files:v0/val.npz',allow_pickle=True) as data:
            val_x = data['x_data'].astype(np.float32)
            val_y = data['y_data'].astype(np.uint8)

        train_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_x, tf.float32), train_y))
        val_dataset = tf.data.Dataset.from_tensor_slices((tf.cast(val_x, tf.float32),val_y))

        train_ds = train_dataset.cache().shuffle(1000, seed=42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_ds = val_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
        x,y = next(iter(train_dataset.take(1)))
        #print(f' target = {y}, \n spectrogram = \n {x}')
        input_shape = tf.expand_dims(x, axis=-1).shape
        #print(input_shape)
        norm_layer.adapt(train_dataset.map(lambda x, y: tf.reshape(x, input_shape)))

        # Initialize a new W&B run 
        checkpoint_path = "training_1/"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Create a callback that saves the model's weights
        baseline_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.experimental.preprocessing.Resizing(32, 32, interpolation="nearest"), 
            norm_layer,
            tf.keras.layers.Conv2D(8, kernel_size=(KERNEL_SIZE,KERNEL_SIZE), strides=(2, 2), activation=ACTIVATION),
            tf.keras.layers.MaxPool2D(pool_size=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(DROPOUT_P),
            tf.keras.layers.Dense(50, activation='softmax')
        ])

        if run.config.optimizer == "adam":
            opt = Adam(
                learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
                )
        elif run.config.optimizer == "adamax":
            opt = Adamax(
                learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
            )
        elif run.config.optimizer == "nadam":
            opt = Nadam(
                learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON
            )

        METRICS = ["accuracy",]
        baseline_model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=METRICS
        )
        def scheduler(epoch, lr):
            ''' a function to increase lr at start of trining
            '''
            if epoch < 10:
                return lr
            else:
                # add somthing like np.linespace([0,-0.1])
                return lr * tf.math.exp(-0.1)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(verbose=0, patience=25), 
            tf.keras.callbacks.LearningRateScheduler(scheduler)
            ,cp_callback,WandbMetricsLogger()]
        history = baseline_model.fit(
             train_ds, 
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=val_ds
        )
        run.finish()