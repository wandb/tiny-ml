def train_model(
    train_dataset,
    validation_dataset,
    input_length,
    callbacks,
    train_sample_count,
    classes,
    wandb_callback,
    run=None,
):
    from tensorflow.keras.optimizers import Adam, Adamax, Nadam

    if wandb_callback:
        print("Using wandb keras integration as callback.")
    override_mode = None
    disable_per_channel_quantization = False

    print(run.config)
    if not args.sweep:
        run.config.optimizer = "adam"
        EPOCHS = args.epochs or 100
        LEARNING_RATE = args.learning_rate or 0.005
        KERNEL_SIZE = args.kernel_size
        DROPOUT_P = args.dropout_p
        ACTIVATION = args.activation
        # this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
        BATCH_SIZE = args.batch_size
        BETA_1 = args.beta_1
        BETA_2 = args.beta_2
        EPSILON = args.epsilon
    else:
        EPOCHS = run.config.epochs or 100
        LEARNING_RATE = args.learning_rate or 0.005
        KERNEL_SIZE = run.config.kernel_size
        DROPOUT_P = run.config.dropout
        ACTIVATION = run.config.activation
        BATCH_SIZE = run.config.batch_size
        BETA_1 = run.config.beta_1
        BETA_2 = run.config.beta_2
        EPSILON = run.config.epsilon

    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
    validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

    # model architecture
    model = Sequential()
    model.add(Reshape((int(input_length / 40), 40), input_shape=(input_length,)))
    model.add(Conv1D(8, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding="same"))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))
    model.add(Dropout(DROPOUT_P))
    model.add(
        Conv1D(16, kernel_size=KERNEL_SIZE, activation=ACTIVATION, padding="same")
    )
    model.add(MaxPooling1D(pool_size=2, strides=2, padding="same"))
    model.add(Dropout(DROPOUT_P))
    model.add(Flatten())
    model.add(Dense(classes, activation="softmax", name="y_pred"))

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

    callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS))

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy",k_metrics.recall_,
    k_metrics.precision_,k_metrics.f1_])
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        verbose=2,
        callbacks=[callbacks, wandb_callback],
    )
    disable_per_channel_quantization = False
    return model, override_mode, disable_per_channel_quantization