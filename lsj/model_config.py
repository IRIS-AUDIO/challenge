
        model = getattr(model, config.model)(
            include_top=False,
            weights=None,
            input_tensor=x,
            backend=tf.keras.backend,
            layers=tf.keras.layers,
            models=tf.keras.models,
            utils=tf.keras.utils,
        )