class test_no_linear_model(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hiddens1 = [
            tf.keras.layers.Conv3D(input_shape=[175, 260, 10, 1], filters=16, kernel_size=(8, 12, 10), use_bias=True,
                                   padding='same', activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(7, 7, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(7, 7, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 2)),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(7, 7, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(7, 7, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(7, 7, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(7, 7, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(7, 7, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(7, 7, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)),
            tf.keras.layers.BatchNormalization()]

        self.hiddens2 = [
            tf.keras.layers.Conv3D(input_shape=[175, 260, 10, 1], filters=16, kernel_size=(8, 12, 10), use_bias=True,
                                   padding='same', activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(5, 5, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 2)),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(5, 5, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(5, 5, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(5, 5, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(5, 5, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(5, 5, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(5, 5, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)),
            tf.keras.layers.BatchNormalization()]

        self.hiddens3 = [
            tf.keras.layers.Conv3D(input_shape=[175, 260, 10, 1], filters=16, kernel_size=(8, 12, 10), use_bias=True,
                                   padding='same', activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 2)),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 3, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)),
            tf.keras.layers.BatchNormalization()]

        self.hiddens4 = [
            tf.keras.layers.Conv3D(input_shape=[175, 260, 10, 1], filters=16, kernel_size=(8, 12, 10), use_bias=True,
                                   padding='same', activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(2, 2, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=16, kernel_size=(2, 2, 5), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 2)),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(2, 2, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=32, kernel_size=(2, 2, 3), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(2, 2, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=(2, 2, 2), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(2, 2, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=(2, 2, 1), use_bias=True, padding='same',
                                   activation=tf.keras.layers.ReLU(),
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1)),
            tf.keras.layers.BatchNormalization()]

        self.dense_layers = [tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=4096, activation='selu', kernel_initializer='lecun_normal'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units=2048, activation='selu', kernel_initializer='lecun_normal'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units=1024, activation='selu', kernel_initializer='lecun_normal'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units=512, activation='selu', kernel_initializer='lecun_normal'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units=256, activation='selu', kernel_initializer='lecun_normal'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Dense(units=4, activation='softmax')]

    def call(self, inputs):
        input1 = tf.identity(inputs)
        input2 = tf.identity(inputs)
        input3 = tf.identity(inputs)
        input4 = tf.identity(inputs)

        output1 = self.hiddens1[0](input1)
        for layer in self.hiddens1[1:]:
            output1 = layer(output1)

        output2 = self.hiddens2[0](input2)
        for layer in self.hiddens2[1:]:
            output2 = layer(output2)

        output3 = self.hiddens3[0](input3)
        for layer in self.hiddens3[1:]:
            output3 = layer(output3)

        output4 = self.hiddens4[0](input4)
        for layer in self.hiddens4[1:]:
            output4 = layer(output4)

        concat = tf.keras.layers.concatenate([output1, output2, output3, output4])

        output = self.dense_layers[0](concat)
        for layer in self.dense_layers[1:]:
            output = layer(output)

        return output