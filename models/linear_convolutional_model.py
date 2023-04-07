import tensorflow as tf

class FCN3D(tf.keras.Model):
    def __init__(self, n_labels=4, **kwargs):
        super.__init__(**kwargs)

        self.FCN3D_layers = [tf.keras.layers.Conv3D(filters=16, kernel_size=(8,12,5), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=16, kernel_size=(3,3,3), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.MaxPooling3D(pool_size=(2,2,2)),
                             tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=32, kernel_size=(3,3,3), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.MaxPooling3D(pool_size=(2,2,2)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,2), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=64, kernel_size=(3,3,2), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.MaxPooling3D(pool_size=(2,2,2)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=128, kernel_size=(3,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.MaxPooling3D(pool_size=(2,2,1)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=256, kernel_size=(3,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.MaxPooling3D(pool_size=(2,2,1)),
                             tf.keras.layers.Conv3D(filters=512, kernel_size=(3,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=512, kernel_size=(3,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.MaxPooling3D(pool_size=(2,2,1)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=1024, kernel_size=(2,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv3D(filters=1024, kernel_size=(2,3,1), use_bias=True, padding='same', activation=tf.keras.layers.ReLU(), kernel_initializer=tf.keras.initializers.HeNormal()),
                             tf.keras.layers.MaxPooling3D(pool_size=(2,4,1)),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(n_labels, activation=('softmax'))]

    def call(self, input):
        output = self.FCN3D_layers[0](input)
        for layer in self.FCN3D_layers[1:]:
            output = layer(output)

        return output