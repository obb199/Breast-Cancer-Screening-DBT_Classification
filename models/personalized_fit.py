def fit_model(train_dirs, labels_file, model, epochs=1, val_dirs=None, data_augmentation_process=False):
    labels = get_labels(labels_file)
    train_results = []
    validation_results = []

    for K in range(epochs):
        print(f"{epochs}/{K + 1}")
        train_groups = separate_data_into_groups(train_dirs, elements_per_group=8, shuffle_dirs=True,
                                                 seed=random.randint(0, 999999999))
        for group in train_groups:
            X, y = [], []
            for element in group:
                full_image = np.load(element)
                slices = separate_slices(full_image)
                X += slices
                y += multiply_labels(labels[element[-14:-4:1]], len(slices) - 1)

            if data_augmentation_process:
                X, y = data_augmentation(X, y)

            random_value = random.randint(0, 99999999)
            X = separate_data_into_groups(X, 64, shuffle_dirs=True, seed=random_value)
            y = separate_data_into_groups(y, 64, shuffle_dirs=True, seed=random_value)
            for i, j in zip(X, y):
                model.train_on_batch(i, j)

        # if val_dirs:
        # resultados nos dados de validação aqui