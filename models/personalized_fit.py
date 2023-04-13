def fit_model(train_dirs, labels_dir, model=None, epochs=1, size_groups=8, size_part_mini_batch=32, val_dirs=None, data_augmentation_process=False):
    best_accuracy = 0
    
    labels = get_labels(labels_dir)
    training = []
    validations = []
    for _ in range(epochs):
        groups = separate_data_into_groups(train_dirs, size_groups)
        for group in groups:
            X, y = [], []
            for element in group:
                full_image = np.load(element)
                X += separate_slices(full_image)
                for _ in range(full_image.shape[-2]+1):
                    y.append(labels[element[-14:-4]])
                
                if data_augmentation_process:
                    X, y = data_augmentation(X, y)
            
            partial_res = []
            for part in range(0, len(X), size_part_mini_batch):
                partial_res.append(model.train_on_batch(np.array(X[part:part+size_part_mini_batch]), np.array(y[part:part+size_part_mini_batch])))
        training.append(np.array(partial_res).mean(axis=0))
    
        if val_dirs:
            validations.append(test_model(val_dirs, labels_dir=labels_dir, model=test, size_groups=2, size_part_mini_batch=32).mean(axis=0))
            if validations[-1][2] > best_accuracy:
                best_accuracy = validations[-1][2]
                weights = model.save_weights('/kaggle/working/')
    
    
    if val_dirs:
        return np.array(training), np.array(validations)
    return np.array(training)            

def test_model(test_dirs, labels_dir, model, size_groups=8, size_part_mini_batch=32):
    labels = get_labels(labels_dir)
    groups = separate_data_into_groups(test_dirs, size_groups)
    for group in groups:
        X_test, y_test = [], []
        res = []
        for element in group:
            full_image = np.load(element)
            X_test += separate_slices(full_image)
            for _ in range(full_image.shape[-2]+1):
                y_test.append(labels[element[-14:-4]])
            for part in range(0, len(X_test), size_part_mini_batch):
                res.append(model.evaluate(np.array(X_test[part:part+size_part_mini_batch]), np.array(y_test[part:part+size_part_mini_batch]), verbose=0))
    
    return np.array(res)

def soft_voting(image, model):
    predicts = []
    full_image = np.load(element)
    X_test = separate_slices(full_image)
    for X in test:
        classes.append([model.predict([X])[0]])
    
    return np.array(predicts).mean(axis=1)

def hard_voting(image, model, dir_labels=None):
    predicts = []
    
    full_image = np.load(element)
    X_test = separate_slices(full_image)
    for X in test:
        predicts[np.argmax(model.predict([X])[0])] += 1
    
    return np.array(predicts)
