def soft_voting(image, model):
    predicts = []
    full_image = np.load(element)
    X_test = separate_slices(full_image)
    for X in test:
        classes.append([model.predict([X])[0]])
    
    return np.array(predicts).mean(axis=1)

def hard_voting(image, model):
    predicts = []
    
    full_image = np.load(element)
    X_test = separate_slices(full_image)
    for X in test:
        predicts[np.argmax(model.predict([X])[0])] += 1
    
    return np.array(predicts)
