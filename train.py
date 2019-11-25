from model import train_model, train_D, evaluator

if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=120,
                              callbacks=[evaluator]
                              )
