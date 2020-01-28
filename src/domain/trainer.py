import logging


class GradientFreeTrainer:
    def fit(self, model):
        logging.info('Begin model training')
        model.train()
        logging.info('Model training done')

    def test(self, model):
        logging.info('Begin model testing')
        model.test()
        logging.info('Model testing done')

    @staticmethod
    def from_config(config):
        return GradientFreeTrainer()