import logging
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset, save_model


logger = logging.getLogger(__name__)

def run_training() -> None:
    logger.info("Training!!!!!!!!!!!!!!!!!!")
    # Read Training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    # Divide train and test set
    y=data['Transported']
    col=data.columns
    col=col.delete(6)
    x=data[col]
    X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=5,stratify = y,test_size = 0.40)
    
    clf=lgb.LGBMClassifier(random_state=5)
    model = clf.fit(X_train,y_train)
    # pred=model.predict(X_test)

    # Persist trained model
    save_model(model_to_persist=model)


if __name__ == '__main__':
    run_training()
