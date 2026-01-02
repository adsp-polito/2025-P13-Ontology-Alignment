import pandas as pd
from training.bi_encoder_training import train_bi_encoder
from training.cross_encoder_training import train_cross_encoder
from training.utils import stratified_split

def train_model(
        df_training: pd.DataFrame,
        model_type: str,
        model_name: str,
        output_dir: str,
        num_epochs: int = 10
) -> None:
    """
    Train a model (bi-encoder or cross-encoder) on the given training dataframe.

    Parameters:
        df_training (pd.DataFrame): training dataframe
        model_type (str): type of the model ("bi-encoder" or "cross-encoder")
        model_name (str): name of the pre-trained model
        output_dir (str): output directory to save the trained model
    """

    df_train, df_val, df_test = stratified_split(df_training)

    if model_type == "bi-encoder":
        
        train_bi_encoder(
            df_train,
            df_val,
            df_test,
            num_epochs=int(num_epochs),
            model_name=model_name,
            output_dir=output_dir
        )

    else:
        
        train_cross_encoder(
            df_train,
            df_val,
            df_test,
            num_epochs=int(num_epochs),
            model_name=model_name,
            output_dir=output_dir
        )