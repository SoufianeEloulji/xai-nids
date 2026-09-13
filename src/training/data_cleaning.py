from src import logger
from abc import ABC, abstractmethod
import pandas as pd
import re
import numpy as np
import os
import joblib
from typing import Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split



class DataStrategy(ABC):
    """Abstract base class for data strategies."""
    
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Any:
        """Method to be implemented by subclasses to handle data."""
        pass


class PreProcessingStrategy(DataStrategy):
    """Concrete strategy for preprocessing data."""
    
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by handling duplicates, standardizing labels andcolumns' names, managing anomalies, and creating new features.

        Args:
            df (pd.DataFrame): The input DataFrame to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        try :
            clean_columns = [re.sub(r'[ ,;{}()\n\t=]+', '_', c.strip()).strip('_') for c in df.columns]
            df.columns = clean_columns


            #deleting duplicates
            df = df.drop_duplicates()


            #Labels standardization
            df["Label"] = df["Label"].str.replace(r"�", " - ", regex=True)


            # Infinity/Null values treatment
            df["Flow_Duration"] = np.where(df["Flow_Duration"] == 0, 1, df["Flow_Duration"])

            df["Total_Bytes"] = df["Total_Length_of_Fwd_Packets"] + df["Total_Length_of_Bwd_Packets"]
            df["Flow_Bytes/s"] = (df["Total_Bytes"] / df["Flow_Duration"]) * 1000000.0
            df = df.drop(columns=["Total_Bytes"])


            #treating anormal values
            condition = df["Flow_Duration"] >= 0

            iat_cols = [
                "Flow_IAT_Mean", "Flow_IAT_Max", "Flow_IAT_Min",
                "Fwd_IAT_Mean", "Fwd_IAT_Max", "Fwd_IAT_Min",
                "Bwd_IAT_Mean", "Bwd_IAT_Max", "Bwd_IAT_Min",
            ]
            for c in iat_cols:
                condition &= (df[c] >= 0)

            range_cols = ["Fwd_Header_Length", "Bwd_Header_Length", "min_seg_size_forward"]
            for c in range_cols:
                condition &= (df[c] >= 0) & (df[c] < 100000)

            df = df[condition].reset_index(drop=True)


            #deduplicating Fwd_Header_Length column
            df = df.drop(columns=["Fwd_Header_Length.1"])




            #Creation of new relevant variables
            df_fe = df.copy()

            df_fe["fwd_bwd_byte_ratio"] = df_fe["Total_Length_of_Fwd_Packets"] / (df_fe["Total_Length_of_Bwd_Packets"] + 1)
            df_fe["fwd_bwd_packet_ratio"] = df_fe["Total_Fwd_Packets"] / (df_fe["Total_Backward_Packets"] + 1)
            df_fe["iat_coeff_variation"] = df_fe["Flow_IAT_Std"] / (df_fe["Flow_IAT_Mean"] + 1)

            flags_sum = (df_fe["FIN_Flag_Count"] + df_fe["SYN_Flag_Count"] + df_fe["RST_Flag_Count"] +
                        df_fe["PSH_Flag_Count"] + df_fe["ACK_Flag_Count"] + df_fe["URG_Flag_Count"])
            df_fe["flag_density"] = flags_sum / (df_fe["Total_Fwd_Packets"] + df_fe["Total_Backward_Packets"] + 1)
            df_fe["has_no_win_scaling_fwd"] = (df_fe["Init_Win_bytes_forward"] == -1).astype(int)
            df_fe["has_no_win_scaling_bwd"] = (df_fe["Init_Win_bytes_backward"] == -1).astype(int)

            regex_pattern = r"(Monday|Tuesday|Wednesday|Thursday|Friday)"
            df_fe["day_of_week_str"] = df_fe["source_filename"].str.extract(regex_pattern, flags=re.IGNORECASE)[0]

            df_fe["Init_Win_bytes_forward_clean"] = np.where(df_fe["Init_Win_bytes_forward"] == -1, 0, df_fe["Init_Win_bytes_forward"])
            df_fe["Init_Win_bytes_backward_clean"] = np.where(df_fe["Init_Win_bytes_backward"] == -1, 0, df_fe["Init_Win_bytes_backward"])


            #deleting redundant columns + clumns containing 0 only
            cols_to_drop = [
                "Active_Max", "Active_Min", "Idle_Std", "Idle_Max", "Idle_Min", "Fwd_PSH_Flags", 
                "Average_Packet_Size", "Packet_Length_Variance", "RST_Flag_Count", "Avg_Fwd_Segment_Size", 
                "Bwd_IAT_Max", "Bwd_IAT_Total", "Subflow_Bwd_Packets", "Subflow_Fwd_Packets", 
                "Subflow_Bwd_Bytes", "Bwd_Packet_Length_Mean", "Bwd_Packet_Length_Max", "Max_Packet_Length", 
                "Fwd_URG_Flags", "Fwd_Packets/s", "Flow_Packets/s", "Flow_IAT_Mean", "Active_Std", "source_filename", "Init_Win_bytes_forward", "Init_Win_bytes_backward", 
                "Bwd_Avg_Bulk_Rate", "Bwd_Avg_Packets/Bulk", "Bwd_Avg_Bytes/Bulk", "Fwd_Avg_Bulk_Rate", 
                "Fwd_Avg_Packets/Bulk", "Fwd_Avg_Bytes/Bulk", "Bwd_URG_Flags", "Bwd_PSH_Flags"
            ]
                            
            df_fe = df_fe.drop(columns=cols_to_drop)
            return df_fe

        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            raise  


class DataDivideStrategy(DataStrategy):
    """Concrete strategy for dividing data into X_train, X_test, y_train, y_test."""
    
    def __init__(self, target_column: str = "Label"):
        self.target_column = target_column

    def handle_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divides the DataFrame into training and testing sets.
        Args:
            df (pd.DataFrame): The input DataFrame to divide.
        Returns:
            tuple: A tuple containing X_train, X_test, y_train, y_test.
        """
        try:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.20,     
                stratify=y, 
                random_state=42      
            )
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            return X_train, X_test, y_train, y_test

        except Exception as e:  
            logger.error(f"Error in dividing data: {e}")
            raise


class DataCleaning:
    """
    Context class for data cleaning that uses a specified strategy."""
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        """
        Handles the data using the provided strategy.
        Returns:
            The result of the strategy's handle_data method.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logger.error(f"Error in data cleaning: {e}")
            raise


class DataEncodingScalingStrategy():
    """Concrete strategy for encoding and scaling data."""
    
    def __init__(self, artifacts_path: str = "artifacts/"):
        self.artifacts_path = artifacts_path

    def handle_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Encodes and scales the training and testing data without data leakage.
        Args:
            X_train (pd.DataFrame): The training features.
            X_test (pd.DataFrame): The testing features.
            y_train (pd.Series): The training labels.
            y_test (pd.Series): The testing labels.
        Returns:
            tuple: A tuple containing the transformed X_train, X_test, y_train_encoded, and y_test_encoded.
        """
        try:
            os.makedirs(self.artifacts_path, exist_ok=True)

            #Labels encoding
            label_freq = y_train.value_counts().index.tolist()
            label_encoder = OrdinalEncoder(categories=[label_freq])

            label_encoder.fit(y_train.to_frame())
            
            y_train_encoded = pd.Series(label_encoder.transform(y_train.to_frame()).flatten(), name="label_idx").astype(int)
            y_test_encoded = pd.Series(label_encoder.transform(y_test.to_frame()).flatten(), name="label_idx").astype(int)
            
            joblib.dump(label_encoder, f"{self.artifacts_path}/label_encoder.joblib")


            #Dividing features to non-normal (to which we'll apply log1p) and normal features and minmax features
            group_log = [
                "Bwd_IAT_Mean","Total_Fwd_Packets", "Total_Backward_Packets", "Total_Length_of_Bwd_Packets", 
                "Avg_Bwd_Segment_Size", "Fwd_Packet_Length_Mean", "Flow_Duration", "Total_Length_of_Fwd_Packets", 
                "Fwd_Packet_Length_Max", "Fwd_Packet_Length_Min", "Fwd_Packet_Length_Std", "Bwd_Packet_Length_Min", 
                "Bwd_Packet_Length_Std", "Flow_Bytes/s", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", 
                "Fwd_IAT_Total", "Fwd_IAT_Mean", "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Std", 
                "Bwd_IAT_Min", "Fwd_Header_Length", "Bwd_Header_Length", "Bwd_Packets/s", "Min_Packet_Length", 
                "Packet_Length_Mean", "Packet_Length_Std", "Down/Up_Ratio", "Subflow_Fwd_Bytes", "act_data_pkt_fwd", 
                "Active_Mean", "Idle_Mean", "fwd_bwd_byte_ratio", "fwd_bwd_packet_ratio", "iat_coeff_variation", "flag_density"
            ]
            group_normal = ["min_seg_size_forward"]
            group_minmax = ["Init_Win_bytes_forward_clean", "Init_Win_bytes_backward_clean"]
            std_cols = group_normal + group_log

            
            for c in group_log:
                X_train[c] = np.log1p(np.clip(X_train[c], 0.0, None))
                X_test[c] = np.log1p(np.clip(X_test[c], 0.0, None))

            #encoding days
            ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            day_encoder = OrdinalEncoder(categories=[ordered_days], handle_unknown='use_encoded_value', unknown_value=-1)
            scaler_std = StandardScaler()
            scaler_minmax = MinMaxScaler()

            # FIT EXCLUSIVEMENT SUR TRAIN
            day_encoder.fit(X_train[['day_of_week_str']])
            scaler_std.fit(X_train[std_cols])
            scaler_minmax.fit(X_train[group_minmax])

            joblib.dump(day_encoder, f"{self.artifacts_path}/day_encoder.joblib")
            joblib.dump(scaler_std, f"{self.artifacts_path}/scaler_std.joblib")
            joblib.dump(scaler_minmax, f"{self.artifacts_path}/scaler_minmax.joblib")

            
            def transform_features(X_df):
                day_idx = day_encoder.transform(X_df[['day_of_week_str']]).astype(int)
                scaled_std = scaler_std.transform(X_df[std_cols])
                scaled_minmax = scaler_minmax.transform(X_df[group_minmax])
                
                features_final = np.hstack([scaled_std, scaled_minmax, day_idx])
                all_ordered_features = std_cols + group_minmax + ["day_of_week_idx"]
                
                df_features = pd.DataFrame(features_final, columns=all_ordered_features, index=X_df.index)
                
                cols_to_keep_end = [
                    "Destination_Port", "FIN_Flag_Count", "SYN_Flag_Count", "PSH_Flag_Count", 
                    "ACK_Flag_Count", "URG_Flag_Count", "CWE_Flag_Count", "ECE_Flag_Count",
                    "has_no_win_scaling_fwd", "has_no_win_scaling_bwd"
                ]
                
                return pd.concat([df_features, X_df[cols_to_keep_end]], axis=1)

            X_train_final = transform_features(X_train)
            X_test_final = transform_features(X_test)

            return X_train_final, X_test_final, y_train_encoded, y_test_encoded

        except Exception as e:
            logger.error(f"Error in encoding and scaling data: {e}")
            raise


