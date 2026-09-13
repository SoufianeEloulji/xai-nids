from src import logger
from abc import ABC, abstractmethod
import pandas as pd
import re
import numpy as np
import joblib
from typing import Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder



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


            df_fe["Init_Win_bytes_forward_clean"] = np.where(df_fe["Init_Win_bytes_forward"] == -1, 0, df_fe["Init_Win_bytes_forward"])
            df_fe["Init_Win_bytes_backward_clean"] = np.where(df_fe["Init_Win_bytes_backward"] == -1, 0, df_fe["Init_Win_bytes_backward"])


            #deleting redundant columns + clumns containing 0 only
            cols_to_drop = [
                "Active_Max", "Active_Min", "Idle_Std", "Idle_Max", "Idle_Min", "Fwd_PSH_Flags", 
                "Average_Packet_Size", "Packet_Length_Variance", "RST_Flag_Count", "Avg_Fwd_Segment_Size", 
                "Bwd_IAT_Max", "Bwd_IAT_Total", "Subflow_Bwd_Packets", "Subflow_Fwd_Packets", 
                "Subflow_Bwd_Bytes", "Bwd_Packet_Length_Mean", "Bwd_Packet_Length_Max", "Max_Packet_Length", 
                "Fwd_URG_Flags", "Fwd_Packets/s", "Flow_Packets/s", "Flow_IAT_Mean", "Active_Std", "Init_Win_bytes_forward", "Init_Win_bytes_backward", 
                "Bwd_Avg_Bulk_Rate", "Bwd_Avg_Packets/Bulk", "Bwd_Avg_Bytes/Bulk", "Fwd_Avg_Bulk_Rate", 
                "Fwd_Avg_Packets/Bulk", "Fwd_Avg_Bytes/Bulk", "Bwd_URG_Flags", "Bwd_PSH_Flags"
            ]
                            
            df_fe = df_fe.drop(columns=cols_to_drop)
            return df_fe

        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}")
            raise  


class DataEncodingScalingStrategy(DataStrategy):
    """Concrete strategy for encoding and scaling data."""
    
    def __init__(self, artifacts_path: str = "artifacts/"):
        self.artifacts_path = artifacts_path
        self.std_scaler = joblib.load(f"{self.artifacts_path}/scaler_std.joblib")
        self.minmax_scaler = joblib.load(f"{self.artifacts_path}/scaler_minmax.joblib")

    def handle_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

            log_cols = ["Bwd_IAT_Mean","Total_Fwd_Packets", "Total_Backward_Packets", "Total_Length_of_Bwd_Packets", 
                                    "Avg_Bwd_Segment_Size", "Fwd_Packet_Length_Mean", "Flow_Duration", "Total_Length_of_Fwd_Packets", 
                                    "Fwd_Packet_Length_Max", "Fwd_Packet_Length_Min", "Fwd_Packet_Length_Std", "Bwd_Packet_Length_Min", 
                                    "Bwd_Packet_Length_Std", "Flow_Bytes/s", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", 
                                    "Fwd_IAT_Total", "Fwd_IAT_Mean", "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Std", 
                                    "Bwd_IAT_Min", "Fwd_Header_Length", "Bwd_Header_Length", "Bwd_Packets/s", "Min_Packet_Length", 
                                    "Packet_Length_Mean", "Packet_Length_Std", "Down/Up_Ratio", "Subflow_Fwd_Bytes", "act_data_pkt_fwd", 
                                    "Active_Mean", "Idle_Mean", "fwd_bwd_byte_ratio", "fwd_bwd_packet_ratio", "iat_coeff_variation", "flag_density"]
            
            std_scaled_cols = ["min_seg_size_forward","Bwd_IAT_Mean","Total_Fwd_Packets", "Total_Backward_Packets", "Total_Length_of_Bwd_Packets", 
                                    "Avg_Bwd_Segment_Size", "Fwd_Packet_Length_Mean", "Flow_Duration", "Total_Length_of_Fwd_Packets", 
                                    "Fwd_Packet_Length_Max", "Fwd_Packet_Length_Min", "Fwd_Packet_Length_Std", "Bwd_Packet_Length_Min", 
                                    "Bwd_Packet_Length_Std", "Flow_Bytes/s", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", 
                                    "Fwd_IAT_Total", "Fwd_IAT_Mean", "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Std", 
                                    "Bwd_IAT_Min", "Fwd_Header_Length", "Bwd_Header_Length", "Bwd_Packets/s", "Min_Packet_Length", 
                                    "Packet_Length_Mean", "Packet_Length_Std", "Down/Up_Ratio", "Subflow_Fwd_Bytes", "act_data_pkt_fwd", 
                                    "Active_Mean", "Idle_Mean", "fwd_bwd_byte_ratio", "fwd_bwd_packet_ratio", "iat_coeff_variation", "flag_density"]
            minmax_scaled_cols = ["Init_Win_bytes_forward_clean", "Init_Win_bytes_backward_clean"]


            def transform_features(X_df):
                scaled_data = X_df.copy()
                scaled_data[minmax_scaled_cols] = self.minmax_scaler.transform(scaled_data[minmax_scaled_cols])
                
                for c in log_cols:
                    scaled_data[c] = np.log1p(np.clip(scaled_data[c], 0.0, None))
                
                scaled_data[std_scaled_cols] = self.std_scaler.transform(scaled_data[std_scaled_cols])

                return scaled_data

            df_final = transform_features(df)

            return df_final
        except Exception as e:
            logger.error(f"Error in encoding and scaling data: {e}")
            raise







