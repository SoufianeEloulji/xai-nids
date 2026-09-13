from typing import Optional, List
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    Destination_Port: Optional[int] = None
    Flow_Duration: Optional[int] = None
    Total_Fwd_Packets: Optional[int] = None
    Total_Backward_Packets: Optional[int] = None
    Total_Length_of_Fwd_Packets: Optional[int] = None
    Total_Length_of_Bwd_Packets: Optional[int] = None
    Fwd_Packet_Length_Max: Optional[int] = None
    Fwd_Packet_Length_Min: Optional[int] = None
    Fwd_Packet_Length_Mean: Optional[float] = None
    Fwd_Packet_Length_Std: Optional[float] = None
    Bwd_Packet_Length_Max: Optional[int] = None
    Bwd_Packet_Length_Min: Optional[int] = None
    Bwd_Packet_Length_Mean: Optional[float] = None
    Bwd_Packet_Length_Std: Optional[float] = None
    Flow_Bytes_s: Optional[float] = Field(default=None, alias="Flow_Bytes/s")
    Flow_Packets_s: Optional[float] = Field(default=None, alias="Flow_Packets/s")
    Flow_IAT_Mean: Optional[float] = None
    Flow_IAT_Std: Optional[float] = None
    Flow_IAT_Max: Optional[int] = None
    Flow_IAT_Min: Optional[int] = None
    Fwd_IAT_Total: Optional[int] = None
    Fwd_IAT_Mean: Optional[float] = None
    Fwd_IAT_Std: Optional[float] = None
    Fwd_IAT_Max: Optional[int] = None
    Fwd_IAT_Min: Optional[int] = None
    Bwd_IAT_Total: Optional[int] = None
    Bwd_IAT_Mean: Optional[float] = None
    Bwd_IAT_Std: Optional[float] = None
    Bwd_IAT_Max: Optional[int] = None
    Bwd_IAT_Min: Optional[int] = None
    Fwd_PSH_Flags: Optional[int] = None
    Bwd_PSH_Flags: Optional[int] = None
    Fwd_URG_Flags: Optional[int] = None
    Bwd_URG_Flags: Optional[int] = None
    Fwd_Header_Length: Optional[int] = None
    Bwd_Header_Length: Optional[int] = None
    Fwd_Packets_s: Optional[float] = Field(default=None, alias="Fwd_Packets/s")
    Bwd_Packets_s: Optional[float] = Field(default=None, alias="Bwd_Packets/s")
    Min_Packet_Length: Optional[int] = None
    Max_Packet_Length: Optional[int] = None
    Packet_Length_Mean: Optional[float] = None
    Packet_Length_Std: Optional[float] = None
    Packet_Length_Variance: Optional[float] = None
    FIN_Flag_Count: Optional[int] = None
    SYN_Flag_Count: Optional[int] = None
    RST_Flag_Count: Optional[int] = None
    PSH_Flag_Count: Optional[int] = None
    ACK_Flag_Count: Optional[int] = None
    URG_Flag_Count: Optional[int] = None
    CWE_Flag_Count: Optional[int] = None
    ECE_Flag_Count: Optional[int] = None
    Down_Up_Ratio: Optional[int] = Field(default=None, alias="Down/Up_Ratio")
    Average_Packet_Size: Optional[float] = None
    Avg_Fwd_Segment_Size: Optional[float] = None
    Avg_Bwd_Segment_Size: Optional[float] = None
    Fwd_Avg_Bytes_Bulk: Optional[int] = Field(default=None, alias="Fwd_Avg_Bytes/Bulk")
    Fwd_Avg_Packets_Bulk: Optional[int] = Field(default=None, alias="Fwd_Avg_Packets/Bulk")
    Fwd_Avg_Bulk_Rate: Optional[int] = None
    Bwd_Avg_Bytes_Bulk: Optional[int] = Field(default=None, alias="Bwd_Avg_Bytes/Bulk")
    Bwd_Avg_Packets_Bulk: Optional[int] = Field(default=None, alias="Bwd_Avg_Packets/Bulk")
    Bwd_Avg_Bulk_Rate: Optional[int] = None
    Subflow_Fwd_Packets: Optional[int] = None
    Subflow_Fwd_Bytes: Optional[int] = None
    Subflow_Bwd_Packets: Optional[int] = None
    Subflow_Bwd_Bytes: Optional[int] = None
    Init_Win_bytes_forward: Optional[int] = None
    Init_Win_bytes_backward: Optional[int] = None
    act_data_pkt_fwd: Optional[int] = None
    min_seg_size_forward: Optional[int] = None
    Active_Mean: Optional[float] = None
    Active_Std: Optional[float] = None
    Active_Max: Optional[int] = None
    Active_Min: Optional[int] = None
    Idle_Mean: Optional[float] = None
    Idle_Std: Optional[float] = None
    Idle_Max: Optional[int] = None
    Idle_Min: Optional[int] = None
    day_of_week_idx: Optional[int] = None

    class Config:
        populate_by_name = True


class FeatureContribution(BaseModel):
    """
    Contribution of a feature to the prediction, as computed by SHAP.
    """
    feature: str
    shap_value: float
    value: Optional[float] = None  


class PredictionOutput(BaseModel):
    class_id: int
    Label: str
    confidence: float
    top_features: List[FeatureContribution] = []
