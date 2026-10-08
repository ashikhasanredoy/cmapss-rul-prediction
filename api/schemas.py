from typing import List, Optional
from pydantic import BaseModel, Field

class EngineCycleData(BaseModel):
    engine_id: Optional[int] = Field(default=1)
    cycle: Optional[int] = Field(default=1)
    setting_1: float = Field(default=0.0)
    setting_2: float = Field(default=0.0)
    setting_3: float = Field(default=100.0)
    sensor_1: float = Field(default=518.67)
    sensor_2: float = Field(default=642.50)
    sensor_3: float = Field(default=1589.00)
    sensor_4: float = Field(default=1406.00)
    sensor_5: float = Field(default=14.62)
    sensor_6: float = Field(default=21.61)
    sensor_7: float = Field(default=553.50)
    sensor_8: float = Field(default=2388.08)
    sensor_9: float = Field(default=9060.00)
    sensor_10: float = Field(default=1.30)
    sensor_11: float = Field(default=47.45)
    sensor_12: float = Field(default=521.60)
    sensor_13: float = Field(default=2388.08)
    sensor_14: float = Field(default=8140.00)
    sensor_15: float = Field(default=8.43)
    sensor_16: float = Field(default=0.03)
    sensor_17: float = Field(default=392.0)
    sensor_18: float = Field(default=2388.0)
    sensor_19: float = Field(default=100.0)
    sensor_20: float = Field(default=38.85)
    sensor_21: float = Field(default=23.32)

class PredictionResponse(BaseModel):
    engine_id: int
    current_cycle: int
    predicted_rul_cycles: float
    estimated_failure_cycle: float
    health_status: str
    status_indicator: str

class BatchPredictionRequest(BaseModel):
    records: List[EngineCycleData]

class HealthStatusResponse(BaseModel):
    status: str
    model_version: str
    dataset: str
