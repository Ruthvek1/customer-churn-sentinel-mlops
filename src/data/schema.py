"""
Pydantic data schemas for input validation and type safety.

Defines the expected structure of customer data at every stage
of the pipeline — from raw input to API request/response.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums for categorical features ---

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class YesNo(str, Enum):
    YES = "Yes"
    NO = "No"


class InternetService(str, Enum):
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"


class Contract(str, Enum):
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class PaymentMethod(str, Enum):
    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class MultiLineService(str, Enum):
    YES = "Yes"
    NO = "No"
    NO_PHONE_SERVICE = "No phone service"


class InternetDependentService(str, Enum):
    YES = "Yes"
    NO = "No"
    NO_INTERNET_SERVICE = "No internet service"


# --- Data Schemas ---

class CustomerInput(BaseModel):
    """Schema for raw customer input data (matches Telco dataset columns)."""
    
    customerID: Optional[str] = Field(None, description="Unique customer identifier")
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether the customer is a senior citizen (0/1)")
    Partner: str = Field(..., description="Whether the customer has a partner")
    Dependents: str = Field(..., description="Whether the customer has dependents")
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed")
    PhoneService: str = Field(..., description="Whether the customer has phone service")
    MultipleLines: str = Field(..., description="Whether the customer has multiple lines")
    InternetService: str = Field(..., description="Customer's internet service provider")
    OnlineSecurity: str = Field(..., description="Whether the customer has online security")
    OnlineBackup: str = Field(..., description="Whether the customer has online backup")
    DeviceProtection: str = Field(..., description="Whether the customer has device protection")
    TechSupport: str = Field(..., description="Whether the customer has tech support")
    StreamingTV: str = Field(..., description="Whether the customer has streaming TV")
    StreamingMovies: str = Field(..., description="Whether the customer has streaming movies")
    Contract: str = Field(..., description="The contract term of the customer")
    PaperlessBilling: str = Field(..., description="Whether the customer has paperless billing")
    PaymentMethod: str = Field(..., description="The customer's payment method")
    MonthlyCharges: float = Field(..., ge=0, description="The amount charged monthly")
    TotalCharges: float = Field(..., ge=0, description="The total amount charged")

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }


class ProcessedFeatures(BaseModel):
    """Schema for processed features ready for model input."""
    
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    tenure_bucket: int
    monthly_to_total_ratio: float
    service_count: int
    
    # Encoded categorical features will be added dynamically
    # based on the preprocessing pipeline

    class Config:
        extra = "allow"  # Allow additional encoded features
