from typing import Annotated, Literal

from pydantic import BaseModel, Field


class EmployeeInput(BaseModel):
    Age: int
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    DailyRate: int
    Department: Literal["Research & Development", "Sales", "Human Resources"]
    DistanceFromHome: int
    Education: Annotated[int, Field(ge=1, le=5)]
    EducationField: Literal[
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Human Resources",
        "Other",
    ]
    EmployeeCount: Literal[1] = 1
    EmployeeNumber: int
    EnvironmentSatisfaction: Annotated[int, Field(ge=1, le=4)]
    Gender: Literal["Male", "Female"]
    HourlyRate: int
    JobInvolvement: Annotated[int, Field(ge=1, le=4)]
    JobLevel: Annotated[int, Field(ge=1, le=5)]
    JobRole: Literal[
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources",
    ]
    JobSatisfaction: Annotated[int, Field(ge=1, le=4)]
    MaritalStatus: Literal["Married", "Single", "Divorced"]
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    Over18: Literal["Y", "N"]
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int
    PerformanceRating: Annotated[int, Field(ge=1, le=4)]
    RelationshipSatisfaction: Annotated[int, Field(ge=1, le=4)]
    StandardHours: Literal[80] = 80
    StockOptionLevel: Annotated[int, Field(ge=1, le=3)]
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: Annotated[int, Field(ge=1, le=4)]
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

class PredictionOutput(BaseModel):
    prediction: bool
    probability: float