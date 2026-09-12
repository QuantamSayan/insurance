# Here only the validation logic lives
# Nothing less nothing more
# The pydantic class accepts user input, validates and return pydantic class object

from pydantic import BaseModel, computed_field, Field
from typing import Annotated, Literal

class UserInput(BaseModel):
    age: Annotated[int, Field(..., ge=0, description="Age of user")]
    sex: Annotated[Literal["male", "female"], Field(..., description="Sex of user")]
    weight: Annotated[float, Field(..., ge=0, description="Weight of user in kilograms")]
    height: Annotated[float, Field(..., gt=0, description="Height of user in meters")]
    children: Annotated[int, Field(..., ge=-1, description="Number of children the user has")]
    smoker: Annotated[Literal["yes", "no"], Field(..., description="Smoke status of user")]
    region: Annotated[Literal["northeast", "northwest", "southeast", "southwest"], Field(..., description="Region of user")]

    @computed_field
    @property
    def bmi(self) -> float:
        bmi = self.weight/(self.height**2)
        return bmi

class InsuranceOutput(BaseModel):
    prediction: float