"""Patient context schema — optional clinical metadata that conditions C4."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PatientContext(BaseModel):
    age: Optional[int] = None
    smoker: Optional[bool] = None
    diabetic: Optional[bool] = None
    last_visit_months: Optional[int] = None
    existing_restorations: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    chief_complaint: Optional[str] = None
