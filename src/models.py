# models.py
from typing import Optional
from pydantic import BaseModel, Field

class CalendarEvent(BaseModel):
    name: str = Field(..., description="Name of the event")
    date: str = Field(..., description="Date of the event in YYYY-MM-DD format")
    time: str = Field(..., description="Time of the event in HH:MM (24-hour) format")
    timezone: str = Field(..., description="IANA timezone string")
    description: Optional[str] = Field(None, description="Description of the event")
    meeting_link: Optional[str] = Field(None, description="Link to the meeting if applicable")

class EventIdentifier(BaseModel):
    identifier: str = Field(..., description="Identifier for the event, can be name or description")

class RescheduleDetails(BaseModel):
    identifier: str = Field(..., description="Identifier for the event to reschedule")
    new_date: Optional[str] = Field(None, description="New date in YYYY-MM-DD format")
    new_time: Optional[str] = Field(None, description="New time in HH:MM (24-hour) format")
    new_timezone: Optional[str] = Field(None, description="New IANA timezone string")