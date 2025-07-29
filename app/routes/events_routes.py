from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
from app.schemas.schema import EventCreate, EventDB, EventUpdate
from app.services.service import (
    create_event,
    get_event_by_id,
    get_all_events,
    get_filtered_events,
    update_event,
    delete_event
)

router = APIRouter()

@router.post("/", response_model=EventDB, status_code=status.HTTP_201_CREATED)
async def create_event_route(event_data: EventCreate):
    new_event = await create_event(event_data)
    return new_event

@router.get("/", response_model=List[EventDB])
async def get_all_events_route(
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD format)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD format)"),
    event_type: Optional[str] = Query(None, description="Event type filter (exit/entrance/both)"),
    limit: Optional[int] = Query(100, description="Maximum number of events to return"),
    skip: Optional[int] = Query(0, description="Number of events to skip (pagination)"),
    sort_by: Optional[str] = Query("time_out", description="Sort field (time_out, time_in)"),
    sort_order: Optional[str] = Query("desc", description="Sort order (asc/desc)")
):
    """
    Get events with optional filtering and pagination
    
    Parameters:
    - start_date: Filter events from this date (YYYY-MM-DD)
    - end_date: Filter events until this date (YYYY-MM-DD)
    - event_type: Filter by event type (exit=only exits, entrance=only returns, both=all)
    - limit: Maximum number of events to return (default: 100)
    - skip: Number of events to skip for pagination (default: 0)
    - sort_by: Field to sort by (default: time_out)
    - sort_order: Sort order - asc or desc (default: desc)
    """
    
    # If no filters are provided, use the original function
    if not any([start_date, end_date, event_type]):
        return await get_all_events()
    
    # Parse date strings to datetime objects
    start_datetime = None
    end_datetime = None
    
    if start_date:
        try:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
    
    if end_date:
        try:
            # Set end date to end of day
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
    
    # Validate event_type
    if event_type and event_type not in ["exit", "entrance", "both"]:
        raise HTTPException(status_code=400, detail="Invalid event_type. Use 'exit', 'entrance', or 'both'")
    
    # Validate sort parameters
    if sort_by not in ["time_out", "time_in"]:
        raise HTTPException(status_code=400, detail="Invalid sort_by. Use 'time_out' or 'time_in'")
    
    if sort_order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Invalid sort_order. Use 'asc' or 'desc'")
    
    return await get_filtered_events(
        start_date=start_datetime,
        end_date=end_datetime,
        event_type=event_type,
        limit=limit,
        skip=skip,
        sort_by=sort_by,
        sort_order=sort_order
    )

@router.get("/{event_id}", response_model=EventDB)
async def get_event_route(event_id: str):
    ev = await get_event_by_id(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return ev

@router.put("/{event_id}", response_model=EventDB)
async def update_event_route(event_id: str, event_data: EventUpdate):
    updated = await update_event(event_id, event_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Event not found or no changes")
    return updated

@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event_route(event_id: str):
    deleted_count = await delete_event(event_id)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return None

@router.get("/statistics/summary")
async def get_events_statistics(
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD format)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD format)")
):
    """
    Get event statistics and summary information
    
    Returns counts of different event types, activity patterns, etc.
    """
    from app.services.service import get_events_statistics
    
    # Parse date strings to datetime objects
    start_datetime = None
    end_datetime = None
    
    if start_date:
        try:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
    
    if end_date:
        try:
            # Set end date to end of day
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
    
    return await get_events_statistics(start_datetime, end_datetime)

@router.get("/recent")
async def get_recent_events(
    limit: Optional[int] = Query(10, description="Number of recent events to return")
):
    """
    Get the most recent events (for dashboard/home page)
    """
    return await get_filtered_events(
        limit=limit,
        skip=0,
        sort_by="time_out",
        sort_order="desc"
    )

@router.get("/today")
async def get_today_events():
    """
    Get all events from today
    """
    from datetime import date
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    
    # Get events from today
    start_datetime = datetime.strptime(today_str, "%Y-%m-%d")
    end_datetime = start_datetime.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return await get_filtered_events(
        start_date=start_datetime,
        end_date=end_datetime,
        sort_by="time_out",
        sort_order="desc"
    )
