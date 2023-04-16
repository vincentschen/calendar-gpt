from typing import List, NamedTuple

import pandas as pd
from icalendar import Calendar, Event


class CalendarEvent(NamedTuple):
    summary: str
    description: str
    location: str
    start_time: str
    end_time: str
    duration: str
    url: str
    attendees: List[str]
    organizer: str


def calendar_row_to_tsv(row):
    return "\t".join(row.astype(str).values.flatten())


def events_to_df(events):
    return pd.DataFrame.from_records(events, columns=CalendarEvent._fields)


def file_to_events(content: str) -> list:
    cal = Calendar.from_ical(content)
    events = []
    for component in cal.walk():
        if not isinstance(component, Event):
            continue

        event = parse_event(component)
        events.append(event)
    return events


def parse_event(component: Event) -> CalendarEvent:
    if component.get("dtstart"):
        event_start = component.get("dtstart").dt
    else:
        event_start = "None"
    if component.get("dtend"):
        event_end = component.get("dtend").dt
    else:
        event_end = "None"

    try:
        duration_s = (event_end - event_start).seconds
    except TypeError:
        duration_s = float("nan")
    f = "%Y-%m-%d %H:%M:%S"  # Time formatter

    try:
        start_time = event_start.strftime(f)
    except AttributeError:
        start_time = "None"

    try:
        end_time = event_end.strftime(f)
    except AttributeError:
        end_time = "None"

    organizer_val = component.get("ORGANIZER")
    organizer_str = str(organizer_val).replace("mailto:", "") if organizer_val else ""

    attendees_val = component.get("ATTENDEE")
    if attendees_val:
        if isinstance(attendees_val, list):
            attendees_str = str([str(x).replace("mailto:", "") for x in attendees_val])
        else:
            attendees_str = str(attendees_val).replace("mailto:", "")
    else:
        attendees_str = ""

    event = CalendarEvent(
        summary=component.get("SUMMARY"),
        description=component.get("DESCRIPTION"),
        location=component.get("LOCATION"),
        start_time=start_time,
        end_time=end_time,
        duration="{:02}:{:02}:{:02}".format(
            duration_s // 3600, duration_s % 3600 // 60, duration_s % 60
        ),
        url=component.get("EVENT"),
        attendees=attendees_str,
        organizer=organizer_str,
    )
    return event
