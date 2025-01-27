"""
ב-MongoDB אין ORM קשיח, אבל ניתן להגדיר כאן מחלקות עזר (Models),
או פונקציות עזר שקשורות ל"Event" ברמה לוגית.
"""

class EventModel:
    def __init__(self, time_out, time_in, video_url):
        self.time_out = time_out
        self.time_in = time_in
        self.video_url = video_url
