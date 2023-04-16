# CalendarGPT
An AI that "indexes your life" and helps you understand how you spend your time.

## Usage
* Setup: https://www.loom.com/share/a5e8934f779040f2b78c713785556f5b
* Q&A: https://www.loom.com/share/d9eb347eb74041329a09e740b3dd8ecf

## Dev Quickstart
### First time
```bash
python -m venv .env  # create an env
source .env/bin/activate  # activate env
pip install -r requirements.txt  # install requirements
```

### Ongoing
```bash
export OPENAI_API_KEY=MY_API_KEY
source .env/bin/activate  # activate env
streamlit run app/app.py
```

### Start from Scratch
```bash
rm -rf db
```

## Resources
* Exporting GCal to `.ics`: https://support.google.com/calendar/answer/37111?hl=en
