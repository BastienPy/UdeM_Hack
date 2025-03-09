import json
from garminconnect import Garmin

def import_garmin_data(email, password):
    client = Garmin(email, password)
    client.login()
    
    # Get latest activities
    activities = client.get_activities(0, 5)  # Fetch last 5 activities
    # Convert activities to JSON
    activities_json = json.dumps(activities, indent=4)

    # Save JSON to a file
    with open("activities.json", "w") as json_file:
        filtered_activities = [
            {
                "activityId": activity["activityId"],
                "activityName": activity["activityName"],
                "startTimeLocal": activity["startTimeLocal"],
                "calories": activity["calories"],
                "bmrCalories": activity["bmrCalories"],
                "steps": activity["steps"]
            }
            for activity in activities
        ]
        activities_json = json.dumps(filtered_activities, indent=4)
        json_file.write(activities_json)

mail = "alexis.bord094@gmail.com"
password = "eUMckMQM94!!"

import_garmin_data(mail, password)