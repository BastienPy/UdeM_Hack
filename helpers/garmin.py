from garminconnect import Garmin

def import_garmin_data(email, password):
    client = Garmin(email, password)
    client.login()
    
    # Get latest activities
    activities = client.get_activities(0, 20)  # Fetch last 5 activities
    print(activities)
    # Filter the activities to extract relevant data
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
        if activity["activityType"]["typeKey"] == "walking"
    ]
    
    # Return the filtered activities as a list
    return filtered_activities
