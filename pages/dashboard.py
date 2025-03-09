import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import helper functions from our CSV-based recommendation module.
from helpers.database import add_activity, get_user, get_garmin_id, get_poids, get_activities, get_pdv

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def show():
    st.title("📊 Dashboard")
    
    username = st.session_state['user']
    user_info = get_user(username)
    
    if not user_info:
        st.write('User not found.')
        return

    user_id = user_info[0]  # Replace with actual user ID fetching logic
    weight_data = get_poids(user_id)
    activity_data = get_activities(username)  # Fetch activity data
    pdv_data = get_pdv(user_id)  # Fetch PDV data for the user
    
    if not weight_data or not user_info:
        st.write("No weight data available.")
        return
    
    # Get today's date
    today = pd.to_datetime("today").strftime("%Y-%m-%d")
    
    # Filter PDV data for today's date
    pdv_today = [entry for entry in pdv_data if entry[7].startswith(today)]  # Assuming date is the 8th element
    
    # Progress bars for PDV variables
    pdv_max_values = {
        'total_fat_PDV': 100,  # Recommended value, change as necessary
        'sugar_PDV': 100,      # Recommended value, change as necessary
        'sodium_PDV': 100,     # Recommended value, change as necessary
        'protein_PDV': 100,    # Recommended value, change as necessary
        'saturated_fat_PDV': 100,  # Recommended value, change as necessary
        'carbohydrates_PDV': 100,  # Recommended value, change as necessary
    }

    # Loop through the PDV variables and show progress
    if pdv_today:
        # Data for bar chart
        bar_data = []
        bar_labels = []
        
        # Loop through all PDV variables
        for pdv_variable, max_value in pdv_max_values.items():
            # Map PDV variable to its index
            if pdv_variable == "total_fat_PDV":
                idx = 0
            elif pdv_variable == "sugar_PDV":
                idx = 1
            elif pdv_variable == "sodium_PDV":
                idx = 2
            elif pdv_variable == "protein_PDV":
                idx = 3
            elif pdv_variable == "saturated_fat_PDV":
                idx = 4
            elif pdv_variable == "carbohydrates_PDV":
                idx = 5

            # Calculate the total value for the PDV variable today
            total_pdv_value = sum([entry[idx] for entry in pdv_today])

            # Add data to lists for Plotly bar chart
            bar_data.append(total_pdv_value)
            bar_labels.append(pdv_variable.replace('_', ' ').title())

        # Create the Plotly bar chart
        fig = go.Figure()

        # Add bars for each PDV variable
        fig.add_trace(go.Bar(
            x=bar_labels,
            y=bar_data,
            name="Intake",
            marker_color='orange',
        ))

        # Add horizontal line at 100 for the goal
        fig.add_trace(go.Scatter(
            x=bar_labels,
            y=[100] * len(bar_labels),
            mode='lines',
            name="Goal Line (100%)",
            line=dict(color='black', dash='dash'),
        ))

        # Update layout to make the chart more readable
        fig.update_layout(
            title="Daily PDV Intake",
            xaxis_title="PDV Variables",
            yaxis_title="Value",
            barmode='group',  # Side by side bars for each PDV variable
            xaxis_tickangle=-45,
            showlegend=True,  # Display legend
            template="plotly_dark"  # Use a dark theme (optional)
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

    # Get today's activities and calories burned
    today_activities = [activity for activity in activity_data if activity[1].startswith(today)]
    total_calories_today = sum([activity[2] for activity in today_activities]) if today_activities else 0

    # Calories burned today graph using Plotly
    #st.write("**Calories burned today**")
    # Set a reasonable daily calorie goal
    calorie_goal = 2000  # Change this based on user or recommendation
    calorie_progress = total_calories_today / calorie_goal * 100

    fig1 = go.Figure()

    # Bar for calories burned today
    fig1.add_trace(go.Bar(
        x=["Calories burned today"],
        y=[total_calories_today],
        name="Calories Burned",
        marker_color='orange'
    ))

    # Horizontal line for the goal (2000 calories)
    fig1.add_trace(go.Scatter(
        x=["Calories burned today"],
        y=[calorie_goal],
        mode='lines',
        name="Goal (2000)",
        line=dict(color='gray', dash='dash'),
    ))

    fig1.update_layout(
        title="Calories Burned Today",
        xaxis_title="Activity",
        yaxis_title="Calories",
        showlegend=True
    )

    st.plotly_chart(fig1)

    # Weight and BMI data visualization using Plotly
    height_m = user_info[5] / 100  # Assuming height is at index 5 in user tuple, convert cm to meters
    df_weight = pd.DataFrame(weight_data, columns=["Weight", "Date"])
    df_weight["Date"] = pd.to_datetime(df_weight["Date"])
    df_weight["BMI"] = df_weight["Weight"] / (height_m ** 2)

    # Layout: Create two columns for the two graphs
    col1, col2 = st.columns(2)

    # First column (Weight and BMI)
    with col1:
        fig2 = go.Figure()

        # Weight trace
        fig2.add_trace(go.Scatter(
            x=df_weight["Date"],
            y=df_weight["Weight"],
            mode='lines+markers',
            name="Weight (kg)",
            line=dict(color='blue'),
            marker=dict(symbol='circle')
        ))

        # BMI trace
        fig2.add_trace(go.Scatter(
            x=df_weight["Date"],
            y=df_weight["BMI"],
            mode='lines+markers',
            name="BMI",
            line=dict(color='red'),
            marker=dict(symbol='square')
        ))

        fig2.update_layout(
            title="Weight and BMI Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            showlegend=True
        )

        st.plotly_chart(fig2)

    # Second column (Activity calories)
    with col2:
        df_activities = pd.DataFrame(activity_data, columns=["Activity", "Start Time", "Calories", "BMR Calories", "Steps"])
        df_activities["Start Time"] = pd.to_datetime(df_activities["Start Time"], errors='coerce')

        if df_activities.empty:
            st.write("No activity data available.")
        else:
            fig3 = go.Figure()

            # Plot activity calories
            fig3.add_trace(go.Scatter(
                x=df_activities["Start Time"],
                y=df_activities["Calories"],
                mode='lines+markers',
                name="Calories Burned",
                line=dict(color='green'),
                marker=dict(symbol='circle')
            ))

            fig3.update_layout(
                title="Activity Calories Burned",
                xaxis_title="Date",
                yaxis_title="Calories Burned",
                showlegend=True
            )

            st.plotly_chart(fig3)

    # PDV Temporal Graph using Plotly
    st.write("**PDV Temporal Graph**")

    # Prepare a DataFrame for PDV data
    pdv_df = pd.DataFrame(pdv_today, columns=[
        'total_fat_PDV', 'sugar_PDV', 'sodium_PDV', 'protein_PDV', 
        'saturated_fat_PDV', 'carbohydrates_PDV', 'some_other_variable', 'Date'
    ])

    # Convert the "Date" column to datetime format
    pdv_df['Date'] = pd.to_datetime(pdv_df['Date'])

    # Check that we have data
    if not pdv_df.empty:
        fig4 = go.Figure()

        # Plot each PDV variable
        for pdv_variable in pdv_max_values.keys():
            fig4.add_trace(go.Scatter(
                x=pdv_df['Date'],
                y=pdv_df[pdv_variable],
                mode='lines+markers',
                name=pdv_variable.replace('_', ' ').title()
            ))

        # Horizontal line at 100 to represent the goal
        fig4.add_trace(go.Scatter(
            x=pdv_df['Date'],
            y=[100] * len(pdv_df),
            mode='lines',
            name="Goal (100%)",
            line=dict(color='gray', dash='dash'),
        ))

        fig4.update_layout(
            title="Daily Temporal PDV Values",
            xaxis_title="Date",
            yaxis_title="PDV (%)",
            showlegend=True
        )

        st.plotly_chart(fig4)

    if "user" not in st.session_state or not st.session_state["user"]:
            st.warning("You must be logged in to access this page.")
            return

    username = st.session_state["user"]
    user = get_user(username)

    if user:  # Vérifier que l'utilisateur a bien été ajouté
        user_id = user[0]
    
    garmin_id, garmin_password = get_garmin_id(user_id)

    if st.button('Update Data'):
        if not garmin_id or not garmin_password:
            st.error("❌ Garmin ID or password not found")
        else:
            update = add_activity(user_id, garmin_id, 'eUMckMQM94!!')
            try:
                if update:
                    st.write("✅ Data updated successfully!")
                else:
                    st.write(f"❌ An error occurred with the API")
            except Exception as e:
                st.write(f"❌ An error occurred with the API")
            
