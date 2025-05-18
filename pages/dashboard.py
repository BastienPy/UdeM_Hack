import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import helper functions from our CSV-based recommendation module.
from helpers.database import get_user, get_poids, get_activities, get_pdv

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def show():

    st.markdown(
        """
        <style>
        .center-text {
            text-align: center;
            font-size: 26px !important; /* This affects all nested elements unless overridden */
            font-weight: bold;
            color: white;
        }
        .highlight {
            color: #006200 !important; /* Only color is changed for highlighted text */
        }
        </style>
        <div class="center-text">
            <h2><span class="highlight">NutriSnap</span> - Your Smart Nutrition Assistant !</h2>
            <p>Scan your fridge, get healthy recipes, track your nutrition, and receive personalized coaching—all in one app!</p>
            <br>
        </div>
        """,
        unsafe_allow_html=True
    )
        
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
            idx_map = {
                "total_fat_PDV": 0,
                "sugar_PDV": 1,
                "sodium_PDV": 2,
                "protein_PDV": 3,
                "saturated_fat_PDV": 4,
                "carbohydrates_PDV": 5,
            }
            idx = idx_map.get(pdv_variable, None)

            # Calculate the total value for the PDV variable today
            if idx is not None:
                total_pdv_value = sum([entry[idx] for entry in pdv_today])

                # Add data to lists for Plotly bar chart
                bar_data.append(total_pdv_value)
                bar_labels.append(pdv_variable.replace('_', ' ').title())
        col1, col2 = st.columns(2)
        with col1:
            # Create the Plotly horizontal bar chart
            fig = go.Figure()

            # Add horizontal bars for each PDV variable
            fig.add_trace(go.Bar(
                x=bar_data,
                y=bar_labels,
                name="Intake",
                marker_color='orange',
                orientation='h'  # Set orientation to horizontal
            ))

            # Add vertical line at 100 for the goal
            fig.add_trace(go.Scatter(
                x=[100] * len(bar_labels),
                y=bar_labels,
                mode='lines',
                name="Goal Line (100%)",
                line=dict(color='green', dash='dash'),
            ))

            # Update layout to make the chart more readable
            fig.update_layout(
                title="Daily PDV Intake",
                yaxis_title="PDV Variables",
                xaxis_title="Value",
                barmode='group',  # Group bars
                showlegend=True,  # Display legend
                template="plotly_dark",  # Use a dark theme (optional)
                height=400  # Adjust height for readability
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)
        with col2:
            

            # Prepare a DataFrame for PDV data
            pdv_df = pd.DataFrame(pdv_data, columns=[
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


    # Get today's activities and calories burned
    today_activities = [activity for activity in activity_data if activity[1].startswith(today)]
    total_calories_today = sum([activity[2] for activity in today_activities]) if today_activities else 0

    # Calories burned today graph using Plotly
    #st.write("**Calories burned today**")
    # Set a reasonable daily calorie goal
    calorie_goal = 1692  # Change this based on user or recommendation
    calorie_progress = total_calories_today / calorie_goal * 100

    col1, col2 = st.columns(2)
    with col1:
        # Récupération des activités
        username = st.session_state["user"]
        activity_data = get_activities(username)

        # Convertir les données en DataFrame
        df_activities = pd.DataFrame(activity_data, columns=["Activity", "Start Time", "Calories", "BMR Calories", "Steps"])
        df_activities["Start Time"] = pd.to_datetime(df_activities["Start Time"], errors='coerce')

        if df_activities.empty:
            st.write("No activity data available.")
        else:
            # Grouper les calories brûlées par jour
            df_activities["Date"] = df_activities["Start Time"].dt.date
            daily_calories = df_activities.groupby("Date")["Calories"].sum().reset_index()

            # Création du graphique avec Plotly
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=daily_calories["Date"],
                y=daily_calories["Calories"],
                mode='lines+markers',
                name="Calories Burned",
                line=dict(color='orange')
            ))

            # Ajouter une ligne horizontale pour l'objectif journalier de 2000 calories
            fig1.add_trace(go.Scatter(
                x=daily_calories["Date"],
                y=[calorie_goal] * len(daily_calories),
                mode='lines',
                name=f"Goal ({calorie_goal} kcal)",
                line=dict(color='gray', dash='dash'),
            ))

            fig1.update_layout(
                title="Calories Burned Since the Beginning",
                xaxis_title="Date",
                yaxis_title="Calories Burned",
                xaxis=dict(tickformat="%Y-%m-%d"),
                showlegend=True,
                template="plotly_dark"
            )

            st.plotly_chart(fig1)

    with col2:
        
        # Weight and BMI data visualization using Plotly
        height_m = user_info[5] / 100  # Assuming height is at index 5 in user tuple, convert cm to meters
        df_weight = pd.DataFrame(weight_data, columns=["Weight", "Date"])
        df_weight["Date"] = pd.to_datetime(df_weight["Date"])
        df_weight["BMI"] = df_weight["Weight"] / (height_m ** 2)
            
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

    # Récupération des activités
    username = st.session_state["user"]
    activity_data = get_activities(username)

    # Convertir les données en DataFrame
    df_activities = pd.DataFrame(activity_data, columns=["Activity", "Start Time", "Calories", "BMR Calories", "Steps"])
    df_activities["Start Time"] = pd.to_datetime(df_activities["Start Time"], errors='coerce')

    # Filtrer les activités depuis le début
    df_last_month = df_activities

    # Compter la fréquence des types d'activités
    activity_counts = df_last_month["Activity"].value_counts().reset_index()
    activity_counts.columns = ["Activity", "Count"]

    # Créer le pie chart avec Plotly
    fig = px.pie(activity_counts, values="Count", names="Activity", title=None)

    fig.update_traces(textinfo='percent+label', textposition='inside')

    fig.update_layout(
        title=dict(
            text="Breakdown of Activities Over the Past Month",
            x=0.5,   
            xanchor="center"
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2
        ),
        margin=dict(t=80)        
    )

    st.plotly_chart(fig)