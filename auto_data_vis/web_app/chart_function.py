import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_data(x_values, y_values, predicted_label):
    if predicted_label == 'pie':
        return plot_pie_chart(x_values, y_values)
    elif predicted_label == 'scatter':
        return plot_scatter_chart(x_values, y_values)
    elif predicted_label == 'line':
        return plot_line_chart(x_values, y_values)
    elif predicted_label == 'bar':
        return plot_bar_chart(x_values, y_values)
    else:
        print("Error! Predicted Label incorrect")
        return

# Function to plot pie chart
def plot_pie_chart(x_values, y_values):
    fig, ax = plt.subplots()
    # Custom function to display actual values
    def absolute_value(val):
        return int(val / 100. * np.sum(y_values))
    
    ax.pie(y_values, labels=x_values, autopct=lambda p: f'{absolute_value(p)}', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# Function to plot scatter plot with grid lines
def plot_scatter_chart(x_values, y_values):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(x_values)), y_values, s=5)  # Scatter plot with small points
    #ax.plot(np.arange(len(x_values)), y_values, linestyle='-', color='blue')  # Adding a line connecting the points
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_title('Scatter Plot')
    
    if len(x_values) > 10:  # If too many data points, reduce tick frequency
        step = max(1, len(x_values) // 10)  # Show approximately 10 ticks
        ax.set_xticks(np.arange(0, len(x_values), step))
        ax.set_xticklabels(x_values[::step], rotation=-90, fontsize=8)
    else:
        ax.set_xticks(np.arange(len(x_values)))  # Ensure ticks match the length of x_values
        ax.set_xticklabels(x_values, rotation=-90, fontsize=8)  # Custom labels for x-values

    ax.grid(True)  # Adding grid lines
    st.pyplot(fig)

# Function to plot line chart with grid lines
def plot_line_chart(x_values, y_values):
    def is_date(string):
        try:
            datetime.strptime(string, "%Y-%m-%d")
            return True
        except ValueError:
            return False
        
    if all(isinstance(x, (int, float, np.number)) for x in x_values):
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        sorted_indices = np.argsort(x_values)
        x_values = x_values[sorted_indices]
        y_values = y_values[sorted_indices]
        
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_title('Line Chart')

    if len(x_values) > 10:
        step = max(1, len(x_values) // 10)
        ax.set_xticks(np.arange(0, len(x_values), step))
        ax.set_xticklabels(x_values[::step], rotation=-90, fontsize=8)
    elif any(isinstance(x, str) for x in x_values):
        ax.set_xticks(np.arange(len(x_values)))  # Ensure ticks match the length of x_values
        ax.set_xticklabels(x_values, rotation=-90)

    ax.grid(True)  # Adding grid lines
    st.pyplot(fig)

# Function to plot bar chart with grid lines
def plot_bar_chart(x_values, y_values):
    fig, ax = plt.subplots()
    ax.bar(x_values, y_values)
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.set_title('Bar Chart')
    if len(x_values) > 10:
        step = max(1, len(x_values) // 10)
        ax.set_xticks(np.arange(0, len(x_values), step))
        ax.set_xticklabels(x_values[::step], rotation=-90, fontsize=8)
    else:
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels(x_values, rotation=-90)

    ax.grid(True)  # Adding grid lines
    st.pyplot(fig)
