import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Task="Write project proposal and record presentation", Start='2024-02-11', Finish='2024-02-18', Owner="Team"),
    dict(Task="Research available social media APIs", Start='2024-02-18', Finish='2024-02-25', Owner="Team"),
    dict(Task="Collect social media data and company 10-Ks", Start='2024-02-25', Finish='2024-03-10', Owner="Team"),
    dict(Task="Research and begin implementing SOTA methods", Start='2024-03-10', Finish='2024-03-31', Owner="Team"),
    dict(Task="Write project progress report", Start='2024-03-26', Finish='2024-03-31', Owner="Team"),
    dict(Task="Fully implement sentiment analysis methods", Start='2024-03-31', Finish='2024-04-07', Owner="Team"),
    dict(Task="Review performance and revise approaches", Start='2024-04-07', Finish='2024-04-21', Owner="Team"),
    dict(Task="Write project report and create slide deck", Start='2024-04-21', Finish='2024-05-05', Owner="Team"),
    dict(Task="Deliver presentation and submit report", Start='2024-04-28', Finish='2024-05-05', Owner="Team")
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task",
                  color="Owner",
                  title="Timeline of CS 534 Artificial Intelligence Project")
fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
fig.show()