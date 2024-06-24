import plotly.graph_objects as go
import datetime as dt
import numpy as np
import pandas as pd
import math as mh
from plotly.subplots import make_subplots

time_log = []
temp_log = []

with open("/home/jutsell/figure-generation/temp_log2.txt", "r") as filestream:
    for line in filestream:
        currentline = line.split(",")
        time_log.append(currentline[0])
        temp_log.append(currentline[1])

temp_log = [e.strip() for e in temp_log if e.strip()]

temp_log = [eval(i) for i in temp_log]
time_log = [eval(i) for i in time_log]

time_log_readable = []

for i in time_log:
    timestamp_with_ms = i
    dt1 = dt.datetime.fromtimestamp(timestamp_with_ms)
#   formatted_time = dt1.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    formatted_time = dt1.strftime('%Y-%m-%d %H:%M:%S.%f')
    time_log_readable.append(formatted_time)

fig = make_subplots(
    rows=3, cols=2,
    specs=[
           [{"colspan": 2}, None],
           [{}, {}],
           [{}, {}]],
    subplot_titles=("Last 24h (1 sample per min)","Last 7d (one sample per 30 min)", "Last 30d (one sample per hour)", "Last 365d (one sample per day)", "Since start (one sample per day)"))

fig.add_trace(go.Scatter(x=time_log_readable[-60*24:], y=temp_log[-60*24:]),
                 row=1, col=1)
fig.add_trace(go.Scatter(x=time_log_readable[-60*24*7::30], y=temp_log[-60*24*7::30]),
                 row=2, col=1)
fig.add_trace(go.Scatter(x=time_log_readable[-60*24*30::60], y=temp_log[-60*24*30::60]),
                 row=2, col=2)
fig.add_trace(go.Scatter(x=time_log_readable[-60*24*365::60*24], y=temp_log[-60*24*365::60*24]),
                 row=3, col=1)
fig.add_trace(go.Scatter(x=time_log_readable[1::60*24], y=temp_log[1::60*24]),
                 row=3, col=2)

fig.update_xaxes(title_text="Time")
fig.update_yaxes(title_text="Temperature [&deg;C]")

current_datetime = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

fig.update_layout(showlegend=False, title_text="E2 Battery Laboratory Temperature History. Current temperature: " + str(round(temp_log[-1],2)) + " &deg;C. Updated: " + str(current_datetime))

fig.write_html('/home/jutsell/figure-generation/second_figure.html', auto_open=False)
