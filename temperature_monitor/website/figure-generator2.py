import plotly.graph_objects as go
import datetime as dt
import pandas as pd
import sys as sy
from plotly.subplots import make_subplots

def filter_data(df, debug_timestamp, time_delta):
    filtered_df = df[df['datetime'] > (debug_timestamp - time_delta)]
    return filtered_df


def downsample_data(df, rule):
    # This creates a new DataFrame for downsampling, original df remains unchanged
    df_copy = df.copy()  # Create a copy to ensure original df is not modified
    if not df_copy.index.equals(df_copy['datetime']):
        df_copy.set_index('datetime', inplace=True)
    return df_copy.resample(rule).mean().reset_index()


def main():
    df = pd.read_csv(sy.argv[1], names=['timestamp', 'temperature'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('Europe/Stockholm')

    current_timestamp = pd.Timestamp.now(tz='Europe/Stockholm')
    current_datetime = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}]],
        subplot_titles=("Last 24h (1 sample per min)",
                        "Last 7d (one sample per 30 min)",
                        "Last 30d (one sample per hour)",
                        "Last 365d (one sample per day)",
                        "Since start (one sample per day)"))

    filtered_df = filter_data(df, current_timestamp, dt.timedelta(hours=24))
    fig.add_trace(go.Scatter(x=filtered_df.datetime, y=filtered_df.temperature),
                  row=1, col=1)

    filtered_df = filter_data(df, current_timestamp, dt.timedelta(days=7))
    downsampled_df = downsample_data(filtered_df, '30min')
    fig.add_trace(go.Scatter(x=downsampled_df.datetime, y=downsampled_df.temperature),
                  row=2, col=1)

    filtered_df = filter_data(df, current_timestamp, dt.timedelta(days=30))
    downsampled_df = downsample_data(filtered_df, '1h')
    fig.add_trace(go.Scatter(x=downsampled_df.datetime, y=downsampled_df.temperature),
                  row=2, col=2)

    filtered_df = filter_data(df, current_timestamp, dt.timedelta(days=365))
    downsampled_df = downsample_data(filtered_df, '1D')
    fig.add_trace(go.Scatter(x=downsampled_df.datetime, y=downsampled_df.temperature),
                  row=3, col=1)

    downsampled_df = downsample_data(df, '1D')
    fig.add_trace(go.Scatter(x=downsampled_df.datetime, y=downsampled_df.temperature), row=3, col=2)

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Temperature [\N{DEGREE SIGN}C]")

    fig_title = (f"<b>E2 Battery Laboratory Temperature History</b>. "
                 f"Current temperature: {round(df.temperature.iloc[-1], 2)} \N{DEGREE SIGN}C. "
                 f"Updated: {current_datetime}")
    fig.update_layout(showlegend=False, title_text=fig_title)
    fig.write_html(sy.argv[2], auto_open=False)


if __name__ == '__main__':
    main()

