# import matplotlib.dates as mdates
import pandas as pd
from config import USERID, PASSWORD
from datetime import datetime
from pprint import pprint
from skyfield.api import Topos, EarthSatellite, load
from spacetrack import SpaceTrackClient
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import simple_cache
import numpy as np
import pytz
import requests, requests_cache  # https://requests-cache.readthedocs.io/en/latest/

requests_cache.install_cache('test_cache', backend='sqlite', expire_after=1800)

st = SpaceTrackClient(USERID, PASSWORD)
ts = load.timescale()


def chart_satellite_height(norad_cat_ids, title, deorbitdatetimes=[]):
    epoch1, x1, ya1, yp1, yv1 = _calculate_apsis(norad_cat_ids[0], seconds=6000)
    epoch2, x2, ya2, yp2, yv2 = _calculate_apsis(norad_cat_ids[1], seconds=6000)
    print(epoch1, epoch2)

    fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(16, 9), dpi=300)
    ax1.spines["top"].set_visible(False)

    ax1.set_ylim([0, max(max(ya2), max(ya1))])
    plt.gcf().autofmt_xdate()

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'xkcd:green'

    # plot Y axeses
    plt.setp(ax1.get_xticklabels(), fontsize=16)
    plt.setp(ax1.get_yticklabels(), fontsize=16)

    ax1.annotate(f'datasource: SpaceTrack.com as of {epoch.strftime("%Y-%m-%d %H:%M:%SZ")}', xy=(1, -.2),
                xycoords='axes fraction', fontsize=9,
                horizontalalignment='right', verticalalignment='bottom')

    # 2021-05-09 02:04
    deorbitdate2021=datetime(2021,5,9,2,04,tzinfo=pytz.UTC)

    daysbefore1 = [(x-x1[-1]).total_seconds()/86400.0 for x in x1]
    daysbefore2 = [(x-deorbitdate2021).total_seconds()/86400.0 for x in x2]

    ax1.plot(daysbefore2, ya2, "-", markersize=10, label='2021 apogee', color='xkcd:royal blue', linewidth=5, )
    ax1.plot(daysbefore2, yp2, "-", markersize=10, label='2021 perigee', color='xkcd:violet', linewidth=5, )

    ax1.plot(daysbefore1, ya1, "--", markersize=10, label='2020 apogee', color='xkcd:baby blue', linewidth=5 )
    ax1.plot(daysbefore1, yp1, "--", markersize=10, label='2020 perigee', color='xkcd:lilac', linewidth=5 )

    # ax1.plot(daysbefore1, yv1, "-", markersize=10, label='2021 velocity', color='xkcd:blue green', linewidth=5 )
    # ax1.plot(daysbefore2, yv2, "--", markersize=10, label='2020 velocity', color='xkcd:seafoam', linewidth=5 )
    # pprint(yv1)
    # pprint(yv2)
    # pprint(daysbefore2)
    # ax2.set_ylim([0, max(yv1)])
    print(daysbefore2[-1])
    ax1.set_xlabel("days before deorbit", fontsize=12)
    ax1.set_ylabel("altitude (miles)", fontsize=12)
    # ax2.set_ylabel('orbital velocity (km/s)', fontsize=10)  # we already handled the x-label with ax1
    # ax2.tick_params(axis='y')

    plt.title(f"{title}", fontsize=20)
    plt.legend(fontsize=16)
    plt.show()

def _calculate_apsis(norad_cat_id, seconds=6000, orderby='epoch asc'):
    my_list = st.tle(norad_cat_id=norad_cat_id, limit=500, orderby=orderby, format='tle').splitlines()
    # my_list = st.tle(norad_cat_id=norad_cat_id, orderby='epoch asc', format='tle').splitlines()
    n = 2
    tles = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]
    prev = ['', '']
    x = []
    ya = []
    yp = []
    yv = []
    for tle in tqdm(tles):
        if tle[1] == prev[1] and tle[0] == prev[0]:
            continue  # ignore duplicate entries
        prev = tle.copy()
        satellite = EarthSatellite(tle[0], tle[1], '0 name', ts)
        epoch = satellite.epoch.utc_datetime()

        vx, vy, vz = satellite.at(satellite.epoch).velocity.km_per_s
        speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        t = ts.now()
        x.append(epoch)
        t = ts.utc(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, range(0, seconds))  # epoch + a full orbit
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()
        yp.append(min(subpoint.elevation.km * 0.621371))
        ya.append(max(subpoint.elevation.km * 0.621371))
        yv.append(speed)
    return epoch, x, ya, yp, yv

@simple_cache.cache_it(filename="calculate_positions_last_orbit.cache", ttl=9999)
def calculate_positions_last_orbit(norad_cat_id, orderby='epoch desc', window=90):
    tle = st.tle(norad_cat_id=norad_cat_id, limit=1, orderby=orderby, format='tle').splitlines()
    satellite = EarthSatellite(tle[0], tle[1], '0 name', ts)
    epoch = satellite.epoch.utc_datetime()
    t = ts.utc(epoch.year, epoch.month, epoch.day, epoch.hour, range(epoch.minute-window, epoch.minute+window))
    geocentric = satellite.at(t)
    subpoints = geocentric.subpoint()
    dts = t.utc_datetime()
    df = pd.DataFrame(list(zip (dts, subpoints.latitude.degrees, subpoints.longitude.degrees)),
                      columns=['date', 'latitude', 'longitude'])
    return epoch, df

def mapit(objects):
    fig = go.Figure()
    for object in objects:
        df = object['df']
        finaldtstr = object['finaldtstr']
        df['name'] = object['name']
        print(f"{object['name']} {len(df)}")
        df['dtstr'] = df.apply(lambda row: row.date.isoformat(), axis=1)
        if finaldtstr is not None:
            df = df[df.dtstr < finaldtstr].copy()
            if 'minutes' in object.keys():
                minutes=object['minutes']
            else:
                minutes=10
        else:
            object['color']='blue'
            minutes=len(df)
        flight_paths = []
        df['start_lon'] = df['longitude'].shift(-1)
        df['start_lat'] = df['latitude'].shift(-1)
        start=len(df)-minutes
        if start < 0:
            start=0
        for i in range(start, len(df)):
            fig.add_trace(
                go.Scattergeo(
                    # locationmode='USA-states',
                    hoverinfo='text',
                    text=f"{object['name']}<br>{df['dtstr'][i]}<br>{df['latitude'][i]:.2f},{df['longitude'][i]:.2f}",
                    lon=[df['start_lon'][i], df['longitude'][i]],
                    lat=[df['start_lat'][i], df['latitude'][i]],
                    mode='lines',
                    line=dict(width=3, color=object['color']),
                    # opacity=float(df['cnt'][i]) / float(df['cnt'].max()),
                )
            )

        fig.update_layout(
            # title_text = 'Feb. 2011 American Airline flight paths<br>(Hover for airport names)',
            showlegend=False,
            geo=dict(
                scope='world',
                projection_type='robinson',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(0, 204, 204)',
                showocean=True, oceancolor="LightBlue",

            ),
        )

    fig.show()


if __name__ == '__main__':
    # plot heights of CZ-5B R/B in 2020 and 2021
    chart_satellite_height([45601, 48275], 'CZ-5B R/B')

    #build PlotLy map of large objects deorbit locations
    objects = []
    epoch, df = calculate_positions_last_orbit(13138)
    objects.append({'df': df, 'finaldtstr': '1991-02-07T03:47:00+00:00',
                    'color': 'red', 'minutes': 8,
                    'name': 'Salyut 7'})

    epoch, df = calculate_positions_last_orbit(16609)
    objects.append({'df': df, 'finaldtstr': '2001-03-23T05:56:00+00:00',
                    'color': 'green',
                    'name': 'Mir'})

    epoch, df = calculate_positions_last_orbit(6633, window=180)
    print(df)
    objects.append({'df': df, 'finaldtstr': "1979-07-04T19:42:00+00:00",
                    'color': 'green',
                    'name': 'Skylab', 'minutes': 8})

    epoch, df = calculate_positions_last_orbit(37820, window=3660)
    objects.append({'df': df, 'finaldtstr': '2018-04-02T00:17:00+00:00',
                    'color': 'green',
                    'name': 'Tiangong-1'})

    epoch, df = calculate_positions_last_orbit(45601, window=180)
    objects.append({'df': df, 'finaldtstr': '2020-05-11T15:39:00+00:00',
                    'color': 'red',
                    'name': 'CZ-5B R/B (2020)'})

    epoch, df = calculate_positions_last_orbit(37872, window=180)
    objects.append({'df': df, 'finaldtstr': '2012-01-15T17:45:00+00:00',
                    'color': 'red', 'minutes': 8,
                    'name': 'Fobos-Grunt'})
    mapit(objects)
