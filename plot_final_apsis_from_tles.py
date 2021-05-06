import requests, requests_cache  # https://requests-cache.readthedocs.io/en/latest/
from skyfield.api import Topos, EarthSatellite, load
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from spacetrack import SpaceTrackClient

requests_cache.install_cache('test_cache', backend='sqlite', expire_after=1800)

st = SpaceTrackClient('userid', 'password')
ts = load.timescale()

def chart_satellite_height(norad_cat_id=48275, name='CZ-5B R/B'):
    my_list = st.tle(norad_cat_id=norad_cat_id, limit=250, orderby='epoch desc', format='tle').splitlines()
    # my_list = st.tle(norad_cat_id=norad_cat_id, orderby='epoch asc', format='tle').splitlines()
    n = 2
    tles = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]
    prev = ['', '']
    x = []
    ya = []
    yp = []
    for tle in tles:
        if tle[1] == prev[1] and tle[0] == prev[0]:
            continue # ignore duplicate entries
        prev = tle.copy()
        satellite = EarthSatellite(tle[0], tle[1], '0 name', ts)
        epoch = satellite.epoch.utc_datetime()
        print(epoch)
        x.append(epoch)
        t = ts.utc(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, (0, 6000))  # epoch + a full orbit
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()
        yp.append(min(subpoint.elevation.km * 0.621371))
        ya.append(max(subpoint.elevation.km * 0.621371))


    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(16, 9), dpi=300)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %b %-m'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylim([0,max(ya)])

    plt.plot(x, ya, marker=".", markersize=10, label='apogee', color='xkcd:blue', linewidth=5, )
    plt.plot(x,yp, marker=".", markersize=10, label='perigee', color='xkcd:violet', linewidth=5, )
    plt.gcf().autofmt_xdate()
    plt.ylabel("altitude (miles)")
    plt.title(f"{name} ({epoch.strftime('%Y')})")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    chart_satellite_height(norad_cat_id=16609, name='Mir')
    chart_satellite_height(norad_cat_id=6633, name='Skylab')
    chart_satellite_height(norad_cat_id=13138, name='Salyut 7')
    chart_satellite_height(norad_cat_id=37872, name='Fobos-Grunt')
    chart_satellite_height(norad_cat_id=45601, name='CZ-5B R/B 2020')
    chart_satellite_height(norad_cat_id=48275, name='CZ-5B R/B 2021')
