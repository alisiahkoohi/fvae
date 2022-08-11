# coding: utf-8

import obspy

from obspy.clients.fdsn.mass_downloader import (
    CircularDomain,
    Restrictions,
    MassDownloader,
)


# Rectangular domain
domain = CircularDomain(
    latitude=47.5798,
    longitude=-122.7994,
    minradius=0,
    maxradius=0.8,
)

# Define data temporal and meta restrictions.
restrictions = Restrictions(
    # Time limits
    starttime=obspy.UTCDateTime(2020, 1, 1),
    endtime=obspy.UTCDateTime(2021, 12, 31),
    # Chunk size in seconds
    chunklength_in_sec=86400,  # 1d
    # Considering the enormous amount of data associated with continuous
    # requests, you might want to limit the data based on SEED identifiers.
    # If the location code is specified, the location priority list is not
    # used; the same is true for the channel argument and priority list.
    network="CC",
    station="CARB",
    location="",
    channel="BHZ",
    # The typical use case for such a data set are noise correlations where
    # gaps are dealt with at a later stage.
    reject_channels_with_gaps=True,
    # Same is true with the minimum length. All data might be useful.
    minimum_length=0.0,
    # Guard against the same station having different names.
    minimum_interstation_distance_in_m=100.0,
)

# Download
import os
from pathlib import Path
path = Path(os.getcwd())

mdl = MassDownloader(providers=["IRIS"])
mdl.download(
    domain,
    restrictions,
    mseed_storage=str(path / 'waveform'),
    stationxml_storage=str(path / 'station'),
)
