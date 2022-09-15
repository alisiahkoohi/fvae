"""Probing the Mars Seismic Catalogue.
"""

from obspy.core.event import read_events
import os
import pickle

from facvae.utils import gitdir, datadir

# Path to the catalog file. Obtained from
# https://www.seis-insight.eu/static/mqs-catalogs/v11/events_extended_multiorigin_v11_2022-04-01.xml
CATALOG_FILENAME = os.path.join(datadir('catalogs'),
                        'events_extended_multiorigin_v11_2022-04-01.xml')
EVENTS_FILENAME = os.path.join(datadir('catalogs'),
                        'events_InSIght.pkl')

if __name__ == '__main__':
    # Read the catalog.
    catalog = read_events(CATALOG_FILENAME)

    # Write a summary of the catalog.
    with open(os.path.join(gitdir(), 'logs', 'events_summary.txt'), 'w') as f:
        f.write(catalog.__str__(print_all=True))

    # Take a look at a single event.
    with open(os.path.join(gitdir(), 'logs', 'event_attributes.txt'), 'w') as f:
        for var in catalog[16]:
            f.write(var + ': ' + str(getattr(catalog[0], var)) + '\n')

    # Extract an attribute from all events.
    attr = 'magnitudes'
    with open(os.path.join(gitdir(), 'logs', 'prob_attribute.txt'),
              'w') as f:
        f.write('Probed attribute: ' + attr + '\n')
        for event in catalog:
            f.write(str(getattr(event, attr)) + '\n')

    with open(EVENTS_FILENAME, 'rb') as f:
        events = pickle.load(f)
