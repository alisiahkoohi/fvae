#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:48:26 2022

@author : greg
@purpose:
@version:
"""

import os

from obspy import read_events
import pandas as pd
import xml.etree.ElementTree as ET
from facvae.utils import catalogsdir

DATA_DIR = catalogsdir()
quakeml_file = os.path.join(catalogsdir(),
                            "events_extended_multiorigin_v14_2023-01-01.xml")
# "events_extended_multiorigin_v11_2022-04-01.xml")


def get_quality_list(root):
    EventQuality = []
    for elem in root.iter(
            '{http://quakeml.org/xmlns/bed/1.2/mars}locationQuality'):
        EventQuality.append(elem.text[elem.text.find('#') + 1:len(elem.text)])

    return EventQuality


def get_df_from_quakeml2(quakeMLfilename):
    """
        Function which read the quakeML file
        ----
        INPUT:
            @quakeMLfilename: str - path to the quakeML file name

    """

    Tree = ET.parse(quakeMLfilename)
    root = Tree.getroot()

    EventQuality = get_quality_list(root)

    event = read_events(quakeMLfilename)
    event_list = []

    print(f"Longueur du fichier : {len(event)}")

    for i in range(len(event)):
        #print(event[i])
        event_dict = {}
        if len(event[i].event_descriptions) > 1:

            if str(event[i].event_descriptions[0].text)[0] in ["S", "T"] and \
                "Elysium" not in str(event[i].event_descriptions[0].text):

                event_dict['name'] = str(event[i].event_descriptions[0].text)
            else:
                event_dict['name'] = str(event[i].event_descriptions[1].text)
        else:
            event_dict['name'] = str(event[i].event_descriptions[0].text)

        event_dict['quality'] = EventQuality[i]

        quaketype = event[i].extra.type.value
        event_dict["type"] = quaketype[quaketype.find("#") + 1:]
        Origin = event[i].preferred_origin()
        event_dict['eventTime'] = Origin.time
        event_dict['distance'] = Origin.arrivals[0].distance
        event_dict['azimuth'] = Origin.arrivals[0].azimuth

        if "magnitudes" in event[i]:
            if len(event[i]['magnitudes']) > 0:
                mag_list = []
                for j in range(0, len(event[i]["magnitudes"])):
                    mag_list.append(event[i]["magnitudes"][j].mag)
                event_dict['magnitudes'] = mag_list
            else:
                event_dict['magnitudes'] = "Unkwown"
        else:
            event_dict['magnitudes'] = "Unkwown"

        pick_list = []
        if len(event[i].picks) > 1:
            for pk in event[i].picks:
                picks_dict = {}
                picks_dict["time"] = pk.time
                if 'time_errors' in pk:
                    picks_dict[
                        "lower_uncertainty"] = pk.time_errors.lower_uncertainty
                    picks_dict[
                        "upper_uncertainty"] = pk.time_errors.upper_uncertainty
                else:
                    picks_dict["lower_uncertainty"] = None
                    picks_dict["upper_uncertainty"] = None
                picks_dict["phase_hint"] = pk.phase_hint
                if pk.phase_hint == 'start':
                    event_dict["start_time"] = pk.time
                if pk.phase_hint == 'end':
                    event_dict["end_time"] = pk.time
                pick_list.append(picks_dict)
            event_dict["picks"] = pick_list
        event_list.append(event_dict)

    df_events = pd.DataFrame(event_list)
    return df_events


def main():
    quakeMLfilename = os.path.join(DATA_DIR, quakeml_file)
    df_events = get_df_from_quakeml2(quakeMLfilename)
    df_events.to_pickle(os.path.join(DATA_DIR, 'events_InSIght_v14.pkl'))


if __name__ == "__main__":
    main()

quakeMLfilename = os.path.join(DATA_DIR, quakeml_file)

df_events = get_df_from_quakeml2(quakeMLfilename)
