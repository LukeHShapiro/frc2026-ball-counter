"""Tests for FCC ELS STA parsing — coordinates, azimuth, dates.

Includes validation against the known Rocket Lab WFF STA example:
  File 0659-EX-ST-2025, Call Sign WZ9XOM
  Location: ~126 deg azimuth from true north, within 10 km,
  centered near NL 37-49-55 (37.8319 N approx)
"""

import pytest
from datetime import datetime

from launch_detector.adapters.fcc_els import (
    parse_coordinates,
    parse_coordinates_standard,
    parse_coordinates_nl,
    parse_azimuth,
    parse_sta_dates,
    _dms_to_decimal,
    _haversine_km,
)


class TestDMSConversion:
    def test_north_positive(self):
        assert abs(_dms_to_decimal(37, 49, 55, "N") - 37.8319) < 0.001

    def test_south_negative(self):
        assert _dms_to_decimal(37, 49, 55, "S") < 0

    def test_west_negative(self):
        assert _dms_to_decimal(75, 27, 30, "W") < 0

    def test_east_positive(self):
        assert _dms_to_decimal(75, 27, 30, "E") > 0


class TestHaversine:
    def test_same_point(self):
        assert _haversine_km(37.83, -75.49, 37.83, -75.49) < 0.01

    def test_nearby_points(self):
        dist = _haversine_km(37.83, -75.49, 37.84, -75.49)
        assert 0.5 < dist < 2.0

    def test_far_points(self):
        dist = _haversine_km(37.83, -75.49, 28.57, -80.65)
        assert dist > 500


class TestCoordinateParsingStandard:
    def test_dms_format(self):
        text = "located at 37°49'55\"N 075°27'30\"W"
        coords = parse_coordinates_standard(text)
        assert coords is not None
        lat, lon = coords
        assert abs(lat - 37.8319) < 0.01
        assert abs(lon - (-75.4583)) < 0.01

    def test_dms_with_spaces(self):
        text = "centered at 37 49 55 N and 075 27 30 W"
        coords = parse_coordinates_standard(text)
        assert coords is not None

    def test_no_match(self):
        assert parse_coordinates_standard("no coordinates here") is None


class TestCoordinateParsingNL:
    def test_nl_wl_format(self):
        """Parse the exact format from the Rocket Lab WFF STA."""
        text = "centered near NL 37-49-55, WL 75-27-30"
        coords = parse_coordinates_nl(text)
        assert coords is not None
        lat, lon = coords
        assert abs(lat - 37.8319) < 0.01
        assert abs(lon - (-75.4583)) < 0.01

    def test_nl_wl_with_context(self):
        text = (
            "Stage 1 and Stage 2 at Wallops Island, VA at approximately "
            "126 degrees azimuth from true North, within 10 km, centered "
            "around coordinates NL 37-49-55 and WL 75-27-30"
        )
        coords = parse_coordinates_nl(text)
        assert coords is not None
        lat, lon = coords
        assert abs(lat - 37.8319) < 0.01

    def test_no_match(self):
        assert parse_coordinates_nl("some other text") is None


class TestCoordinateParsing:
    def test_standard_format(self):
        coords = parse_coordinates("37°49'55\"N 075°27'30\"W")
        assert coords is not None

    def test_nl_format_fallback(self):
        coords = parse_coordinates("NL 37-49-55, WL 75-27-30")
        assert coords is not None

    def test_no_coords(self):
        assert parse_coordinates("hello world") is None


class TestAzimuthParsing:
    def test_degrees_azimuth(self):
        text = "approximately 126 degrees azimuth from true North"
        az = parse_azimuth(text)
        assert az is not None
        assert abs(az - 126.0) < 0.1

    def test_deg_abbreviation(self):
        text = "at 126 deg azimuth"
        az = parse_azimuth(text)
        assert az is not None
        assert abs(az - 126.0) < 0.1

    def test_degree_symbol(self):
        text = "heading 126° azimuth"
        az = parse_azimuth(text)
        assert az is not None

    def test_no_azimuth(self):
        assert parse_azimuth("no heading info") is None


class TestDateParsing:
    def test_slash_format(self):
        text = "valid from 06/15/2025 to 07/15/2025"
        dates = parse_sta_dates(text)
        assert dates is not None
        start, end = dates
        assert start == datetime(2025, 6, 15)
        assert end == datetime(2025, 7, 15)

    def test_through_format(self):
        text = "period 01/01/2025 through 02/28/2025"
        dates = parse_sta_dates(text)
        assert dates is not None

    def test_no_dates(self):
        assert parse_sta_dates("no dates here") is None


class TestRocketLabSTAExample:
    """Validate parsing against the known Rocket Lab WFF STA.

    Reference: File 0659-EX-ST-2025, Call Sign WZ9XOM
    - Location: NL 37-49-55 (lat ~37.832)
    - Azimuth: ~126 degrees from true north
    - Site: Wallops Island, VA
    - Within 10 km of center
    """

    STA_TEXT = (
        "Rocket Lab USA, Inc. has been granted Experimental Special Temporary "
        "Authorization with File Number 0659-EX-ST-2025 and Call Sign WZ9XOM. "
        "Stage 1 and Stage 2 operations at Wallops Island, VA at approximately "
        "126 degrees azimuth from true North, within 10 km, centered around "
        "coordinates NL 37-49-55 and WL 75-27-30. "
        "Frequency band 2200-2300 MHz with notches on 2257 MHz and 2280.5 MHz. "
        "The frequency band 2025-2110 MHz is not authorized for use. "
        "Valid from 03/01/2025 to 04/30/2025."
    )

    def test_coordinates_parsed(self):
        coords = parse_coordinates(self.STA_TEXT)
        assert coords is not None
        lat, lon = coords
        assert abs(lat - 37.8319) < 0.01

    def test_near_wallops(self):
        coords = parse_coordinates(self.STA_TEXT)
        assert coords is not None
        lat, lon = coords
        dist = _haversine_km(lat, lon, 37.8319, -75.4877)
        assert dist < 15.0

    def test_azimuth_parsed(self):
        az = parse_azimuth(self.STA_TEXT)
        assert az is not None
        assert abs(az - 126.0) < 0.5

    def test_dates_parsed(self):
        dates = parse_sta_dates(self.STA_TEXT)
        assert dates is not None
        start, end = dates
        assert start == datetime(2025, 3, 1)
        assert end == datetime(2025, 4, 30)

    def test_all_fields_present(self):
        """Integration: all fields parse from the full STA text."""
        coords = parse_coordinates(self.STA_TEXT)
        az = parse_azimuth(self.STA_TEXT)
        dates = parse_sta_dates(self.STA_TEXT)
        assert coords is not None
        assert az is not None
        assert dates is not None
