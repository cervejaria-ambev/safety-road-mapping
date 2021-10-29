from folium import Map
from folium.features import GeoJson, GeoJsonTooltip
from folium.map import Icon, Marker
import numpy as np
from openrouteservice import client
from pandas.core.frame import DataFrame, Series
from pandas import Interval
from typeguard import typechecked
from typing import Union
import math
import pandas as pd
from geopy import distance
import unidecode
from colour import Color
import copy
from pathlib import Path
from dotenv import load_dotenv
import os


def generate_base_map(default_location: list = [-14, -50],
                      default_zoom_start: Union[int, float] = 4) -> Map:
    """
    Generates the basemap where the routes with their safety scores will be plotted.

    Parameters
    ----------
    default_location : list, optional
        Map's default location, by default [-14, -50]
    default_zoom_start : Union[int, float], optional
        Default zoom to be applied, by default 4

    Returns
    -------
    Map
        Map object to be used.
    """
    return Map(location=default_location, control_scale=True, prefer_canvas=True,
               zoom_start=default_zoom_start, tiles="cartodbpositron",)


@typechecked
class SafetyMap(object):
    def __init__(self, accidents_data_file_path: str, start_point: tuple, end_point: tuple,
                 basemap: Map, sub_section_dist: float = 5., env_path: str = '.env',
                 map_save_path: str = "./maps/safety_map.html", color_value: int = None,
                 origin_name: str = '', destination_name: str = ''):
        """
        Initializes some important variables

        Parameters
        ----------
        accidents_data_file_path : str
            Path where the accidents .csv file is located in the disk.
        start_point : tuple
            Route start point in the format: (longitude, latitude)
        end_point : tuple
            Route end point in the format: (longitude, latitude)
        basemap : Map
            Map where the routes will be plotted
        sub_section_dist : float, optional
            Length of each subsection in the route in km, by default 5.
        env_path : str, optional
            Path to .env file, default ".env"
        map_save_path : str, optional
            Path where the .html file with the route map will be saved, by default
            "./maps/safety_map.html"
        color_value : int, optional
            Color to use on the icons in the map. This is special useful when you are plotting
            more routes in the same map. By default None
            You have to pass an integer between 0 and 18 as a dictionary key:
            {0: 'red', 1: 'lightred', 2: 'darkblue', 3: 'orange', 4: 'darkgreen',
             5: 'cadetblue', 6: 'purple', 7: 'black', 8: 'gray', 9: 'lightblue',
             10: 'beige', 11: 'lightgray', 12: 'lightgreen', 13: 'blue', 14: 'pink',
             15: 'darkpurple', 16: 'green', 17: 'white', 18: 'darkred'}
        origin_name : str, optional
            Name given to the origin point, by default ""
        destination_name : str, optional
            Name given to the destination point, by default ""
        """
        color_dict = {0: 'red', 1: 'lightred', 2: 'darkblue', 3: 'orange', 4: 'darkgreen',
                      5: 'cadetblue', 6: 'purple', 7: 'black', 8: 'gray', 9: 'lightblue',
                      10: 'beige', 11: 'lightgray', 12: 'lightgreen', 13: 'blue', 14: 'pink',
                      15: 'darkpurple', 16: 'green', 17: 'white', 18: 'darkred'}
        dotenv_path = Path(env_path)
        load_dotenv(dotenv_path=dotenv_path)
        TOKEN = os.getenv('TOKEN')
        self.clnt = client.Client(key=TOKEN)
        self.base_map = basemap
        self.route = self._add_route_to_map(start_point, end_point)
        self.coor_df = self._gen_coordinates_df()
        self.accidents = self._treat_accidents_data(accidents_data_file_path)
        self.sub_section_dist = sub_section_dist
        self.map_save_path = map_save_path
        self.origin_name = origin_name
        self.destination_name = destination_name
        self.icon_color = color_dict.get(color_value)

    def _treat_accidents_data(self, path: str) -> DataFrame:
        """
        Method to open the csv file containing accidents information, treat the data and assign it
        to an attribute called accidents (DataFrame).

        Parameters
        ----------
        path : str
            The path where the file is located on the machine.

        Returns
        -------
        DataFrame
            Treated accidents DataFrame
        """
        dtype_dict = {'data_inversa': np.dtype('O'), 'dia_semana': np.dtype('O'),
                      'horario': np.dtype('O'), 'uf': np.dtype('O'), 'br': np.dtype('O'),
                      'km': np.dtype('O'), 'municipio': np.dtype('O'),
                      'causa_acidente': np.dtype('O'), 'tipo_acidente': np.dtype('O'),
                      'classificacao_acidente': np.dtype('O'), 'fase_dia': np.dtype('O'),
                      'sentido_via': np.dtype('O'), 'condicao_metereologica': np.dtype('O'),
                      'tipo_pista': np.dtype('O'), 'tracado_via': np.dtype('O'),
                      'pessoas': np.dtype('int64'), 'mortos': np.dtype('int64'),
                      'feridos_graves': np.dtype('int64'), 'latitude': np.dtype('O'),
                      'longitude': np.dtype('O')}
        self.accidents = pd.read_csv(path, encoding='latin1', sep=';', dtype=dtype_dict)
        self.accidents.loc[:, 'latitude'] = (self.accidents['latitude'].str.replace(',', '.')
                                             .astype('float'))
        self.accidents.loc[:, 'longitude'] = (self.accidents['longitude'].str.replace(',', '.')
                                              .astype('float'))
        l1 = list(self.accidents.query('longitude > 0').index)
        l2 = list(self.accidents.query('longitude < -75').index)
        self.accidents = self.accidents.drop(labels=l1+l2)
        self.accidents = self.accidents.drop_duplicates()
        self._filter_accident_data()
        return self.accidents

    def _add_route_to_map(self, start_point: tuple, end_point: tuple) -> dict:
        """
        Generates the route based on the start and end points.

        Parameters
        ----------
        start_point : tuple
            Start point format: (longitude, latitude)
        end_point : tuple
            End point format: (longitude, latitude)

        Returns
        -------
        dict
            A dictionary with route information to plot in a map
        """
        request_params = {
            'coordinates': [start_point, end_point],
            'format_out': 'geojson',
            'profile': 'driving-car',
            'instructions': 'false',
            'radiuses': [-1, -1],
            'preference': 'recommended',
        }

        self.route = self.clnt.directions(**request_params)
        self.route_distance = (self.route['features'][0]['properties']
                               ['summary']['distance'])
        self.trip_duration = (self.route['features'][0]['properties']
                              ['summary']['duration'])
        return self.route

    def format_duration(self) -> str:
        """
        Format trip duration that is in second for hours, minutes and seconds

        Returns
        -------
        str
            String with the trip duration formated: HH:mm:ss
        """
        if self.trip_duration is not None:
            duration_h = self.trip_duration / 3600
            hours = math.trunc(duration_h)
            duration_m = (duration_h - hours) * 60
            minutes = math.trunc(duration_m)
            seconds = math.trunc(((duration_m) - minutes) * 60)
            return f"{hours:02}:{minutes:02}:{seconds:02}"

    def _gen_coordinates_df(self) -> DataFrame:
        """
        Generate coordinates DataFrame based on route dictionary. This method gets the coordinates
        and extracts the latitude and longitude.
        It also calculates the distante between point in the route to generate the subsections and
        categorize them in groups. The coord_df contains the distance between coordinates to
        calculate the sections of the route.

        Returns
        -------
        DataFrame
            Coordinates DataFrame with the route's latitude and longitude, also it has the distance
            between the coordinate and the subsequent point. Based on the cumulative distance the
            route's subsections are created.
        """
        coor = self.route['features'][0]['geometry']['coordinates']
        self.coor_df = pd.DataFrame(coor).rename(columns={0: 'olong', 1: 'olat'})
        self.coor_df[['dlong', 'dlat']] = self.coor_df[['olong', 'olat']].shift(-1)
        self.coor_df['origin_tuple'] = list(self.coor_df[['olat', 'olong']]
                                            .itertuples(index=False, name=None))
        self.coor_df['destination_tuple'] = list(self.coor_df[['dlat', 'dlong']]
                                                 .itertuples(index=False, name=None))
        self.coor_df = self.coor_df.dropna()
        distance_list = []
        for _, row in self.coor_df.iterrows():
            origin = row.origin_tuple
            destination = row.destination_tuple
            distance_list.append(distance.distance(origin, destination).km)
        self.coor_df = self.coor_df.assign(route_dist=distance_list)
        self.coor_df = self.coor_df.assign(cum_sum=self.coor_df['route_dist']
                                           .cumsum())
        return self.coor_df

    def _gen_sections(self) -> DataFrame:
        """
        Method to create intervals with step `self.sub_section_dist`. Each group is one subsection
        of the route.

        Returns
        -------
        DataFrame
            sections DataFrame with the following information: coordinates' latitude and longitude,
            cum_sum_min (the minimum distance of that route section), cum_sum_max (the maximum
            distance of that route section), sections (the first km and the last km included in
            that section, origin (section first coordinate), destination (section last coordinate))
        """
        max_dis = self.coor_df['cum_sum'].max()
        interval = pd.interval_range(start=0, end=max_dis + self.sub_section_dist,
                                     freq=self.sub_section_dist, closed='left')
        self.coor_df['sections'] = pd.cut(self.coor_df['cum_sum'], bins=interval)
        coor_sum_min = (self.coor_df.groupby(by='sections').agg({'cum_sum': 'min'})
                        .reset_index().rename(columns={'cum_sum': 'cum_sum_min'}))
        coor_sum_max = (self.coor_df.groupby(by='sections').agg({'cum_sum': 'max'})
                        .reset_index().rename(columns={'cum_sum': 'cum_sum_max'}))
        cols_min = ['sections', 'cum_sum_min', 'olong', 'olat', 'origin_tuple']
        cols_max = ['cum_sum_max' if i == 'cum_sum_min' else i for i in cols_min]
        coor_sum_min = (coor_sum_min.merge(self.coor_df, left_on='cum_sum_min',
                                           right_on='cum_sum')
                        .rename(columns={'sections_x': 'sections'})[cols_min])
        coor_sum_max = (coor_sum_max.merge(self.coor_df, left_on='cum_sum_max',
                                           right_on='cum_sum')
                        .rename(columns={'sections_x': 'sections'})[cols_max])
        rename_dict = {'olong_x': 'olong',	'olat_x': 'olat', 'origin_tuple_x': 'origin',
                       'olong_y': 'dlong', 'olat_y': 'dlat', 'origin_tuple_y': 'destination'}
        return coor_sum_min.merge(coor_sum_max, on='sections').rename(columns=rename_dict)

    @staticmethod
    def _normalize_string(string: str) -> str:
        """
        Normalizes strings removing accentuation, lowering them and joining them with underline (_)

        Parameters
        ----------
        string : str
            String to be normalized

        Returns
        -------
        str
            Normalized string
        """
        string_no_accents = unidecode.unidecode(string)
        string_lower = string_no_accents.lower()
        string_without_space = string_lower.split(' ')
        return '_'.join(string_without_space)

    def _classes_accidents(self, accident: str) -> int:
        """
        Creates a score for the route's section. Those scores are arbitrary and can be tuned for
        what makes more sense

        Parameters
        ----------
        accident : str
            Accident class. 'sem_vitimas' when there are no victims; 'com_vitimas_feridas' when
            there are injured victims; 'com_vitimas_fatais' when there are fatal victims.

        Returns
        -------
        int
            The accident score based on its class

        Raises
        ------
        Exception
            Raises an exception if an unexpected class is passed as a parameter.
        """
        accident = self._normalize_string(accident)
        if accident == 'sem_vitimas':
            return 1
        elif accident == 'com_vitimas_feridas':
            return 5
        elif accident == 'com_vitimas_fatais':
            return 10
        else:
            raise Exception("Accident class doesn't mapped in the original dataset!")

    def _filter_accident_data(self) -> DataFrame:
        """
        Filters the accidents DataFrame based on the route coordinates. In other words, it gets the
        accidents points near the route only.

        Returns
        -------
        DataFrame
            Filtered accidents DataFrame
        """
        min_olong = np.round(self.coor_df['olong'].min(), 3)
        max_olong = np.round(self.coor_df['olong'].max(), 3)
        min_olat = np.round(self.coor_df['olat'].min(), 3)
        max_olat = np.round(self.coor_df['olat'].max(), 3)
        query = (f'({min_olat} <= latitude <= {max_olat}) and '
                 f'({min_olong} <= longitude <= {max_olong})')
        filtered = self.accidents.query(query)
        if not filtered.empty:
            self.accidents = filtered
        return self.accidents

    @staticmethod
    def _days_from_accident(df_date_col: Series) -> Series:
        """
        Calculates how many days has passed from the accident (based on the date of the last
        accident on dataset)

        Parameters
        ----------
        df_date_col : Series
            Accident dates column

        Returns
        -------
        Series
            Column with days from accident
        """
        max_date = df_date_col.max()
        return (np.datetime64(max_date) - pd.to_datetime(df_date_col)).apply(lambda x: x.days)

    @staticmethod
    def _haversine(Olat: float, Olon: float, Dlat: Series, Dlon: Series) -> Series:
        """
        Calculates haversine distance. For more information look at:
        https://en.wikipedia.org/wiki/Haversine_formula

        Parameters
        ----------
        Olat : float
            Origin latitude
        Olon : float
            Origin longitude
        Dlat : Series
            Destiny latitude
        Dlon : Series
            Destiny longitude

        Returns
        -------
        Series
            Distance Series
        """
        radius = 6371.  # km
        d_lat = np.radians(Dlat - Olat)
        d_lon = np.radians(Dlon - Olon)
        a = (np.sin(d_lat / 2.) * np.sin(d_lat / 2.) + np.cos(np.radians(Olat)) *
             np.cos(np.radians(Dlat)) * np.sin(d_lon / 2.) * np.sin(d_lon / 2.))
        c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1. - a))
        return radius * c

    def _rank_subsections(self, df: DataFrame, flag: str) -> DataFrame:
        """
        Generates the score for each route subsection.

        Parameters
        ----------
        df : DataFrame
            DataFrame with coordinate, subsections and distances
        flag : str
            Flag to indicate if df rows represent each point in the route or if they are the route's
            subsections. Possible values are: 'point' ou 'route'

        Returns
        -------
        DataFrame
            DataFrame with the scores for the route's sections

        Raises
        ------
        Exception
            If the flag is not set to 'point' or 'route'
        """
        last_val = int(df['sections'].values[-1].right)
        rank_df_list = []
        for i in range(0, last_val, int(self.sub_section_dist)):
            interval = Interval(float(i), float(i + 5.0), closed='left')
            filtered_route = self.route_df[self.route_df['sections'] == interval]
            rank_accidents = self.score_accidents.copy()
            distances_list = []
            for _, row in filtered_route.iterrows():
                distances = self._haversine(row['origin_tuple'][0], row['origin_tuple'][1],
                                            self.score_accidents.loc[:, 'latitude'],
                                            self.score_accidents.loc[:, 'longitude'])
                distances_list.append(distances)
            filtered_sections = df[df['sections'] == interval]
            if flag == 'point':
                rank_df = filtered_sections[['sections', 'origin', 'destination']]
            elif flag == 'route':
                rank_df = filtered_sections[['sections', 'origin_tuple']]
            else:
                raise Exception("The flag used is not a valid option!!!")
            distances_list.append(rank_accidents['score'])
            df_dist = pd.concat(distances_list, axis=1)
            rank = df_dist[(df_dist.iloc[:, :-1] <= 1).sum(axis=1) > 0]['score'].sum()
            rank_df = rank_df.assign(score=rank)
            rank_df_list.append(rank_df)
        return pd.concat(rank_df_list)

    def _getcolor(self, rank: float) -> str:
        """
        Generates the color for the subsection on the route based on its score.

        Parameters
        ----------
        rank : float
            Subsection's score

        Returns
        -------
        str
            Hexadecimal color or grey if the subsection has no score.
        """
        max_score = int(np.ceil(self.final_rank_sections['score'].max()))
        try:
            colors = list(Color('green').range_to(Color('red'), max_score))
        except ValueError:
            return 'grey'
        else:
            colors = [color.get_web() for color in colors]
            if rank == 0:
                return 'grey'
            else:
                return colors[int(rank)]

    def _plot_route_score(self):
        """
        Plots the subsections in the route on the map with different colors based on the the score
        of each subsection.
        """
        rank_json = copy.deepcopy(self.route)
        properties = rank_json['features'][0]['properties']
        rank_json['features'] = []
        last_val = int(self.sections_df['sections'].values[-1].right)
        p_type = 'Feature'
        for i in range(0, last_val, int(self.sub_section_dist)):
            interval = Interval(float(i), float(i) + 5.0, closed='left')
            subsection = self.final_rank_route[self.final_rank_route['sections'] == interval]
            coor_list = subsection['origin_tuple'].apply(lambda x: list(x[::-1])).to_list()
            bbox = self.route['bbox']
            id = str(i)
            rank_value = int(subsection['score'].unique()[0])
            color = subsection['score'].apply(self._getcolor).unique()[0]
            properties['score'] = rank_value
            properties['color'] = color
            append_dict = {'bbox': bbox, 'type': p_type, 'properties': properties,
                           'id': id, 'geometry': {}}
            append_dict['geometry']['type'] = self.route['features'][0]['geometry']['type']
            append_dict['geometry']['coordinates'] = coor_list
            rank_json['features'].append(copy.deepcopy(append_dict))
        GeoJson(data=rank_json, overlay=True, smooth_factor=2,
                style_function=lambda x: {'color': x['properties']['color'], 'weight': 5,
                                          'fillOpacity': 1},
                highlight_function=lambda x: {'weight': 10, 'color': x['properties']['color']},
                tooltip=GeoJsonTooltip(fields=['score'], aliases=["section risk's score: "],
                                       labels=True, sticky=True,
                                       toLocaleString=True)).add_to(self.base_map)
        self.base_map.save(self.map_save_path)

    def _calculate_final_score(self) -> float:
        """
        Calculates the route's final score. To do this the scores of each subsection are summed and
        them divided by the route distance in kilometers. This is a way to normalize the final
        score. So, if two routes have the same summed score, the smaller one will have the higher
        final score.

        Returns
        -------
        float
            The final score calculated as stated above.
        """
        sum_score = self.final_rank_sections['score'].sum()
        self.score = sum_score / (self.route_distance / 1000)
        return self.score

    def _plot_final_score(self):
        """
        Plots the final score on a marker on the map. To open the popup with the message, the user
        needs to click in the marker located approximately on the middle of the route.
        """
        origin_label = ': ' + self.origin_name if self.origin_name else ''
        destination_label = ': ' + self.destination_name if self.destination_name else ''
        begin_color = self.icon_color or 'green'
        end_color = self.icon_color or 'red'
        score_color = self.icon_color or 'blue'
        score = str(np.round(self.score, 2))
        popup = ('<h3 align="center" style="font-size:16px">Route final score: '
                 f'<b>{score}</b></h3>')
        tooltip = '<strong>Click here to see route score</strong>'
        middle_pos = int(len(self.final_rank_route) / 2)
        marker_pos = self.final_rank_route.loc[middle_pos, 'origin_tuple']
        begin = self.final_rank_route.loc[0, 'origin_tuple']
        begin_formated = tuple(map(lambda coor: round(coor, 3), begin))
        end = self.final_rank_route.loc[len(self.final_rank_route) - 1, 'origin_tuple']
        end_formated = tuple(map(lambda coor: round(coor, 3), end))
        tooltip_begin = ('<h3 align="center" style="font-size:14px">Route Begin Point'
                         f'<b>{origin_label}</b></h3>')
        tooltip_end = ('<h3 align="center" style="font-size:14px">Route End Point'
                       f'<b>{destination_label}</b></h3>')
        (Marker(location=begin, tooltip=tooltip_begin, popup=f'{begin_formated}',
                icon=Icon(color=begin_color, icon='fa-truck', prefix='fa')).add_to(self.base_map))
        (Marker(location=end, tooltip=tooltip_end, popup=f'{end_formated}',
                icon=Icon(color=end_color, icon='fa-truck', prefix='fa')).add_to(self.base_map))
        (Marker(location=marker_pos, tooltip=tooltip, popup=popup,
                icon=Icon(color=score_color, icon='stats')).add_to(self.base_map))

    def _calculate_score_weight(self):
        """
        Calculates the weight to multiply the class score based on how many days the accident
        occurred from the last date in the dataset. If the accident is recent the weight is near 1,
        if it occurred long time ago the weight is near 0.
        """
        W_max = self.score_accidents['days_from_accident'].max()
        W_min = self.score_accidents['days_from_accident'].min()
        self.score_accidents['W'] = ((W_max - self.score_accidents['days_from_accident']) /
                                     (W_max - W_min))
        self.score_accidents['score'] = self.score_accidents['classes'] * self.score_accidents['W']

    def path_risk_score(self, save_map: bool = False) -> DataFrame:
        """
        This method call the others above to generate the subsections, calculate the scores and plot
        all on the map

        Parameters
        ----------
        save_map : bool, optional
            If True, the map is save in .html format on the disk, by default False

        Returns
        -------
        DataFrame
            Final DataFrame with the score for each subsection on the route.
        """
        self.score_accidents = self.accidents[['data_inversa', 'latitude', 'longitude',
                                               'classificacao_acidente']]
        classes = self.score_accidents['classificacao_acidente'].apply(self._classes_accidents)
        self.score_accidents = (self.score_accidents.assign(classes=classes)
                                .drop(columns='classificacao_acidente'))
        days_from = self._days_from_accident(self.score_accidents['data_inversa'])
        self.score_accidents['days_from_accident'] = days_from
        self._calculate_score_weight()
        self.score_accidents['lat_long'] = (list(self.score_accidents[['latitude', 'longitude']]
                                                 .itertuples(index=False, name=None)))
        self.sections_df = self._gen_sections()
        self.route_df = self.coor_df[['origin_tuple', 'destination_tuple', 'sections']]
        self.final_rank_sections = self._rank_subsections(self.sections_df, flag='point')
        self.final_rank_route = self._rank_subsections(self.coor_df, flag='route')
        self._calculate_final_score()
        print(f'The final route score is {self.score:.2f}.')
        if save_map:
            print('Plotting route and final score on map...')
            self._plot_final_score()
            self._plot_route_score()
        return self.final_rank_sections


if __name__ == "__main__":
    import time

    t0 = time.time()
    # Extrema: -22.864969298862736, -46.35471817331918
    # Nova Rio: -22.864365417300693, -43.60680685910165
    inicio = (-46.35471817331918, -22.864969298862736)
    fim = (-43.60680685910165, -22.864365417300693)
    data_path = "./extract_data/accidents_final.csv"
    basemap = generate_base_map()
    s = SafetyMap(accidents_data_file_path=data_path, start_point=inicio, end_point=fim,
                  basemap=basemap, map_save_path="./maps/teste_com_dados2.html",
                  env_path='./.env')
    s.path_risk_score(save_map=True)
    t1 = time.time()
    print(f'Tempo necessário: {t1 - t0} segundos')
