Module safety_road_mapping.safety
=================================

Functions
---------

    
`generate_base_map(default_location: list = [-14, -50], default_zoom_start: Union[int, float] = 4) ‑> folium.folium.Map`
:   Generates the basemap where the routes with their safety scores will be plotted.
    
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

Classes
-------

`SafetyMap(accidents_data_file_path: str, start_point: tuple, end_point: tuple, basemap: folium.folium.Map, sub_section_dist: float = 5.0, env_path: str = '.env', map_save_path: str = './maps/safety_map.html', color_value: int = None, origin_name: str = '', destination_name: str = '')`
:   Initializes some important variables
    
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

    ### Static methods

    `_days_from_accident(df_date_col: pandas.core.series.Series) ‑> pandas.core.series.Series`
    :   Calculates how many days has passed from the accident (based on the date of the last
        accident on dataset)
        
        Parameters
        ----------
        df_date_col : Series
            Accident dates column
        
        Returns
        -------
        Series
            Column with days from accident

    `_haversine(Olat: float, Olon: float, Dlat: pandas.core.series.Series, Dlon: pandas.core.series.Series) ‑> pandas.core.series.Series`
    :   Calculates haversine distance. For more information look at:
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

    `_normalize_string(string: str) ‑> str`
    :   Normalizes strings removing accentuation, lowering them and joining them with underline (_)
        
        Parameters
        ----------
        string : str
            String to be normalized
        
        Returns
        -------
        str
            Normalized string

    ### Methods

    `_add_route_to_map(self, start_point: tuple, end_point: tuple) ‑> dict`
    :   Generates the route based on the start and end points.
        
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

    `_calculate_final_score(self) ‑> float`
    :   Calculates the route's final score. To do this the scores of each subsection are summed and
        them divided by the route distance in kilometers. This is a way to normalize the final
        score. So, if two routes have the same summed score, the smaller one will have the higher
        final score.
        
        Returns
        -------
        float
            The final score calculated as stated above.

    `_calculate_score_weight(self)`
    :   Calculates the weight to multiply the class score based on how many days the accident
        occurred from the last date in the dataset. If the accident is recent the weight is near 1,
        if it occurred long time ago the weight is near 0.

    `_classes_accidents(self, accident: str) ‑> int`
    :   Creates a score for the route's section. Those scores are arbitrary and can be tuned for what
        makes more sense
        
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

    `_filter_accident_data(self) ‑> pandas.core.frame.DataFrame`
    :   Filters the accidents DataFrame based on the route coordinates. In other words, it gets the
        accidents points near the route only.
        
        Returns
        -------
        DataFrame
            Filtered accidents DataFrame

    `_gen_coordinates_df(self) ‑> pandas.core.frame.DataFrame`
    :   Generate coordinates DataFrame based on route dictionary. This method gets the coordinates
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

    `_gen_sections(self) ‑> pandas.core.frame.DataFrame`
    :   Method to create intervals with step `self.sub_section_dist`. Each group is one subsection of the
        route.
        
        Returns
        -------
        DataFrame
            sections DataFrame with the following information: coordinates' latitude and longitude,
            cum_sum_min (the minimum distance of that route section), cum_sum_max (the maximum
            distance of that route section), sections (the first km and the last km included in that section,
            origin (section first coordinate), destination (section last coordinate))

    `_getcolor(self, rank: float) ‑> str`
    :   Generates the color for the subsection on the route based on its score.
        
        Parameters
        ----------
        rank : float
            Subsection's score
        
        Returns
        -------
        str
            Hexadecimal color or grey if the subsection has no score.

    `_plot_final_score(self)`
    :   Plots the final score on a marker on the map. To open the popup with the message, the user
        needs to click in the marker located approximately on the middle of the route.

    `_plot_route_score(self)`
    :   Plots the subsections in the route on the map with different colors based on the the score
        of each subsection.

    `_rank_subsections(self, df: pandas.core.frame.DataFrame, flag: str) ‑> pandas.core.frame.DataFrame`
    :   Generates the score for each route subsection.
        
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

    `_treat_accidents_data(self, path: str) ‑> pandas.core.frame.DataFrame`
    :   Method to open the csv file containing accidents information, treat the data and assign it
        to an attribute called accidents (DataFrame).
        
        Parameters
        ----------
        path : str
            The path where the file is located on the machine.
        
        Returns
        -------
        DataFrame
            Treated accidents DataFrame

    `format_duration(self) ‑> str`
    :   Format trip duration that is in second for hours, minutes and seconds
        
        Returns
        -------
        str
            String with the trip duration formated: HH:mm:ss

    `path_risk_score(self, save_map: bool = False) ‑> pandas.core.frame.DataFrame`
    :   This method call the others above to generate the subsections, calculate the scores and plot
        all on the map
        
        Parameters
        ----------
        save_map : bool, optional
            If True, the map is save in .html format on the disk, by default False
        
        Returns
        -------
        DataFrame
            Final DataFrame with the score for each subsection on the route.