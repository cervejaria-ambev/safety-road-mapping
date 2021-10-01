General Instructions
====================

Generating API token
--------------------

This project uses `openrouteservice
API <https://openrouteservice.org>`__ to plot maps and routes. So the
following steps are necessary at first:

1. Sign up on
   `openrouteservice.org <https://openrouteservice.org/dev/#/signup>`__
   to generate an API token;
2. Create a ``.env`` file with the following content:
   ``TOKEN=XXXXXXXXXXXXXXX``, where ``XXXXXXXXXXXXXXX`` is the token
   generated in the step before;

Accident road data
------------------

The accidents data used were extracted from the Polícia Rodoviária
Federal website. The notebook ``get_data.ipynb`` inside
``safety_road_mapping/notebooks`` folder is responsible to download and
extract the data used. If you want to directly download the files you
can `click
here <https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes>`__.

Roadmap
-------

-  The accidents data used comes just from federal police source, so
   there are some routes that don't receive score because they are state
   highways.
-  The routes subsections are not connected, once they are plotted
   individually in the map. Visually it can be interesting to connect
   them. (Is it possible or necessary?)

